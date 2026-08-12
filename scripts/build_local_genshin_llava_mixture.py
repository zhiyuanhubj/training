#!/usr/bin/env python3
"""Build a node-local canonical Parquet root for Genshin + LLaVA.

The downloaded LLaVA repository is already split into canonical subdirectories.
Genshin arrives as a flat directory of Parquet shards.  This script creates a
small symlink-only union root, appends Genshin's row count to the canonical
index, writes a validation shard, and emits the mixture YAML consumed by
``lib_mixture.py``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import yaml


def sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def parquet_rows(path: Path) -> int:
    return pq.ParquetFile(path).metadata.num_rows


def safe_symlink(source: Path, destination: Path) -> None:
    source = source.resolve()
    if destination.is_symlink():
        if destination.resolve() != source:
            raise RuntimeError(
                f"refusing to replace symlink {destination} -> {destination.resolve()}"
            )
        return
    if destination.exists():
        raise RuntimeError(f"refusing to replace existing path: {destination}")
    destination.symlink_to(source, target_is_directory=source.is_dir())


def first_rows(files: list[Path], count: int) -> pa.Table:
    batches = []
    # Prefer one row from each source shard so the tiny validation set is not
    # dominated by one player/session.
    for path in files:
        iterator = pq.ParquetFile(path).iter_batches(batch_size=1)
        try:
            batches.append(next(iterator))
        except StopIteration:
            continue
        if len(batches) == count:
            return pa.Table.from_batches(batches)
    raise RuntimeError(f"Genshin contains fewer than {count} non-empty shards")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--llava-root", required=True)
    parser.add_argument("--genshin-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--genshin-passes", type=float, default=3.0)
    parser.add_argument("--llava-passes", type=float, default=0.162)
    parser.add_argument("--val-rows", type=int, default=32)
    parser.add_argument("--workers", type=int, default=32)
    args = parser.parse_args()

    llava_root = Path(args.llava_root).resolve()
    genshin_root = Path(args.genshin_root).resolve()
    output_root = Path(args.output_root).resolve()
    union_root = output_root / "parquet"
    val_path = output_root / "validation" / "genshin-val.parquet"
    yaml_path = output_root / "genshin-llava-128k.local.yaml"
    union_root.mkdir(parents=True, exist_ok=True)
    val_path.parent.mkdir(parents=True, exist_ok=True)

    llava_index_path = llava_root / "canonical_index.json"
    if not llava_index_path.is_file():
        raise RuntimeError(f"missing LLaVA canonical index: {llava_index_path}")
    llava_index = json.loads(llava_index_path.read_text(encoding="utf-8"))
    llava_items = llava_index.get("datasets") or []
    canonical_names = [
        item.get("canonical_name")
        for item in llava_items
        if isinstance(item.get("canonical_name"), str)
    ]
    if not canonical_names:
        raise RuntimeError("LLaVA canonical index contains no datasets")
    for name in canonical_names:
        source = llava_root / name
        if not source.is_dir():
            raise RuntimeError(f"canonical LLaVA directory is missing: {source}")
        safe_symlink(source, union_root / name)

    genshin_files = sorted(genshin_root.glob("*.parquet"))
    if not genshin_files:
        raise RuntimeError(f"no Genshin Parquet files found under {genshin_root}")
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        genshin_rows = sum(pool.map(parquet_rows, genshin_files))
    safe_symlink(genshin_root, union_root / "genshin")

    union_index = dict(llava_index)
    union_items = list(llava_items)
    union_items = [
        item for item in union_items if item.get("canonical_name") != "genshin"
    ]
    union_items.append(
        {
            "canonical_name": "genshin",
            "stats": {
                "output_rows": genshin_rows,
                "source": "yuanshen-bc-formatted-sl135-encrypted",
            },
        }
    )
    union_index["datasets"] = union_items
    (union_root / "canonical_index.json").write_text(
        json.dumps(union_index, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    genshin_sha_manifest = genshin_root.parent / "genshin-encrypted" / "SHA256SUMS"
    source_manifest = {
        "version": 1,
        "llava": {
            "root": str(llava_root),
            "canonical_index_sha256": sha256(llava_index_path),
            "manifest_sha256": sha256(llava_root / "manifest.jsonl"),
            "run_summary_sha256": sha256(llava_root / "run_summary.json"),
        },
        "genshin": {
            "root": str(genshin_root),
            "archive_sha256sums_sha256": sha256(genshin_sha_manifest),
            "parquet_files": len(genshin_files),
            "rows": genshin_rows,
        },
    }
    (union_root / "source_manifest.json").write_text(
        json.dumps(source_manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    if not val_path.is_file():
        table = first_rows(genshin_files, args.val_rows)
        pq.write_table(table, val_path, compression="zstd")

    config = {
        "version": 1,
        "format": "parquet",
        "root": os.fspath(union_root),
        "val": os.fspath(val_path),
        "download": "never",
        "loader": {
            "streaming": True,
            "dataset_shuffle": True,
            "shuffle_buffer_size": 10000,
            "dataset_mix": "interleave",
            "interleave_unit": "subdataset",
            "interleave_weight": "sample_count",
            "stopping_strategy": "all_exhausted_without_replacement",
            "enable_channel_loss": True,
            "channel_columns": {
                "conversations": "messages",
                "game": "channel",
                "data_source": "channel",
            },
            "add_genshin_special_tokens": True,
        },
        "datasets": {
            "genshin": {"path": "genshin", "passes": args.genshin_passes},
            **{
                name: {"path": name, "passes": args.llava_passes}
                for name in canonical_names
            },
        },
    }
    yaml_path.write_text(
        yaml.safe_dump(config, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )

    llava_rows = sum(
        int((item.get("stats") or {}).get("output_rows") or 0)
        for item in llava_items
    )
    print(f"union_root={union_root}")
    print(f"mixture_yaml={yaml_path}")
    print(f"validation={val_path}")
    print(f"genshin_files={len(genshin_files)} genshin_rows={genshin_rows}")
    print(f"llava_datasets={len(canonical_names)} llava_rows={llava_rows}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
