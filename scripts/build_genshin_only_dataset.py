#!/usr/bin/env python3
"""Prepare a leakage-free, node-local Genshin-only Parquet dataset.

The encrypted Hugging Face archive already contains trainable Parquet shards.
This script validates every footer, reserves one or more complete shards for
validation, creates a symlink-only training view, and writes the canonical
index and mixture YAML consumed by the 128K launcher.
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


SOURCE_REPO = "thomaslee1818/yuanshen-bc-formatted-sl135-encrypted"


def sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def inspect_parquet(path: Path) -> tuple[Path, int, tuple[str, ...]]:
    parquet = pq.ParquetFile(path)
    return path, parquet.metadata.num_rows, tuple(parquet.schema_arrow.names)


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
    destination.symlink_to(source)


def validation_rows(files: list[Path], count: int) -> pa.Table:
    batches: list[pa.RecordBatch] = []
    remaining = count
    for path in files:
        for batch in pq.ParquetFile(path).iter_batches(batch_size=remaining):
            if len(batch) > remaining:
                batch = batch.slice(0, remaining)
            batches.append(batch)
            remaining -= len(batch)
            if remaining == 0:
                return pa.Table.from_batches(batches)
    raise RuntimeError(f"held-out shards contain fewer than {count} rows")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--genshin-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--passes", type=float, default=3.0)
    parser.add_argument("--val-rows", type=int, default=32)
    parser.add_argument(
        "--holdout-shards",
        type=int,
        default=0,
        help="complete source shards reserved for validation; 0 selects the minimum automatically",
    )
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--archive-manifest")
    parser.add_argument("--overwrite-validation", action="store_true")
    args = parser.parse_args()

    if args.passes <= 0:
        parser.error("--passes must be positive")
    if args.val_rows <= 0:
        parser.error("--val-rows must be positive")
    if args.holdout_shards < 0:
        parser.error("--holdout-shards cannot be negative")

    genshin_root = Path(args.genshin_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    parquet_root = output_root / "parquet"
    train_root = parquet_root / "genshin"
    validation_path = output_root / "validation" / "genshin-val.parquet"
    yaml_path = output_root / "genshin-only-128k.local.yaml"

    files = sorted(genshin_root.glob("*.parquet"))
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        inspected = list(pool.map(inspect_parquet, files))
    empty = [str(path) for path, rows, _columns in inspected if rows <= 0]
    if empty:
        raise RuntimeError(f"empty Parquet shards: {empty[:10]}")
    required = {"images"}
    invalid = [
        str(path)
        for path, _rows, columns in inspected
        if not required.issubset(columns)
        or not ({"messages", "conversations"} & set(columns))
    ]
    if invalid:
        raise RuntimeError(f"unexpected Genshin schema: {invalid[:10]}")

    rows_by_path = {path: rows for path, rows, _columns in inspected}
    holdout_shards = args.holdout_shards
    if holdout_shards == 0:
        accumulated = 0
        for holdout_shards, path in enumerate(files, start=1):
            accumulated += rows_by_path[path]
            if accumulated >= args.val_rows:
                break
    if len(files) <= holdout_shards:
        raise RuntimeError(
            f"need more than {holdout_shards} Parquet shards under {genshin_root}"
        )
    held_out = files[:holdout_shards]
    train_files = files[holdout_shards:]
    train_rows = sum(rows_by_path[path] for path in train_files)
    held_out_rows = sum(rows_by_path[path] for path in held_out)

    train_root.mkdir(parents=True, exist_ok=True)
    validation_path.parent.mkdir(parents=True, exist_ok=True)
    for source in train_files:
        safe_symlink(source, train_root / source.name)

    expected_names = {path.name for path in train_files}
    unexpected = [
        path.name
        for path in train_root.glob("*.parquet")
        if path.name not in expected_names
    ]
    if unexpected:
        raise RuntimeError(
            "training view contains stale/unexpected shards; use a fresh output root: "
            + ", ".join(unexpected[:10])
        )

    if args.overwrite_validation or not validation_path.is_file():
        table = validation_rows(held_out, args.val_rows)
        temporary = validation_path.with_suffix(
            validation_path.suffix + f".tmp.{os.getpid()}"
        )
        pq.write_table(table, temporary, compression="zstd")
        temporary.replace(validation_path)
    actual_val_rows = pq.ParquetFile(validation_path).metadata.num_rows
    if actual_val_rows != args.val_rows:
        raise RuntimeError(
            f"validation row mismatch: expected={args.val_rows} actual={actual_val_rows}"
        )

    canonical_index = {
        "version": 1,
        "datasets": [
            {
                "canonical_name": "genshin",
                "stats": {
                    "output_rows": train_rows,
                    "source_output_rows": train_rows + held_out_rows,
                    "held_out_rows": held_out_rows,
                    "parquet_files": len(train_files),
                },
            }
        ],
    }
    (parquet_root / "canonical_index.json").write_text(
        json.dumps(canonical_index, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    archive_manifest = (
        Path(args.archive_manifest).expanduser().resolve()
        if args.archive_manifest
        else genshin_root.parent / "genshin-encrypted" / "SHA256SUMS"
    )
    source_manifest = {
        "version": 1,
        "source_repo": SOURCE_REPO,
        "genshin_root": str(genshin_root),
        "archive_sha256sums": str(archive_manifest),
        "archive_sha256sums_sha256": sha256(archive_manifest),
        "source_parquet_files": len(files),
        "source_rows": train_rows + held_out_rows,
        "train_parquet_files": len(train_files),
        "train_rows": train_rows,
        "held_out_files": [path.name for path in held_out],
        "held_out_rows": held_out_rows,
        "validation_rows": actual_val_rows,
    }
    (parquet_root / "source_manifest.json").write_text(
        json.dumps(source_manifest, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )

    config = {
        "version": 1,
        "format": "parquet",
        "root": str(parquet_root),
        "val": str(validation_path),
        "download": "never",
        "loader": {
            "streaming": True,
            "dataset_shuffle": True,
            "shuffle_buffer_size": 64,
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
        "datasets": {"genshin": {"path": "genshin", "passes": args.passes}},
    }
    yaml_path.write_text(
        yaml.safe_dump(config, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )

    print(f"source_repo={SOURCE_REPO}")
    print(f"source_files={len(files)} source_rows={train_rows + held_out_rows}")
    print(f"train_files={len(train_files)} train_rows={train_rows}")
    print(f"held_out={','.join(path.name for path in held_out)}")
    print(f"validation={validation_path} rows={actual_val_rows}")
    print(f"mixture_yaml={yaml_path} passes={args.passes}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
