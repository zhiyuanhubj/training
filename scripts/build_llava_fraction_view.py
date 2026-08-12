#!/usr/bin/env python3
"""Materialize an exact row-prefix fraction of every canonical LLaVA subset.

Complete source shards are hard-linked (zero additional source-node storage).
Only the boundary shard of each subset is rewritten. The resulting directory is
self-contained when copied to another node and contains exactly the row quotas
used by the streaming ``path#N`` loader.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def target_rows(rows: int, fraction: Decimal) -> int:
    return int((Decimal(rows) * fraction).to_integral_value(rounding=ROUND_HALF_UP))


def write_prefix(source: Path, destination: Path, rows: int) -> None:
    existing = None
    if destination.is_file():
        existing = pq.ParquetFile(destination).metadata.num_rows
    if existing == rows:
        return
    # Some LLaVA schemas contain deeply nested chunked arrays that pyarrow
    # cannot expose through iter_batches. Shards are bounded in size, and the
    # H200 nodes have 2 TB RAM, so read the one boundary shard then slice it.
    table = pq.read_table(source).slice(0, rows)
    if len(table) != rows:
        raise RuntimeError(f"{source} has fewer than requested {rows} rows")
    temporary = destination.with_suffix(destination.suffix + f".tmp.{os.getpid()}")
    pq.write_table(table, temporary, compression="zstd")
    temporary.replace(destination)


def build_subset(source_root: Path, output_root: Path, item: dict, fraction: Decimal) -> tuple[dict, int]:
    name = item["canonical_name"]
    source_dir = source_root / name
    output_dir = output_root / name
    output_dir.mkdir(parents=True, exist_ok=True)
    for stale in output_dir.rglob("*.prefix-*.parquet"):
        stale.unlink()
    source_rows = int((item.get("stats") or {})["output_rows"])
    wanted = target_rows(source_rows, fraction)
    physical_rows = 0
    output_files = 0
    for source in sorted(source_dir.rglob("*.parquet")):
        rows = pq.ParquetFile(source).metadata.num_rows
        if physical_rows >= wanted:
            break
        remaining = wanted - physical_rows
        destination = output_dir / source.relative_to(source_dir)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if rows > remaining:
            write_prefix(source, destination, remaining)
            physical_rows += remaining
        else:
            if destination.exists():
                if not destination.samefile(source):
                    if pq.ParquetFile(destination).metadata.num_rows != rows:
                        raise RuntimeError(f"unexpected existing file: {destination}")
            else:
                os.link(source, destination)
            physical_rows += rows
        output_files += 1
    if physical_rows < wanted:
        raise RuntimeError(f"subset {name} is short by {wanted - physical_rows} rows")

    output_item = copy.deepcopy(item)
    output_item.setdefault("stats", {})["source_output_rows"] = source_rows
    output_item["stats"]["output_rows"] = wanted
    output_item["stats"]["physical_rows"] = physical_rows
    output_item["stats"]["selection"] = f"row_prefix_fraction={fraction}"
    return output_item, output_files


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--fraction", default="0.162")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    source_root = Path(args.source).resolve()
    output_root = Path(args.output).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    fraction = Decimal(args.fraction)
    if not (Decimal(0) < fraction <= Decimal(1)):
        raise ValueError("fraction must be in (0, 1]")

    index_path = source_root / "canonical_index.json"
    source_index = json.loads(index_path.read_text(encoding="utf-8"))
    items = source_index.get("datasets") or []
    output_items = []
    total_files = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(build_subset, source_root, output_root, item, fraction) for item in items]
        for completed, future in enumerate(as_completed(futures), 1):
            item, file_count = future.result()
            output_items.append(item)
            total_files += file_count
            if completed % 10 == 0 or completed == len(futures):
                print(f"[sample] subsets={completed}/{len(futures)} files={total_files}", flush=True)

    order = {item["canonical_name"]: index for index, item in enumerate(items)}
    output_items.sort(key=lambda item: order[item["canonical_name"]])
    output_index = copy.deepcopy(source_index)
    output_index["datasets"] = output_items
    (output_root / "canonical_index.json").write_text(
        json.dumps(output_index, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    manifest = {
        "version": 1,
        "source": str(source_root),
        "source_canonical_index_sha256": sha256(index_path),
        "fraction": str(fraction),
        "subsets": len(output_items),
        "rows": sum(int(item["stats"]["output_rows"]) for item in output_items),
        "files": total_files,
    }
    (output_root / "source_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
