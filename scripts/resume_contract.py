#!/usr/bin/env python3
"""Create or validate the immutable configuration for an exact training resume."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import sys
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def as_bool(name: str, default: str = "false") -> bool:
    value = os.environ.get(name, default).lower()
    if value not in {"1", "0", "true", "false", "yes", "no", "on", "off"}:
        raise ValueError(f"{name} is not a boolean: {value!r}")
    return value in {"1", "true", "yes", "on"}


def as_int(name: str, default: int = 0) -> int:
    return int(os.environ.get(name, str(default)))


def dataset_evidence(dataset_spec: str) -> list[dict[str, Any]]:
    evidence = []
    seen_manifests = set()
    for token in shlex.split(dataset_spec):
        raw_path = token.rsplit("#", 1)[0]
        path = Path(raw_path).expanduser()
        canonical_path = path
        if path.name == ".":
            canonical_path = path.parent
        item: dict[str, Any] = {
            "path": str(path),
            "resolved": str(canonical_path.resolve()),
        }
        if canonical_path.is_file():
            stat = canonical_path.stat()
            item.update({"kind": "file", "size": stat.st_size, "mtime_ns": stat.st_mtime_ns})
        elif canonical_path.is_dir():
            item["kind"] = "directory"
            root = canonical_path.parent
            for candidate_root in (canonical_path, root):
                for name in (
                    "canonical_index.json",
                    "source_manifest.json",
                    "manifest.jsonl",
                    "SHA256SUMS",
                ):
                    manifest = candidate_root / name
                    resolved = str(manifest.resolve())
                    if resolved in seen_manifests or not manifest.is_file():
                        continue
                    seen_manifests.add(resolved)
                    evidence.append(
                        {
                            "manifest": str(manifest),
                            "resolved": resolved,
                            "sha256": sha256(manifest),
                        }
                    )
        else:
            item["kind"] = "missing"
        evidence.append(item)
    return evidence


def build_contract() -> dict[str, Any]:
    project_root = Path(os.environ["PROJECT_ROOT"])
    model_path = Path(os.environ["MODEL_PATH"])
    dataset_spec = os.environ.get("DATASET_SPEC") or os.environ.get("TRAIN_FILE", "")
    world_size = as_int("NNODES", 1) * as_int("NPROC_PER_NODE", 1)
    tp = as_int("TP", 1)
    pp = as_int("PP", 1)
    if world_size % (tp * pp):
        raise ValueError(f"WORLD_SIZE={world_size} is not divisible by TP*PP={tp * pp}")

    mixture_yaml = Path(os.environ["MIXTURE_YAML"]) if os.environ.get("MIXTURE_YAML") else None
    return {
        "version": 1,
        "model": {
            "path": str(model_path),
            "resolved": str(model_path.resolve()),
            "config_sha256": sha256(model_path / "config.json"),
        },
        "data": {
            "dataset_spec": dataset_spec,
            "evidence": dataset_evidence(dataset_spec),
            "mixture_yaml": str(mixture_yaml) if mixture_yaml else None,
            "mixture_yaml_sha256": sha256(mixture_yaml) if mixture_yaml else None,
            "columns": os.environ.get("CHANNEL_COLUMNS_JSON", ""),
            "streaming": as_bool("STREAMING"),
            "streaming_shard_by_dp": as_bool("STREAMING_SHARD_BY_DP"),
            "dataset_shuffle": as_bool("DATASET_SHUFFLE"),
            "shuffle_buffer_size": as_int("SHUFFLE_BUFFER_SIZE", 10000),
            "interleave_prob": os.environ.get("INTERLEAVE_PROB", ""),
            "stopping_strategy": os.environ.get("STOPPING_STRATEGY", "first_exhausted"),
            "stop_at_dataset_end": as_bool("STOP_AT_DATASET_END"),
            "packing": as_bool("PACKING"),
            "packing_length": as_int("PACKING_LENGTH"),
            "packing_strategy": os.environ.get("PACKING_STRATEGY", "binpack"),
            "packing_interval": as_int("PACKING_INTERVAL", 128),
            "dataset_num_proc": as_int("DATASET_NUM_PROC", 8),
            "seed": as_int("SEED", 42),
            "add_genshin_special_tokens": as_bool("ADD_GENSHIN_SPECIAL_TOKENS", "true"),
        },
        "parallel": {
            "nnodes": as_int("NNODES", 1),
            "nproc_per_node": as_int("NPROC_PER_NODE", 1),
            "world_size": world_size,
            "tp": tp,
            "pp": pp,
            "dp": world_size // (tp * pp),
            "micro_batch_size": as_int("MICRO_BS", 1),
            "global_batch_size": as_int("GLOBAL_BS", 1),
            "sequence_parallel": as_bool("SEQ_PARALLEL", "true"),
        },
        "sequence": {
            "max_length": as_int("MAX_LEN"),
            "image_max_tokens": as_int("IMG_MAX_TOK"),
        },
        "code": {
            "streaming_schema_patch_sha256": sha256(
                project_root / "scripts" / "swift_streaming_schema_patch.py"
            ),
            "mtp_checkpoint_patch_sha256": sha256(
                project_root / "scripts" / "mcore_mtp_checkpoint_patch.py"
            ),
            "swift_packing_sha256": sha256(
                project_root / ".deps" / "ms-swift" / "swift" / "dataset" / "packing.py"
            ),
            "swift_sft_pipeline_sha256": sha256(
                project_root
                / ".deps"
                / "ms-swift"
                / "swift"
                / "pipelines"
                / "train"
                / "sft.py"
            ),
            "swift_megatron_trainer_sha256": sha256(
                project_root
                / ".deps"
                / "ms-swift"
                / "swift"
                / "megatron"
                / "trainers"
                / "base.py"
            ),
            "swift_megatron_dataloader_sha256": sha256(
                project_root
                / ".deps"
                / "ms-swift"
                / "swift"
                / "megatron"
                / "trainers"
                / "utils.py"
            ),
            "swift_megatron_args_sha256": sha256(
                project_root
                / ".deps"
                / "ms-swift"
                / "swift"
                / "megatron"
                / "arguments"
                / "megatron_args.py"
            ),
            "swift_data_args_sha256": sha256(
                project_root
                / ".deps"
                / "ms-swift"
                / "swift"
                / "arguments"
                / "base_args"
                / "data_args.py"
            ),
            "swift_dataset_loader_sha256": sha256(
                project_root
                / ".deps"
                / "ms-swift"
                / "swift"
                / "dataset"
                / "loader.py"
            ),
            "swift_dataloader_dispatcher_sha256": sha256(
                project_root
                / ".deps"
                / "ms-swift"
                / "swift"
                / "dataloader"
                / "dispatcher.py"
            ),
        },
    }


def differences(expected: Any, actual: Any, prefix: str = "") -> list[str]:
    if isinstance(expected, dict) and isinstance(actual, dict):
        result = []
        for key in sorted(set(expected) | set(actual)):
            path = f"{prefix}.{key}" if prefix else key
            if key not in expected:
                result.append(f"{path}: added {actual[key]!r}")
            elif key not in actual:
                result.append(f"{path}: missing (was {expected[key]!r})")
            else:
                result.extend(differences(expected[key], actual[key], path))
        return result
    if expected != actual:
        return [f"{prefix}: checkpoint={expected!r}, current={actual!r}"]
    return []


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", required=True)
    parser.add_argument("--require-existing", action="store_true")
    args = parser.parse_args()
    path = Path(args.path)
    current = build_contract()

    if path.exists():
        expected = json.loads(path.read_text(encoding="utf-8"))
        diff = differences(expected, current)
        if diff:
            print("resume contract mismatch:", file=sys.stderr)
            for line in diff[:50]:
                print(f"  - {line}", file=sys.stderr)
            return 2
        print(f"[resume-contract] validated {path}")
        return 0

    if args.require_existing:
        print(f"resume contract is missing: {path}", file=sys.stderr)
        return 2
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(current, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
    print(f"[resume-contract] wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
