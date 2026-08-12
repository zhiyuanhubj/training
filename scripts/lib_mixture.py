#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""按每个数据集的总训练倍率生成 ms-swift ``path#N`` 规格和 phase manifest。

配置模板见 ``data_configs/mixture.example.yaml``。

推荐字段：
  - passes: 0.6  随机无重复使用数据集的 60%
  - passes: 1.0  完整训练一遍
  - passes: 2.3  完整训练两遍，再随机使用 30%

兼容旧字段：
  - multiplier: 与 passes 含义相同
  - sample:     直接指定绝对采样条数

混训数据已经在这里展开为整个训练任务所需的最终配额，因此训练器必须只
运行一个 epoch。各数据集合并后由 ms-swift 全局 shuffle。

JSONL mixture 直接统计单文件或子集目录内所有 ``*.jsonl`` shard 的行数。
Parquet mixture 从 root 下的 canonical_index.json 读取每个 canonical
子数据集的行数，并支持 loader 策略与数据配比放在同一个 YAML 中。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from pathlib import Path


def count_lines(path: str) -> int:
    n = 0
    with open(path, "rb") as f:
        for _ in f:
            n += 1
    return n


def count_jsonl_path(path: Path) -> int:
    """Count rows in one JSONL file or all JSONL shards below a directory."""
    if path.is_file():
        return count_lines(os.fspath(path))
    if path.is_dir():
        shards = sorted(path.rglob("*.jsonl"))
        if not shards:
            raise ValueError(f"JSONL 数据集目录中没有 .jsonl 文件: {path}")
        return sum(count_lines(os.fspath(shard)) for shard in shards)
    raise ValueError(f"数据集路径不存在: {path}")


def _jsonl_root(cfg: dict, override: str | None = None) -> Path | None:
    value = override or cfg.get("root")
    if not value:
        return None
    root = Path(os.path.expandvars(os.fspath(value)))
    if not root.is_absolute() and os.environ.get("NFS_DIR"):
        root = Path(os.environ["NFS_DIR"]) / "data" / root
    return root


def load_yaml(path: str) -> dict:
    try:
        import yaml
    except ImportError:
        sys.exit("[ERR] 需要 pyyaml: pip install pyyaml")
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError("YAML 顶层必须是 mapping")
    return cfg


def _positive_decimal(value, field: str, name: str) -> Decimal:
    try:
        number = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(f"数据集 {name} 的 {field} 不是有效数字: {value!r}") from exc
    if not number.is_finite() or number <= 0:
        raise ValueError(f"数据集 {name} 的 {field} 必须是有限正数")
    return number


def _parquet_root(cfg: dict, override: str | None = None) -> Path:
    value = override or cfg.get("root")
    if not isinstance(value, str) or not value:
        raise ValueError("parquet mixture requires root or --data-root")
    root = Path(os.path.expandvars(value))
    if not root.is_absolute() and os.environ.get("NFS_DIR"):
        root = Path(os.environ["NFS_DIR"]) / "data" / root
    return root


def _parquet_sizes(cfg: dict, override: str | None = None) -> tuple[Path, dict[str, int]]:
    root = _parquet_root(cfg, override)
    index_path = root / "canonical_index.json"
    if not index_path.is_file():
        raise ValueError(f"canonical parquet index does not exist: {index_path}")
    with index_path.open(encoding="utf-8") as f:
        index = json.load(f)
    sizes = {}
    for item in index.get("datasets") or []:
        name = item.get("canonical_name")
        rows = (item.get("stats") or {}).get("output_rows")
        if isinstance(name, str) and isinstance(rows, int) and rows > 0:
            sizes[name] = rows
    if not sizes:
        raise ValueError(f"canonical parquet index contains no dataset row counts: {index_path}")
    return root, sizes


def compute(cfg: dict, data_root: str | None = None) -> dict:
    if "weight" in cfg or "total_samples" in cfg:
        raise ValueError("不再支持 weight/total_samples；请为每个数据集设置 passes")
    if cfg.get("datasets") is not None and cfg.get("games") is not None:
        raise ValueError("YAML 不能同时包含 datasets 和 games")

    datasets = cfg.get("datasets")
    if datasets is None:
        datasets = cfg.get("games")  # 兼容旧配置名
    if not isinstance(datasets, dict) or not datasets:
        raise ValueError("YAML 中必须包含非空的 datasets mapping")

    data_format = cfg.get("format", "jsonl")
    if data_format not in {"jsonl", "parquet"}:
        raise ValueError("format must be jsonl or parquet")
    parquet_root = None
    parquet_sizes = None
    if data_format == "parquet":
        parquet_root, parquet_sizes = _parquet_sizes(cfg, data_root)

    info = {}
    for name, item in datasets.items():
        if not isinstance(item, dict):
            raise ValueError(f"数据集 {name} 的配置必须是 mapping")
        if "weight" in item:
            raise ValueError(f"数据集 {name} 仍在使用 weight；请改为 passes")
        controls = [key for key in ("passes", "multiplier", "sample") if key in item]
        if len(controls) > 1:
            raise ValueError(
                f"数据集 {name} 只能设置 passes/multiplier/sample 中的一项，"
                f"当前设置了: {', '.join(controls)}"
            )
        if "path" not in item and data_format == "jsonl" and _jsonl_root(cfg, data_root) is None:
            raise ValueError(f"JSONL 数据集 {name} 缺少 path，且 YAML 未设置 root")

        if data_format == "parquet":
            relative = os.fspath(item.get("path", name))
            path = os.fspath(parquet_root / relative)
            if name not in parquet_sizes:
                raise ValueError(f"数据集 {name} 不在 canonical_index.json 中")
            if not os.path.isdir(path):
                raise ValueError(f"数据集 {name} 的目录不存在: {path}")
            size = parquet_sizes[name]
            spec_path = f"{path}/."
        else:
            root = _jsonl_root(cfg, data_root)
            relative = os.fspath(item.get("path", name))
            path_value = Path(os.path.expandvars(relative))
            if not path_value.is_absolute() and root is not None:
                path_value = root / path_value
            if "path" not in item and not path_value.exists():
                jsonl_candidate = path_value.with_suffix(".jsonl")
                if jsonl_candidate.is_file():
                    path_value = jsonl_candidate
            path = os.fspath(path_value)
            size = count_jsonl_path(path_value)
            if size <= 0:
                raise ValueError(f"数据集 {name} 没有有效 JSONL 行: {path}")
            spec_path = path

        if "sample" in item:
            sample = int(item["sample"])
            if sample <= 0:
                raise ValueError(f"数据集 {name} 的 sample 必须 > 0")
            passes = Decimal(sample) / Decimal(size)
            mode = "sample"
        else:
            field = "passes" if "passes" in item else "multiplier"
            passes = _positive_decimal(item.get(field, 1), field, name)
            sample = int(
                (Decimal(size) * passes).to_integral_value(rounding=ROUND_HALF_UP)
            )
            mode = field if field in item else "default"

        info[name] = {
            "path": path,
            "spec_path": spec_path,
            "size": size,
            "passes": passes,
            "n": sample,
            "mode": mode,
        }
    return info


def build_pass_plan(info: dict, seed: int) -> dict:
    """Return a versioned, phase-aware plan without expanding sample indices.

    The plan deliberately contains quotas rather than a giant list of indices.
    The SWIFT plugin derives each permutation from this immutable plan, the
    seed, dataset name and phase id.  This makes the plan cheap to share among
    workers while retaining deterministic replay.
    """
    phases: dict[int, dict[str, int]] = {}
    datasets: dict[str, dict] = {}
    for name, item in info.items():
        size, target = int(item["size"]), int(item["n"])
        full_passes, remainder = divmod(target, size)
        datasets[name] = {
            "path": item["path"],
            "size": size,
            "target_samples": target,
            "full_passes": full_passes,
            "fractional_samples": remainder,
        }
        for phase_id in range(full_passes):
            phases.setdefault(phase_id, {})[name] = size
        if remainder:
            # A sub-one pass belongs to phase zero; otherwise it follows all
            # complete passes for that dataset.
            phase_id = full_passes
            phases.setdefault(phase_id, {})[name] = remainder

    phase_list = []
    for phase_id in sorted(phases):
        quotas = phases[phase_id]
        phase_list.append({
            "pass_id": phase_id,
            "datasets": quotas,
            "total_samples": sum(quotas.values()),
        })
    manifest = {
        "version": 1,
        "seed": int(seed),
        "total_samples": sum(item["target_samples"] for item in datasets.values()),
        "datasets": datasets,
        "phases": phase_list,
    }
    # Lets launchers detect a stale or partially replaced shared manifest.
    payload = json.dumps(manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    manifest["digest"] = hashlib.sha256(payload.encode()).hexdigest()
    return manifest


def write_pass_plan(path: str, info: dict, seed: int) -> dict:
    manifest = build_pass_plan(info, seed)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f".{target.name}.tmp.{os.getpid()}")
    tmp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(target)
    return manifest


def print_audit(info: dict, stream=sys.stdout) -> None:
    total = sum(item["n"] for item in info.values())
    print(
        f"{'数据集':<20}{'原始条数':>12}{'passes':>12}"
        f"{'采样条数':>12}{'实际概率':>12}{'模式':>12}",
        file=stream,
    )
    print("-" * 80, file=stream)
    for name, item in info.items():
        probability = item["n"] / total if total else 0
        print(
            f"{name:<20}{item['size']:>12}{str(item['passes']):>12}"
            f"{item['n']:>12}{probability:>11.4%}{item['mode']:>12}",
            file=stream,
        )
    print("-" * 80, file=stream)
    print(
        f"{'合计':<20}{sum(item['size'] for item in info.values()):>12}"
        f"{'':>12}{total:>12}{100:>11.4f}%",
        file=stream,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("yaml")
    parser.add_argument("--print-spec", action="store_true")
    parser.add_argument(
        "--print-interleave-prob",
        action="store_true",
        help="print sample-count-normalized interleave probabilities in dataset order",
    )
    parser.add_argument(
        "--print-total-samples",
        action="store_true",
        help="print the final sample exposure after applying every dataset pass multiplier",
    )
    parser.add_argument("--print-val", action="store_true")
    parser.add_argument("--summary", action="store_true")
    parser.add_argument("--data-root", help="override the mixture data root")
    parser.add_argument("--seed", type=int, default=42, help="deterministic seed stored in a pass-aware plan")
    parser.add_argument("--write-plan", help="write a versioned pass-aware manifest")
    args = parser.parse_args()

    cfg = load_yaml(args.yaml)
    if args.print_val:
        val = os.path.expandvars(os.fspath(cfg.get("val", "")))
        if cfg.get("format") == "parquet" and val and not Path(val).is_absolute():
            val = _parquet_root(cfg, args.data_root).parent / val
        print(val)
        return

    info = compute(cfg, args.data_root)
    if args.print_interleave_prob:
        total = sum(item["n"] for item in info.values())
        if total <= 0:
            raise ValueError("所有数据集的最终采样条数都是 0")
        print(" ".join(str(item["n"] / total) for item in info.values()))
        return
    if args.print_total_samples:
        print(sum(item["n"] for item in info.values()))
        return
    active = [(name, item) for name, item in info.items() if item["n"] > 0]
    skipped = [name for name, item in info.items() if item["n"] == 0]
    if not active:
        raise ValueError("所有数据集的最终采样条数都是 0")

    if args.print_spec:
        print_audit(info, stream=sys.stderr)
        if skipped:
            print(
                "[mixture][WARN] 配额取整为 0，未加入训练: " + ", ".join(skipped),
                file=sys.stderr,
            )
        print(" ".join(f"{item['spec_path']}#{item['n']}" for _, item in active))
        return

    if args.write_plan:
        manifest = write_pass_plan(args.write_plan, info, args.seed)
        print(f"wrote pass-aware plan: {args.write_plan} ({len(manifest['phases'])} phase(s), "
              f"{manifest['total_samples']} samples)")
        return

    print_audit(info)
    print(f"\nval: {cfg.get('val', '(无)')}")
    print(
        "DATASET_SPEC = "
        + " ".join(f"{item['spec_path']}#{item['n']}" for _, item in active)
    )


if __name__ == "__main__":
    main()
