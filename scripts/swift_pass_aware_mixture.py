#!/usr/bin/env python
"""Pass-aware streaming mixture plugin for Megatron-SWIFT.

Set ``PASS_AWARE_MIXTURE=true`` and point ``MIXTURE_PLAN`` at a manifest made
by :mod:`lib_mixture`.  Unlike ``path#N``, the stream below never emits an
item from pass N+1 before all quotas in pass N have been consumed.  It also
preserves the plan's deterministic order across restarts when the topology is
unchanged.

This module intentionally patches only public dependencies used by the
existing streaming patch.  If the installed SWIFT/datasets API cannot support
the required hooks, it fails before training instead of silently falling back
to the unsafe global-shuffle behaviour.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from datasets import IterableDataset
from swift.dataset.dataset_meta import BaseDatasetLoader
from swift.template.templates.qwen import Qwen2VLTemplate
from swift.utils import get_logger

logger = get_logger()


def _seed(base: int, dataset: str, phase: int) -> int:
    value = f"{base}:{dataset}:{phase}".encode()
    return int.from_bytes(hashlib.sha256(value).digest()[:8], "big")


def load_plan(path: str | os.PathLike[str]) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as f:
        plan = json.load(f)
    if plan.get("version") != 1:
        raise RuntimeError(f"unsupported pass-aware manifest version: {plan.get('version')!r}")
    if not isinstance(plan.get("seed"), int) or not isinstance(plan.get("phases"), list):
        raise RuntimeError("invalid pass-aware manifest: seed/phases are required")
    datasets = plan.get("datasets")
    if not isinstance(datasets, dict) or not datasets:
        raise RuntimeError("invalid pass-aware manifest: datasets are required")
    expected_total = 0
    for phase in plan["phases"]:
        if not isinstance(phase, dict) or not isinstance(phase.get("pass_id"), int):
            raise RuntimeError("invalid pass-aware manifest phase")
        quotas = phase.get("datasets")
        if not isinstance(quotas, dict) or not quotas:
            raise RuntimeError("invalid pass-aware manifest phase datasets")
        for name, quota in quotas.items():
            if name not in datasets or not isinstance(quota, int) or quota < 1:
                raise RuntimeError(f"invalid pass-aware quota: {name}={quota!r}")
            expected_total += quota
    if expected_total != plan.get("total_samples"):
        raise RuntimeError("pass-aware manifest total_samples does not match phases")
    digest = plan.get("digest")
    unsigned = dict(plan)
    unsigned.pop("digest", None)
    expected_digest = hashlib.sha256(
        json.dumps(unsigned, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if digest != expected_digest:
        raise RuntimeError("pass-aware manifest digest mismatch")
    return plan


def _line_offsets(path: Path) -> list[int]:
    offsets = []
    with path.open("rb") as f:
        while True:
            offset = f.tell()
            line = f.readline()
            if not line:
                break
            if line.strip():
                offsets.append(offset)
    return offsets


def _rows_for_phase(plan: dict[str, Any], name: str, phase_id: int, quota: int) -> Iterator[dict[str, Any]]:
    spec = plan["datasets"][name]
    path = Path(spec["path"])
    expected_size = int(spec["size"])
    offsets = _line_offsets(path)
    if len(offsets) != expected_size:
        raise RuntimeError(
            f"dataset changed after manifest creation: {path} has {len(offsets)} rows, expected {expected_size}")
    if quota > expected_size:
        raise RuntimeError(f"phase quota exceeds source size: {name} {quota}>{expected_size}")
    order = list(range(expected_size))
    random.Random(_seed(plan["seed"], name, phase_id)).shuffle(order)
    with path.open("rb") as f:
        for original_index in order[:quota]:
            f.seek(offsets[original_index])
            row = json.loads(f.readline())
            # Template preprocessing normally forwards unknown columns until
            # packing_row.  The guard below deliberately fails if a future
            # SWIFT release drops it before packing, rather than mixing phases.
            row["_mixture_dataset"] = name
            row["_mixture_original_index"] = original_index
            row["_mixture_pass"] = phase_id
            yield row


def iter_plan_rows(plan: dict[str, Any]) -> Iterator[dict[str, Any]]:
    """Yield phase ordered rows, with weighted random mixing inside a phase."""
    for phase in plan["phases"]:
        phase_id = phase["pass_id"]
        streams = {
            name: _rows_for_phase(plan, name, phase_id, int(quota))
            for name, quota in phase["datasets"].items()
        }
        remaining = {name: int(quota) for name, quota in phase["datasets"].items()}
        rng = random.Random(_seed(plan["seed"], "__interleave__", phase_id))
        while remaining:
            names = sorted(remaining)
            # Sampling by remaining quota produces a weighted interleave while
            # only ever consuming the head of each per-dataset pass stream.
            choice = rng.randrange(sum(remaining.values()))
            selected = names[-1]
            for name in names:
                choice -= remaining[name]
                if choice < 0:
                    selected = name
                    break
            yield next(streams[selected])
            remaining[selected] -= 1
            if remaining[selected] == 0:
                del remaining[selected]


def _data_parallel_rank() -> tuple[int, int]:
    """Return the DP rank only; TP/PP replicas must see the same samples."""
    try:
        import torch.distributed as dist
        from megatron.core import mpu
        if not dist.is_available() or not dist.is_initialized():
            return 0, 1
        group = mpu.get_data_parallel_group(with_context_parallel=True)
        return dist.get_rank(group=group), dist.get_world_size(group=group)
    except Exception as exc:  # pragma: no cover - exercised in a real launcher
        raise RuntimeError("cannot resolve Megatron data-parallel group for pass-aware mixture") from exc


def iter_plan_rows_for_rank(plan: dict[str, Any]) -> Iterator[dict[str, Any]]:
    """Deterministically shard the global stream across DP, never TP/PP."""
    rank, world_size = _data_parallel_rank()
    for global_index, row in enumerate(iter_plan_rows(plan)):
        if global_index % world_size == rank:
            yield row


def _pass_aware_interleave(_datasets, *args, **kwargs):
    # Do not reuse SWIFT's source iterables: path#N has already expanded or
    # repeated them.  The manifest owns all ordering and quota semantics.
    return IterableDataset.from_generator(iter_plan_rows_for_rank, gen_kwargs={"plan": _PLAN})


_ORIGINAL_PACKING_ROW = Qwen2VLTemplate.packing_row


def _phase_safe_packing_row(template, rows):
    phase_ids = {row.get("_mixture_pass") for row in rows}
    if None in phase_ids:
        raise RuntimeError(
            "PASS_AWARE_MIXTURE requires SWIFT to preserve _mixture_pass through preprocessing; "
            "installed SWIFT API is incompatible")
    if len(phase_ids) != 1:
        raise RuntimeError(f"pass-aware packing attempted to cross a phase boundary: {sorted(phase_ids)!r}")
    for row in rows:
        row.pop("_mixture_dataset", None)
        row.pop("_mixture_original_index", None)
        row.pop("_mixture_pass", None)
    return _ORIGINAL_PACKING_ROW(template, rows)


def _install() -> None:
    global _PLAN
    plan_path = os.environ.get("MIXTURE_PLAN")
    if not plan_path:
        raise RuntimeError("PASS_AWARE_MIXTURE=true requires MIXTURE_PLAN")
    _PLAN = load_plan(plan_path)
    if not hasattr(BaseDatasetLoader, "interleave_datasets"):
        raise RuntimeError("installed SWIFT lacks BaseDatasetLoader.interleave_datasets")
    if not hasattr(IterableDataset, "from_generator"):
        raise RuntimeError("installed datasets lacks IterableDataset.from_generator")
    BaseDatasetLoader.interleave_datasets = staticmethod(_pass_aware_interleave)
    Qwen2VLTemplate.packing_row = _phase_safe_packing_row
    logger.info(
        "Pass-aware mixture enabled: %s phase(s), %s samples, manifest=%s",
        len(_PLAN["phases"]), _PLAN["total_samples"], plan_path)


if os.environ.get("PASS_AWARE_MIXTURE", "false").lower() in {"1", "true", "yes", "on"}:
    _install()
