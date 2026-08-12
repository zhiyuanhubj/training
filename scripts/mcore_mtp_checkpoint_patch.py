#!/usr/bin/env python3
"""Fix replicated MTP checkpoint metadata in the pinned mcore-bridge.

The bridge leaves ``MultiTokenPredictionLayer.tp_group`` as ``None``. When
Megatron wraps replicated norm parameters for distributed checkpointing, that
makes every TP rank look like TP rank zero, so all four copies are marked as
main replicas and sharding validation fails. Resolve the actual TP process
group after model-parallel initialization.
"""

from __future__ import annotations

from megatron.core import parallel_state
from mcore_bridge.model.modules.mtp_layer import MultiTokenPredictionLayer
from swift.utils import get_logger


logger = get_logger()
_ORIGINAL_INIT = MultiTokenPredictionLayer.__init__


def _init_with_tp_group(self, *args, **kwargs):
    _ORIGINAL_INIT(self, *args, **kwargs)
    if getattr(self, "tp_group", None) is None:
        self.tp_group = parallel_state.get_tensor_model_parallel_group()


MultiTokenPredictionLayer.__init__ = _init_with_tp_group
logger.info("Installed MTP checkpoint TP-replica metadata patch.")
