"""
LTX 2.3 pipeline optimized for high-VRAM GPUs (48GB+).

Only change from parent: Gemma text encoder stays on GPU permanently.
This eliminates the 15-second offload/reload cycle on prompt changes.

Everything else (transformer block streaming, VAE management) uses the
parent's implementation unchanged — it already detects when all blocks
fit on GPU and avoids streaming in that case.
"""

import logging

from scope_ltx_2.pipeline import LTX2Pipeline

from .schema import LTX2FastConfig

logger = logging.getLogger(__name__)


class LTX2FastPipeline(LTX2Pipeline):
    """LTX 2.3 — keeps Gemma on GPU to eliminate prompt change latency."""

    @classmethod
    def get_config_class(cls):
        return LTX2FastConfig

    def __init__(self, **kwargs):
        kwargs.setdefault("ffn_chunk_size", None)
        super().__init__(**kwargs)

    # Only no-op the Gemma offloading — this is what causes the 15s penalty
    def _offload_text_encoder(self):
        pass

    def _load_text_encoder(self):
        pass
