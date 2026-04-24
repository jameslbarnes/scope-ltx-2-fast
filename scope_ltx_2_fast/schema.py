"""Config schema for the high-VRAM LTX 2.3 pipeline."""

from typing import ClassVar

from scope_ltx_2.schema import LTX2Config


class LTX2FastConfig(LTX2Config):
    pipeline_id: ClassVar[str] = "ltx2-fast"
    pipeline_name: ClassVar[str] = "LTX 2.3 (High-VRAM)"
    pipeline_description: ClassVar[str] = (
        "LTX 2.3 with all models GPU-resident — no offloading, instant prompt changes. "
        "Requires 48GB+ VRAM."
    )
    pipeline_version: ClassVar[str] = "0.1.0"
    estimated_vram_gb: ClassVar[float | None] = 45.0

    # Landscape defaults (parent defaults are portrait 384x320)
    height: int = 384
    width: int = 512

    # Disable FFN chunking by default (not needed with 96GB)
    ffn_chunk_size: int | None = None
