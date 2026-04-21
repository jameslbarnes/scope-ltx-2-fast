"""
LTX 2.3 pipeline with all models GPU-resident.

Subclasses scope_ltx_2.LTX2Pipeline and no-ops every offloading method.
On a 96GB GPU, Gemma (13GB) + transformer (23GB) + VAEs (2GB) all fit
simultaneously, so the costly model-swapping on every prompt change is
eliminated entirely.

"""

import logging

import torch

from scope_ltx_2.pipeline import LTX2Pipeline

from .schema import LTX2FastConfig

logger = logging.getLogger(__name__)


class LTX2FastPipeline(LTX2Pipeline):
    """LTX 2.3 with no GPU↔CPU offloading — everything stays on GPU."""

    @classmethod
    def get_config_class(cls):
        return LTX2FastConfig

    def __init__(self, **kwargs):
        # Disable FFN chunking (not needed with ample VRAM)
        kwargs.setdefault("ffn_chunk_size", None)
        logger.info("[LTX2Fast] __init__ starting, calling super()...")
        super().__init__(**kwargs)
        logger.info(f"[LTX2Fast] super().__init__ done. _transformer exists: {hasattr(self, '_transformer')}, is None: {getattr(self, '_transformer', 'MISSING') is None}")

        # Move transformer from CPU → GPU permanently (parent loads to CPU)
        if hasattr(self, '_transformer') and self._transformer is not None:
            logger.info(f"Moving transformer to GPU (type={type(self._transformer).__name__})...")
            try:
                self._transformer = self._transformer.to(self.device)
                logger.info(
                    f"Transformer on GPU: {torch.cuda.memory_allocated() / 1e9:.1f}GB allocated"
                )
            except Exception as e:
                logger.error(f"Failed to move transformer to GPU: {e}")
                # Try moving individual blocks
                try:
                    if hasattr(self._transformer, 'transformer_blocks'):
                        for i, block in enumerate(self._transformer.transformer_blocks):
                            block.to(self.device)
                        logger.info(f"Moved {len(self._transformer.transformer_blocks)} blocks to GPU: {torch.cuda.memory_allocated() / 1e9:.1f}GB")
                except Exception as e2:
                    logger.error(f"Block-level move also failed: {e2}")
        else:
            logger.warning(f"No _transformer attribute found! attrs={[a for a in dir(self) if 'trans' in a.lower()]}")

        # Ensure VAEs are on GPU
        if not self._vaes_on_gpu:
            LTX2Pipeline._move_vaes_to_gpu(self)
            self._vaes_on_gpu = True

        # Move scaffold (embedding layers, etc.) to GPU
        try:
            LTX2Pipeline._move_transformer_scaffold_to_gpu(self)
        except Exception:
            pass

        # Move connectors to GPU
        try:
            LTX2Pipeline._move_connectors_to_gpu(self)
        except Exception:
            pass

        # Move text projection to GPU
        if hasattr(self, "_text_projection") and self._text_projection is not None:
            self._text_projection.to(self.device)

        logger.info(
            f"All models GPU-resident: {torch.cuda.memory_allocated() / 1e9:.1f}GB allocated"
        )

    # ── No-op all offloading methods ─────────────────────────────────────────

    def _offload_text_encoder(self):
        pass  # Gemma stays on GPU

    def _load_text_encoder(self):
        pass  # Already on GPU

    def _offload_vaes(self):
        pass  # VAEs stay on GPU

    def _move_vaes_to_gpu(self):
        pass  # Already on GPU

    def _teardown_denoising(self):
        pass  # No streaming state to tear down

    def _cleanup_block_streaming(self):
        pass  # No block streaming

    def _free_blocks_for_decode(self):
        pass  # Transformer stays on GPU during VAE decode

    def _reload_resident_blocks(self):
        pass  # Never offloaded

    def _apply_ffn_chunking(self, chunk_size=None):
        pass  # Not needed with ample VRAM

    def _undo_ffn_chunking(self):
        pass

    def _move_connectors_to_gpu(self):
        pass  # Already on GPU

    def _move_connectors_to_cpu(self):
        pass  # Stay on GPU

    def _move_transformer_scaffold_to_gpu(self):
        pass  # Already on GPU

    def _move_transformer_scaffold_to_cpu(self):
        pass  # Stay on GPU

    def _ensure_denoising_ready(self, total_tokens=0):
        # Call parent's implementation which moves all blocks to GPU when they fit
        # (on 275GB B300, all 48 blocks always fit)
        LTX2Pipeline._ensure_denoising_ready(self, total_tokens)

    def unload(self):
        """Clean up all GPU memory."""
        logger.info("Unloading all models from GPU...")
        for attr in [
            "_transformer",
            "_text_encoder",
            "_text_projection",
            "_video_vae",
            "_audio_vae",
            "_vocoder",
        ]:
            model = getattr(self, attr, None)
            if model is not None:
                model.cpu()
                delattr(self, attr)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("All models unloaded")
