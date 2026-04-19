"""
LTX 2.3 pipeline with all models GPU-resident.

Subclasses scope_ltx_2.LTX2Pipeline and no-ops every offloading method.
On a 96GB GPU, Gemma (13GB) + transformer (23GB) + VAEs (2GB) all fit
simultaneously, so the costly model-swapping on every prompt change is
eliminated entirely.

Also implements last-frame conditioning: saves the final frame of each
chunk and uses it as i2v reference for the next chunk, creating visual
continuity between clips.
"""

import logging
import os
import tempfile

import numpy as np
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
        super().__init__(**kwargs)

        # Move transformer from CPU → GPU permanently (parent loads to CPU)
        if self._transformer is not None:
            logger.info("Moving transformer to GPU (permanent)...")
            self._transformer.to(self.device)
            logger.info(
                f"Transformer on GPU: {torch.cuda.memory_allocated() / 1e9:.1f}GB allocated"
            )

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

        # Last-frame conditioning state
        self._last_frame_path = os.path.join(tempfile.gettempdir(), "ltx2_fast_last_frame.png")
        self._has_last_frame = False

        logger.info(
            f"All models GPU-resident: {torch.cuda.memory_allocated() / 1e9:.1f}GB allocated"
        )

    # ── Last-frame conditioning for smooth transitions ────────────────────────

    def __call__(self, **kwargs) -> dict:
        # If we have a last frame and no explicit i2v_image, inject it
        if self._has_last_frame and kwargs.get("i2v_image") is None:
            kwargs["i2v_image"] = self._last_frame_path
            kwargs.setdefault("i2v_strength", 0.6)
            logger.info(f"Injecting last-frame conditioning (strength={kwargs['i2v_strength']})")

        result = super().__call__(**kwargs)

        # Save the last frame for next chunk's conditioning
        if "video" in result and result["video"] is not None:
            try:
                video = result["video"]  # (N, H, W, 3) uint8
                last_frame = video[-1]  # Last frame
                if isinstance(last_frame, torch.Tensor):
                    last_frame = last_frame.cpu().numpy()
                if last_frame.dtype != np.uint8:
                    last_frame = (last_frame * 255).clip(0, 255).astype(np.uint8)
                from PIL import Image
                Image.fromarray(last_frame).save(self._last_frame_path)
                self._has_last_frame = True
            except Exception as e:
                logger.warning(f"Failed to save last frame: {e}")
                self._has_last_frame = False

        return result

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
        pass  # Everything already on GPU, no streaming setup needed

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
