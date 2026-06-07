# Copyright 2026 The ACESTEO Team. All rights reserved.
"""Integration tests for the CAPH-steered remix workflow.

Tests the full pipeline:
  [Vocal Stem] → f0 → chroma → CAPHAligner → CDL steering
  [Instrumental Stem] → src_latents → Flow-Edit loop
  Result: genre-shifted instrumental harmonically consonant with original vocals.

All tests are CPU-only and use mock model forward passes.
"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

import torch

from acestep.models.common.caph_aligner import CAPHAligner, f0_to_chroma
from acestep.models.common.caph_steering import (
    apply_caph_gradient_steering,
    maybe_create_caph_aligner,
)
from acestep.models.common.vocal_f0_extraction import extract_vocal_chroma


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_fake_model(latent_dim: int, seq_len: int, batch_size: int = 1):
    """Build a minimal mock DiT model that returns fixed-velocity outputs."""
    model = MagicMock()
    # decoder returns (velocity, new_kv_cache)
    velocity = torch.zeros(batch_size, seq_len, latent_dim)
    cache_mock = MagicMock()
    model.decoder.return_value = (velocity, cache_mock)
    model.null_condition_emb = torch.zeros(1, 1, latent_dim)
    return model


def _make_chroma(batch: int = 1, seq: int = 32) -> torch.Tensor:
    """Return a synthetic vocal chroma tensor with A440 pitch throughout."""
    f0 = torch.full((batch, seq), 440.0)  # A4 = 440 Hz, chroma bin 9
    return f0_to_chroma(f0)


# ---------------------------------------------------------------------------
# Unit: MaybeCreateCAPHAligner
# ---------------------------------------------------------------------------

class TestMaybeCreateCAPHAligner(unittest.TestCase):
    """Test lazy aligner instantiation guard."""

    def test_returns_none_when_scale_zero(self):
        """CDL disabled (scale=0) → no aligner created."""
        chroma = _make_chroma()
        result = maybe_create_caph_aligner(
            latent_dim=64, vocal_chroma=chroma, cdl_guidance_scale=0.0,
            device=torch.device("cpu"), dtype=torch.float32,
        )
        self.assertIsNone(result)

    def test_returns_none_when_chroma_none(self):
        """No vocal chroma → no aligner created even with scale > 0."""
        result = maybe_create_caph_aligner(
            latent_dim=64, vocal_chroma=None, cdl_guidance_scale=0.5,
            device=torch.device("cpu"), dtype=torch.float32,
        )
        self.assertIsNone(result)

    def test_returns_aligner_when_both_provided(self):
        """Both chroma and scale > 0 → CAPHAligner on correct device/dtype."""
        chroma = _make_chroma()
        aligner = maybe_create_caph_aligner(
            latent_dim=64, vocal_chroma=chroma, cdl_guidance_scale=0.3,
            device=torch.device("cpu"), dtype=torch.float32,
        )
        self.assertIsInstance(aligner, CAPHAligner)

    def test_head_divisibility_fallback(self):
        """Odd latent_dim (not divisible by 8/4/2) falls back to 1 head."""
        chroma = _make_chroma()
        aligner = maybe_create_caph_aligner(
            latent_dim=7, vocal_chroma=chroma, cdl_guidance_scale=0.1,
            device=torch.device("cpu"), dtype=torch.float32,
        )
        self.assertIsInstance(aligner, CAPHAligner)
        self.assertEqual(aligner.num_heads, 1)


# ---------------------------------------------------------------------------
# Unit: ApplyCAPHGradientSteering
# ---------------------------------------------------------------------------

class TestApplyCAPHGradientSteering(unittest.TestCase):
    """Test that gradient steering reduces CDL loss."""

    def test_steering_reduces_cdl(self):
        """After one gradient step the steered latent has lower CDL than original."""
        latent_dim, seq_len = 16, 8
        aligner = CAPHAligner(dit_dim=latent_dim, num_heads=2)
        chroma = _make_chroma(seq=seq_len)  # [1, 8, 12]

        latent = torch.randn(1, seq_len, latent_dim)

        _, loss_before = aligner(latent, chroma)
        steered = apply_caph_gradient_steering(latent, chroma, aligner, cdl_guidance_scale=0.5)
        _, loss_after = aligner(steered, chroma)

        self.assertLess(loss_after.item(), loss_before.item(),
                        "CDL should decrease after gradient steering step")

    def test_temporal_mismatch_interpolated(self):
        """Chroma with different T from latent is interpolated without error."""
        latent_dim, latent_seq = 16, 20
        chroma_seq = 10  # half the latent length
        aligner = CAPHAligner(dit_dim=latent_dim, num_heads=2)

        latent = torch.randn(1, latent_seq, latent_dim)
        chroma = _make_chroma(seq=chroma_seq)  # [1, 10, 12]

        steered = apply_caph_gradient_steering(latent, chroma, aligner, cdl_guidance_scale=0.1)
        self.assertEqual(steered.shape, (1, latent_seq, latent_dim))

    def test_no_inplace_modification(self):
        """Original latent tensor must not be modified in-place."""
        latent_dim, seq_len = 16, 8
        aligner = CAPHAligner(dit_dim=latent_dim, num_heads=2)
        chroma = _make_chroma(seq=seq_len)

        latent = torch.randn(1, seq_len, latent_dim)
        original_data = latent.clone()

        apply_caph_gradient_steering(latent, chroma, aligner, cdl_guidance_scale=0.3)
        self.assertTrue(torch.allclose(latent, original_data),
                        "Original latent must not be modified in-place")


# ---------------------------------------------------------------------------
# Integration: Full Flow-Edit + CAPH Steering loop
# ---------------------------------------------------------------------------

class TestFlowEditCAPHIntegration(unittest.TestCase):
    """Simulate the full cover+flow_edit+CAPH workflow with mock model."""

    def _run_loop(self, cdl_guidance_scale: float) -> tuple[torch.Tensor, torch.Tensor]:
        """Run flowedit_sampling_loop with and without CAPH steering.

        Returns (unsteered_latent, steered_latent) pair.
        """
        from acestep.models.common.flow_edit import flowedit_sampling_loop
        from transformers.cache_utils import DynamicCache, EncoderDecoderCache

        latent_dim, seq_len, bsz = 16, 8, 1
        src_latents = torch.randn(bsz, seq_len, latent_dim)
        chroma = _make_chroma(seq=seq_len)

        def _make_mock_pack():
            enc_hs = torch.zeros(bsz, 4, latent_dim)
            enc_am = torch.ones(bsz, 4, latent_dim)
            ctx = torch.zeros(bsz, seq_len, latent_dim)
            attn = torch.ones(bsz, seq_len, latent_dim)
            return enc_hs, enc_am, ctx, attn

        src_pack = _make_mock_pack()
        tar_pack = _make_mock_pack()
        model = _make_fake_model(latent_dim, seq_len, bsz)

        aligner = maybe_create_caph_aligner(
            latent_dim=latent_dim,
            vocal_chroma=chroma if cdl_guidance_scale > 0 else None,
            cdl_guidance_scale=cdl_guidance_scale,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )

        result = flowedit_sampling_loop(
            model,
            src_encoder_hidden_states=src_pack[0],
            src_encoder_attention_mask=src_pack[1],
            src_context_latents=src_pack[2],
            tar_encoder_hidden_states=tar_pack[0],
            tar_encoder_attention_mask=tar_pack[1],
            tar_context_latents=tar_pack[2],
            src_latents=src_latents,
            attention_mask=torch.ones(bsz, seq_len),
            null_condition_emb=torch.zeros(1, 1, latent_dim),
            infer_steps=4,
            n_min=0.0,
            n_max=0.8,
            diffusion_guidance_scale=1.0,  # disable CFG so mock single-batch output is valid
            use_progress_bar=False,
            vocal_chroma=chroma if cdl_guidance_scale > 0 else None,
            cdl_guidance_scale=cdl_guidance_scale,
            caph_aligner=aligner,
        )
        return result["target_latents"]

    def test_cdl_zero_no_steering(self):
        """With cdl_guidance_scale=0 the loop runs normally (backward compat)."""
        result = self._run_loop(0.0)
        self.assertEqual(result.shape[2], 16)  # latent_dim preserved

    def test_cdl_nonzero_changes_output(self):
        """With cdl_guidance_scale>0 output differs from unsteered (steering had effect)."""
        unsteered = self._run_loop(0.0)
        steered = self._run_loop(0.5)
        # They should differ because gradient steering shifted the latent
        self.assertFalse(
            torch.allclose(unsteered, steered),
            "CAPH steering should produce a different latent than unsteered",
        )

    def test_steered_lower_cdl(self):
        """Steered latent has strictly lower CDL against the vocal chroma."""
        from acestep.models.common.flow_edit import flowedit_sampling_loop

        latent_dim, seq_len = 16, 8
        src_latents = torch.randn(1, seq_len, latent_dim)
        chroma = _make_chroma(seq=seq_len)
        aligner = CAPHAligner(dit_dim=latent_dim, num_heads=2)

        model = _make_fake_model(latent_dim, seq_len)

        def _run(scale):
            _aligner = aligner if scale > 0 else None
            _chroma = chroma if scale > 0 else None
            return flowedit_sampling_loop(
                model,
                src_encoder_hidden_states=torch.zeros(1, 4, latent_dim),
                src_encoder_attention_mask=torch.ones(1, 4, latent_dim),
                src_context_latents=torch.zeros(1, seq_len, latent_dim),
                tar_encoder_hidden_states=torch.zeros(1, 4, latent_dim),
                tar_encoder_attention_mask=torch.ones(1, 4, latent_dim),
                tar_context_latents=torch.zeros(1, seq_len, latent_dim),
                src_latents=src_latents,
                attention_mask=torch.ones(1, seq_len),
                null_condition_emb=torch.zeros(1, 1, latent_dim),
                infer_steps=5,
                n_min=0.0,
                n_max=1.0,
                diffusion_guidance_scale=1.0,  # disable CFG; mock returns single-batch tensor
                use_progress_bar=False,
                vocal_chroma=_chroma,
                cdl_guidance_scale=scale,
                caph_aligner=_aligner,
            )["target_latents"]

        unsteered = _run(0.0)
        steered = _run(0.5)

        _, cdl_unsteered = aligner(unsteered, chroma)
        _, cdl_steered = aligner(steered, chroma)

        self.assertLess(
            cdl_steered.item(), cdl_unsteered.item(),
            f"Steered CDL ({cdl_steered.item():.4f}) should be < "
            f"unsteered ({cdl_unsteered.item():.4f})",
        )


# ---------------------------------------------------------------------------
# Unit: _extract_vocal_chroma_safe (generation_progress helper)
# ---------------------------------------------------------------------------

class TestExtractVocalChromaSafe(unittest.TestCase):
    """Test the safe UI-boundary chroma extraction helper."""

    def _get_helper(self):
        from acestep.ui.gradio.events.results.generation_progress import _extract_vocal_chroma_safe
        return _extract_vocal_chroma_safe

    def test_returns_none_when_scale_zero(self):
        fn = self._get_helper()
        self.assertIsNone(fn("/some/path.wav", 0.0))

    def test_returns_none_when_path_none(self):
        fn = self._get_helper()
        self.assertIsNone(fn(None, 0.5))

    def test_returns_none_when_path_empty(self):
        fn = self._get_helper()
        self.assertIsNone(fn("", 0.5))

    def test_returns_none_on_extraction_failure(self):
        """Extraction error is swallowed and returns None (never raises)."""
        fn = self._get_helper()
        with patch(
            "acestep.models.common.vocal_f0_extraction.extract_vocal_chroma",
            side_effect=RuntimeError("torchaudio not available"),
        ):
            result = fn("/nonexistent/vocal.wav", 0.3)
        self.assertIsNone(result)

    def test_returns_chroma_on_success(self):
        """Successful extraction returns [1, T, 12] tensor."""
        fn = self._get_helper()
        fake_chroma = torch.zeros(1, 50, 12)
        with patch(
            "acestep.models.common.vocal_f0_extraction.extract_vocal_chroma",
            return_value=fake_chroma,
        ):
            result = fn("/fake/vocal.wav", 0.3)
        self.assertIsNotNone(result)
        self.assertEqual(result.shape[-1], 12)


if __name__ == "__main__":
    unittest.main()
