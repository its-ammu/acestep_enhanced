# Copyright 2026 The ACESTEO Team. All rights reserved.
"""Unit tests for the CAPHAligner module."""

import unittest

import torch

from acestep.models.common.caph_aligner import CAPHAligner, f0_to_chroma


class TestF0ToChroma(unittest.TestCase):
    """Test case for f0_to_chroma conversion utility."""

    def test_voiced_mapping_exact(self):
        """Test that A440 (MIDI 69, chroma index 9) maps peak energy to bin 9."""
        f0 = torch.tensor([[440.0]])  # Shape: [1, 1]
        chroma = f0_to_chroma(f0, sigma=0.1)  # Small sigma for sharp peak
        # Index 9 corresponds to A
        self.assertEqual(chroma.shape, (1, 1, 12))
        max_idx = chroma.argmax(dim=-1).item()
        self.assertEqual(max_idx, 9)
        # Ensure it sums to 1.0 approximately
        self.assertAlmostEqual(chroma.sum().item(), 1.0, places=5)

    def test_unvoiced_masking(self):
        """Test that unvoiced frames (f0 <= 10.0) yield zero chroma vectors."""
        f0 = torch.tensor([[0.0, -10.0, 5.0]])  # Shape: [1, 3]
        chroma = f0_to_chroma(f0)
        self.assertEqual(chroma.shape, (1, 3, 12))
        # All elements should be exactly 0
        self.assertTrue((chroma == 0.0).all().item())

    def test_smooth_chroma_sum(self):
        """Test that chroma vector sums to 1.0 for valid voiced inputs."""
        f0 = torch.tensor([[100.0, 261.63, 1000.0]])  # Various Hz
        chroma = f0_to_chroma(f0)
        sums = chroma.sum(dim=-1)
        for i in range(3):
            self.assertAlmostEqual(sums[0, i].item(), 1.0, places=5)


class TestCAPHAligner(unittest.TestCase):
    """Test case for CAPHAligner module."""

    def test_matrix_properties(self):
        """Test that the music theory matrix M is symmetric and has expected values."""
        aligner = CAPHAligner(dit_dim=64)
        m = aligner.M
        self.assertEqual(m.shape, (12, 12))
        # Symmetry check
        self.assertTrue(torch.allclose(m, m.T))
        # Consonant unison (0 interval)
        self.assertAlmostEqual(m[0, 0].item(), 0.0, places=5)
        # Highly dissonant minor 2nd (1 interval)
        self.assertAlmostEqual(m[0, 1].item(), 1.0, places=5)
        # Highly dissonant tritone (6 interval)
        self.assertAlmostEqual(m[0, 6].item(), 1.0, places=5)
        # Moderately consonant perfect 4th (5 interval)
        self.assertAlmostEqual(m[0, 5].item(), 0.4, places=5)

    def test_forward_output_shapes(self):
        """Test alignment output shapes with both local and global window options."""
        batch_size = 2
        seq_len = 16
        dit_dim = 64
        aligner = CAPHAligner(dit_dim=dit_dim, num_heads=4)

        x_dit = torch.randn(batch_size, seq_len, dit_dim)
        vocal_chroma = torch.randn(batch_size, seq_len, 12).abs()

        # Local window forward
        attn_out, loss = aligner(x_dit, vocal_chroma, local_window_size=5)
        self.assertEqual(attn_out.shape, (batch_size, seq_len, dit_dim))
        self.assertEqual(loss.shape, ())

        # Global window forward
        attn_out_global, loss_global = aligner(x_dit, vocal_chroma, local_window_size=None)
        self.assertEqual(attn_out_global.shape, (batch_size, seq_len, dit_dim))
        self.assertEqual(loss_global.shape, ())

    def test_gradient_flow(self):
        """Test that Chord Distance Loss is differentiable and steers latents."""
        dit_dim = 16
        aligner = CAPHAligner(dit_dim=dit_dim, num_heads=2)

        x_dit = torch.randn(2, 8, dit_dim, requires_grad=True)
        vocal_chroma = f0_to_chroma(torch.tensor([[440.0] * 8, [261.63] * 8]))

        _, loss = aligner(x_dit, vocal_chroma)
        loss.backward()

        self.assertIsNotNone(x_dit.grad)
        # Gradients must contain non-zero values
        self.assertTrue((x_dit.grad.abs().sum() > 0.0).item())


if __name__ == "__main__":
    unittest.main()
