# Copyright 2026 The ACESTEO Team. All rights reserved.
"""Cross-Attention Pitch-to-Harmony (CAPH) Aligner implementation.

Provides the CAPHAligner module for music-theory-guided latent alignment
and the differentiable Chord Distance Loss (CDL).
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def f0_to_chroma(f0: torch.Tensor, sigma: float = 0.5) -> torch.Tensor:
    """Convert fundamental frequency (f0) sequence to smooth 12-dimensional pitch chroma.

    Args:
        f0: Tensor of shape [Batch, Seq_Len] with fundamental frequency values in Hz.
        sigma: Standard deviation of the Gaussian soft chroma assignment.

    Returns:
        Chroma tensor of shape [Batch, Seq_Len, 12] with values in [0, 1].
    """
    voiced_mask = (f0 > 10.0).float()  # Assume values below 10 Hz are unvoiced
    # Replace non-voiced frequencies with 440.0 Hz to avoid log2 of <= 0
    safe_f0 = torch.where(voiced_mask > 0, f0, torch.ones_like(f0) * 440.0)
    midi = 12.0 * torch.log2(safe_f0 / 440.0) + 69.0

    # Calculate fractional pitch class (0 to 11)
    pitch_class = midi % 12.0

    # Compute circular distance to each of the 12 chroma bins
    bins = torch.arange(12, dtype=f0.dtype, device=f0.device).view(1, 1, 12)
    pc_expanded = pitch_class.unsqueeze(-1)

    diff = torch.abs(pc_expanded - bins)
    circular_diff = torch.minimum(diff, 12.0 - diff)

    # Gaussian soft assignment
    chroma = torch.exp(-0.5 * (circular_diff / sigma) ** 2)

    # Normalize chroma vector at each voiced frame
    chroma_sum = chroma.sum(dim=-1, keepdim=True) + 1e-8
    chroma = chroma / chroma_sum

    # Zero out unvoiced frames
    chroma = chroma * voiced_mask.unsqueeze(-1)
    return chroma


class CAPHAligner(nn.Module):
    """Cross-Attention Pitch-to-Harmony (CAPH) Aligner for latent music diffusion.

    Aligns instrumental hidden states with vocal pitch class chroma using
    localized cross-attention and computes Chord Distance Loss (CDL).
    """

    def __init__(self, dit_dim: int, num_heads: int = 8, chroma_dim: int = 12):
        """Initialize CAPHAligner block.

        Args:
            dit_dim: Hidden dimension of the Diffusion Transformer block.
            num_heads: Number of attention heads.
            chroma_dim: Dimension of pitch chroma vectors (default: 12).
        """
        super().__init__()
        self.dit_dim = dit_dim
        self.num_heads = num_heads
        self.head_dim = dit_dim // num_heads
        self.scale = self.head_dim**-0.5

        # Projection layers for cross-attention
        self.vocal_k_proj = nn.Linear(chroma_dim, dit_dim, bias=False)
        self.vocal_v_proj = nn.Linear(chroma_dim, dit_dim, bias=False)

        # Instrumental chroma probe to map DiT hidden states back to chroma space
        self.inst_chroma_probe = nn.Linear(dit_dim, chroma_dim)

        # Register standard consonance/dissonance penalty matrix based on circle of fifths
        self.register_buffer("M", self._generate_music_theory_matrix())

    def _generate_music_theory_matrix(self) -> torch.Tensor:
        """Generate the stationary 12x12 Music Theory Penalty Matrix.

        Returns:
            A 12x12 tensor representing interval consonance/dissonance rules.
        """
        matrix = torch.zeros(12, 12)
        for i in range(12):
            for j in range(12):
                interval = abs(i - j) % 12
                if interval in (0, 3, 4, 7, 9):  # Unison, Minor/Major 3rd, 5th, Major 6th
                    matrix[i, j] = 0.0
                elif interval in (5, 8):  # Perfect 4th, Minor 6th
                    matrix[i, j] = 0.4
                elif interval in (2, 10):  # Major 2nd, Minor 7th
                    matrix[i, j] = 0.7
                else:  # Minor 2nd, Tritone, Major 7th
                    matrix[i, j] = 1.0
        return matrix

    def forward(
        self,
        x_dit: torch.Tensor,
        vocal_chroma: torch.Tensor,
        local_window_size: Optional[int] = 5,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Align instrumental latents with vocal chroma and compute Chord Distance Loss.

        Args:
            x_dit: Latent instrumental hidden states from DiT, shape [Batch, Seq_Len, Dim].
            vocal_chroma: Vocal pitch class chroma sequence, shape [Batch, Seq_Len, 12].
            local_window_size: Odd integer for localized sliding attention window. None for global.

        Returns:
            Tuple containing:
            - Attention output tensor of shape [Batch, Seq_Len, Dim].
            - Chord Distance Loss scalar tensor.
        """
        batch_size, seq_len, dim = x_dit.shape

        # 1. Project vocal chroma to DiT embedding keys and values
        k_vocal = self.vocal_k_proj(vocal_chroma)  # [B, T, D]
        v_vocal = self.vocal_v_proj(vocal_chroma)  # [B, T, D]

        # Reshape to [Batch, Num_Heads, Seq_Len, Head_Dim]
        q_inst = x_dit.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k_vocal = k_vocal.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v_vocal = v_vocal.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Compute raw attention scores
        attn_scores = torch.matmul(q_inst, k_vocal.transpose(-2, -1)) * self.scale  # [B, H, T, T]

        # 2. Localized cross-attention masking
        if local_window_size is not None:
            # Create a localized diagonal band mask
            indices = torch.arange(seq_len, device=x_dit.device)
            diff = torch.abs(indices.unsqueeze(1) - indices.unsqueeze(0))  # [T, T]
            mask = diff > (local_window_size // 2)
            # Expand to [1, 1, T, T]
            mask = mask.unsqueeze(0).unsqueeze(0)
            attn_scores = attn_scores.masked_fill(mask, float("-inf"))

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v_vocal)  # [B, H, T, Head_Dim]

        # Reshape back to [Batch, Seq_Len, Dim]
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, dim)

        # 3. Chord Distance Loss (CDL) computation
        # Project DiT hidden states to instrumental chroma distribution
        inst_chroma_logits = self.inst_chroma_probe(x_dit)  # [B, T, 12]
        inst_chroma_pred = F.softmax(inst_chroma_logits, dim=-1)

        # Loss = Inst^T * M * Vocal
        # Compute quadratic product across batch and sequence length
        penalty = torch.einsum("bti,ij,btj->bt", inst_chroma_pred, self.M, vocal_chroma)
        cdl_loss = penalty.mean()

        return attn_output, cdl_loss
