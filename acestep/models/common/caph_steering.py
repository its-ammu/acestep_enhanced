"""CAPH gradient steering helpers for flow-edit sampling loops.

Provides inference-time gradient steering using the CAPHAligner's Chord
Distance Loss (CDL) to guide diffusion latents toward harmonic consonance
with a vocal chroma reference.  Kept separate from flow_edit.py to honour
the 200 LOC module cap.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from loguru import logger

from .caph_aligner import CAPHAligner


def maybe_create_caph_aligner(
    latent_dim: int,
    vocal_chroma: Optional[torch.Tensor],
    cdl_guidance_scale: float,
    device: torch.device,
    dtype: torch.dtype,
) -> Optional[CAPHAligner]:
    """Instantiate a CAPHAligner if steering is requested, else return None.

    Args:
        latent_dim: Channel dimension of the diffusion latents (``src_latents.shape[-1]``).
        vocal_chroma: Pre-computed vocal chroma tensor, or None.
        cdl_guidance_scale: Steering strength (0.0 = disabled).
        device: Target device for the aligner parameters.
        dtype: Target dtype for the aligner parameters.

    Returns:
        A ready-to-use CAPHAligner on the correct device/dtype, or None.
    """
    if cdl_guidance_scale <= 0.0 or vocal_chroma is None:
        return None

    # Pick num_heads such that latent_dim is divisible. Prefer 8, fallback to 4, 2, 1.
    num_heads = 8
    for candidate in (8, 4, 2, 1):
        if latent_dim % candidate == 0:
            num_heads = candidate
            break

    aligner = CAPHAligner(dit_dim=latent_dim, num_heads=num_heads)
    aligner = aligner.to(device=device, dtype=dtype)
    aligner.eval()
    logger.info(
        "[caph_steering] created CAPHAligner — latent_dim={}, heads={}, "
        "cdl_scale={:.3f}, chroma_shape={}",
        latent_dim, num_heads, cdl_guidance_scale, list(vocal_chroma.shape),
    )
    return aligner


def apply_caph_gradient_steering(
    zt_edit: torch.Tensor,
    vocal_chroma: torch.Tensor,
    caph_aligner: nn.Module,
    cdl_guidance_scale: float,
) -> torch.Tensor:
    """Apply one gradient-steering step to the latent using CDL.

    Computes ``zt_edit <- zt_edit - lambda * grad(CDL)`` where lambda is
    ``cdl_guidance_scale``.  Uses ``torch.enable_grad()`` inside
    ``@torch.no_grad()`` context to compute the single gradient without
    accumulating graph history across steps.

    Args:
        zt_edit: Current diffusion latent, shape ``[B, T, D]``.
        vocal_chroma: Vocal pitch chroma, shape ``[B, T_chroma, 12]``.
        caph_aligner: Initialised CAPHAligner module.
        cdl_guidance_scale: Gradient step size (lambda).

    Returns:
        Steered latent tensor (same shape as input).
    """
    # Align temporal dimensions if they differ (interpolate chroma to latent length).
    if vocal_chroma.shape[1] != zt_edit.shape[1]:
        # [B, T_chroma, 12] -> [B, 12, T_chroma] -> interpolate -> [B, 12, T] -> [B, T, 12]
        vc = vocal_chroma.permute(0, 2, 1)
        vc = torch.nn.functional.interpolate(vc, size=zt_edit.shape[1], mode="linear", align_corners=False)
        vocal_chroma = vc.permute(0, 2, 1)

    with torch.enable_grad():
        zt_steer = zt_edit.detach().requires_grad_(True)
        _, cdl_loss = caph_aligner(zt_steer, vocal_chroma)
        cdl_grad = torch.autograd.grad(cdl_loss, zt_steer)[0]

    return zt_edit - cdl_guidance_scale * cdl_grad
