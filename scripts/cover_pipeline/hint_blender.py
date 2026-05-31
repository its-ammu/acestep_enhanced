"""Semantic hint blending: combine chord-accurate and timbre-neutral hints.

Blends semantic hints from the original audio (chord-accurate but
timbre-locked) with hints from a synthesized source (chord-accurate
with neutral timbre). The blended hints enable the model to generate
correct chords with different instrument character.

Adapted from RyanOnTheInside's ACEStep15SemanticHintsBlend node.
"""

from pathlib import Path
from typing import Union

import torch
from loguru import logger


def blend_hints(
    chord_hints: torch.Tensor,
    timbre_hints: torch.Tensor,
    blend_factor: Union[float, list[float], torch.Tensor],
) -> torch.Tensor:
    """Blend semantic hints from two sources via linear interpolation.

    Formula: blended = (1 - factor) * chord_hints + factor * timbre_hints

    At factor=0.0, output equals chord_hints (original timbre, correct chords).
    At factor=1.0, output equals timbre_hints (neutral timbre, correct chords).

    Args:
        chord_hints: Hints from original audio. Shape [B, T, D].
        timbre_hints: Hints from synthesized source. Shape [B, T, D].
        blend_factor: Scalar float, list of per-frame floats, or 1D tensor.

    Returns:
        Blended hints tensor [B, T, D].

    Raises:
        ValueError: If shapes mismatch, blend_factor out of range, or
            per-frame array length doesn't match temporal dimension.
    """
    # Shape validation
    if chord_hints.shape != timbre_hints.shape:
        raise ValueError(
            f"Hint tensor shapes must match. "
            f"chord_hints: {chord_hints.shape}, timbre_hints: {timbre_hints.shape}"
        )

    T = chord_hints.shape[1]  # Temporal dimension (hints are [B, T, D])

    if isinstance(blend_factor, (int, float)):
        # Scalar blend
        factor = float(blend_factor)
        if factor < 0.0 or factor > 1.0:
            raise ValueError(
                f"blend_factor must be in [0.0, 1.0], got {factor}"
            )
        blended = (1.0 - factor) * chord_hints + factor * timbre_hints

    elif isinstance(blend_factor, list):
        # Per-frame blend from list
        if len(blend_factor) != T:
            raise ValueError(
                f"Per-frame weight array length ({len(blend_factor)}) "
                f"must equal temporal dimension T ({T})"
            )
        # Validate all elements in range
        out_of_range = [
            (i, v) for i, v in enumerate(blend_factor)
            if v < 0.0 or v > 1.0
        ]
        if out_of_range:
            indices = [str(i) for i, _ in out_of_range[:5]]
            raise ValueError(
                f"Per-frame weights must be in [0.0, 1.0]. "
                f"Out-of-range at indices: {', '.join(indices)}"
            )
        # Shape [1, T, 1] to broadcast across batch and feature dims
        weight = torch.tensor(
            blend_factor, dtype=chord_hints.dtype, device=chord_hints.device
        ).reshape(1, T, 1)
        blended = (1.0 - weight) * chord_hints + weight * timbre_hints

    elif isinstance(blend_factor, torch.Tensor):
        # Per-frame blend from tensor
        if blend_factor.numel() != T:
            raise ValueError(
                f"Per-frame weight tensor length ({blend_factor.numel()}) "
                f"must equal temporal dimension T ({T})"
            )
        weight = blend_factor.to(
            dtype=chord_hints.dtype, device=chord_hints.device
        ).reshape(1, T, 1)
        blended = (1.0 - weight) * chord_hints + weight * timbre_hints

    else:
        raise ValueError(
            f"blend_factor must be float, list[float], or torch.Tensor, "
            f"got {type(blend_factor)}"
        )

    logger.debug(
        f"Hint blending: chord_hints mean={chord_hints.mean():.4f}, "
        f"timbre_hints mean={timbre_hints.mean():.4f}, "
        f"blended mean={blended.mean():.4f}"
    )

    return blended


def extract_and_blend(
    handler,
    chord_source_path: str | Path,
    timbre_source_path: str | Path,
    blend_factor: float = 0.5,
) -> torch.Tensor:
    """Extract hints from both sources and blend them.

    Convenience function that handles extraction, length alignment, and blending.

    Args:
        handler: Initialized AceStepHandler with model loaded.
        chord_source_path: Path to chord-accurate audio (original bass stem).
        timbre_source_path: Path to timbre-neutral audio (MIDI piano rendering).
        blend_factor: Blend weight (0.0=original, 1.0=synthesized).

    Returns:
        Blended hints tensor [B, T, D].

    Raises:
        ValueError: If audio files don't exist or are too short.
    """
    from .semantic_cover import extract_semantic_hints

    chord_source_path = Path(chord_source_path)
    timbre_source_path = Path(timbre_source_path)

    if not chord_source_path.exists():
        raise ValueError(f"Chord source not found: {chord_source_path}")
    if not timbre_source_path.exists():
        raise ValueError(f"Timbre source not found: {timbre_source_path}")

    # Extract hints from both sources
    logger.info(f"Extracting chord hints from: {chord_source_path.name}")
    chord_hints = extract_semantic_hints(handler, str(chord_source_path))

    logger.info(f"Extracting timbre hints from: {timbre_source_path.name}")
    timbre_hints = extract_semantic_hints(handler, str(timbre_source_path))

    # Align temporal dimensions (truncate to shorter)
    T_chord = chord_hints.shape[1]
    T_timbre = timbre_hints.shape[1]

    if T_chord != T_timbre:
        min_T = min(T_chord, T_timbre)
        logger.info(
            f"Hint length mismatch: chord={T_chord}, timbre={T_timbre}. "
            f"Truncating to {min_T} frames."
        )
        chord_hints = chord_hints[:, :min_T, :]
        timbre_hints = timbre_hints[:, :min_T, :]

    # Blend
    blended = blend_hints(chord_hints, timbre_hints, blend_factor)
    logger.info(
        f"Blended hints: shape={blended.shape}, factor={blend_factor}"
    )

    return blended
