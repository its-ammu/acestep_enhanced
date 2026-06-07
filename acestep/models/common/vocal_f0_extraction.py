"""Vocal f0 extraction and chroma conversion for CAPH alignment.

Extracts fundamental frequency (f0) from a vocal audio waveform using
torchaudio's pitch detection, then converts to a 12-dimensional chroma
vector via :func:`caph_aligner.f0_to_chroma`.

Designed for inference-time use: accepts raw waveform or file path, returns
a chroma tensor ready to feed into the CAPH-steered sampling loop.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import torch
import torchaudio
from loguru import logger

from .caph_aligner import f0_to_chroma


_DEFAULT_SAMPLE_RATE = 44100
_F0_FRAME_HOP_MS = 10  # 10ms hop for f0 estimation


def _load_audio(
    audio: Union[str, Path, torch.Tensor],
    sample_rate: Optional[int],
) -> tuple[torch.Tensor, int]:
    """Load audio from path or validate tensor input.

    Args:
        audio: File path string/Path or waveform tensor ``[channels, samples]``.
        sample_rate: Sample rate (required when audio is a tensor).

    Returns:
        Tuple of (mono waveform ``[1, samples]``, sample_rate).
    """
    if isinstance(audio, (str, Path)):
        waveform, sr = torchaudio.load(str(audio))
    else:
        if sample_rate is None:
            raise ValueError("sample_rate is required when audio is a tensor")
        waveform, sr = audio, sample_rate

    # Convert to mono if stereo
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    return waveform, sr


def _extract_f0_yin(
    waveform: torch.Tensor,
    sample_rate: int,
    frame_hop_ms: float = _F0_FRAME_HOP_MS,
) -> torch.Tensor:
    """Extract f0 using torchaudio's YIN-based pitch detection.

    Args:
        waveform: Mono audio tensor ``[1, samples]``.
        sample_rate: Audio sample rate in Hz.
        frame_hop_ms: Hop size in milliseconds between f0 frames.

    Returns:
        f0 tensor of shape ``[1, num_frames]`` in Hz.
    """
    frame_length = int(sample_rate * 0.04)  # 40ms window
    hop_length = int(sample_rate * frame_hop_ms / 1000.0)
    # torchaudio.functional.detect_pitch_frequency uses autocorrelation-based YIN
    f0 = torchaudio.functional.detect_pitch_frequency(
        waveform, sample_rate,
        frame_time=frame_hop_ms / 1000.0,
        freq_low=50, freq_high=2000,
    )
    return f0


def extract_vocal_chroma(
    audio: Union[str, Path, torch.Tensor],
    sample_rate: Optional[int] = None,
    target_length: Optional[int] = None,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Extract vocal chroma from an audio source.

    End-to-end: load audio → extract f0 → convert to chroma → optionally
    resample to match the diffusion latent sequence length.

    Args:
        audio: File path or waveform tensor.
        sample_rate: Required if audio is a tensor.
        target_length: If set, interpolate chroma to this sequence length.
        device: Target device for the output tensor.
        dtype: Target dtype for the output tensor.

    Returns:
        Chroma tensor of shape ``[1, T, 12]``.
    """
    waveform, sr = _load_audio(audio, sample_rate)
    logger.info(
        "[vocal_f0] extracting f0 — sr={}, duration={:.2f}s",
        sr, waveform.shape[-1] / sr,
    )
    f0 = _extract_f0_yin(waveform, sr)  # [1, num_frames]
    chroma = f0_to_chroma(f0)  # [1, num_frames, 12]

    if target_length is not None and chroma.shape[1] != target_length:
        # Interpolate: [1, T, 12] -> [1, 12, T] -> interp -> [1, 12, T'] -> [1, T', 12]
        chroma = chroma.permute(0, 2, 1)
        chroma = torch.nn.functional.interpolate(
            chroma.float(), size=target_length, mode="linear", align_corners=False,
        )
        chroma = chroma.permute(0, 2, 1)

    if device is not None:
        chroma = chroma.to(device=device)
    if dtype is not None:
        chroma = chroma.to(dtype=dtype)

    logger.info("[vocal_f0] chroma shape={}", list(chroma.shape))
    return chroma
