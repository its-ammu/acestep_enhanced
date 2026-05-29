"""SongFormer music structure analysis — public API.

Provides analyze_structure() and analyze_and_unload() entry points.
"""

import os
from pathlib import Path

import torch
from loguru import logger

from .songformer_inference import INPUT_SR, run_inference
from .songformer_setup import (
    SongFormerStack,
    load_models,
    unload_models,
)


def analyze_structure(
    stack: SongFormerStack,
    audio_path: str,
    device: str = "cuda:0",
) -> list[dict]:
    """Run SongFormer inference on a single audio file.

    Args:
        stack: Loaded SongFormerStack from load_models().
        audio_path: Path to the audio file.
        device: CUDA device string.

    Returns:
        List of segment dicts with "label", "start", "end" keys.
    """
    import librosa

    audio_path = str(Path(audio_path).resolve())
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    logger.info(f"SongFormer: analyzing {audio_path}")
    wav, _ = librosa.load(audio_path, sr=INPUT_SR)
    audio = torch.tensor(wav).to(device)

    segments = run_inference(stack, audio, device)
    logger.info(f"SongFormer: found {len(segments)} segments")
    return segments


def analyze_and_unload(
    audio_path: str,
    device: str = "cuda:0",
) -> list[dict]:
    """Load SongFormer, analyze one file, unload. Convenience wrapper.

    Args:
        audio_path: Path to the audio file.
        device: CUDA device string.

    Returns:
        List of segment dicts with "label", "start", "end" keys.
    """
    stack = load_models(device)
    try:
        return analyze_structure(stack, audio_path, device)
    finally:
        unload_models(stack)
