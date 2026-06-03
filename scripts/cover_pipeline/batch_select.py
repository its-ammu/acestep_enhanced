"""Batch generation with automated quality scoring and selection.

Generates N versions of the cover, scores each on:
1. QC pass rate (chord accuracy)
2. Spectral flux (how dynamic/varied the audio is)
3. Dynamic range (loudness contrast between sections)

Selects the best version automatically.
"""

from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
from loguru import logger


def score_spectral_flux(audio_path: str, sr: int = 22050) -> float:
    """Measure average spectral flux — higher = more dynamic/varied.

    Args:
        audio_path: Path to audio file.
        sr: Sample rate for analysis.

    Returns:
        Mean spectral flux value (higher = more varied over time).
    """
    import librosa

    y, _ = librosa.load(audio_path, sr=sr, mono=True)
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))
    flux = np.sqrt(np.mean(np.diff(S, axis=1) ** 2, axis=0))
    return float(np.mean(flux))


def score_dynamic_range(
    audio_path: str, segments: list, sr: int = 22050
) -> float:
    """Measure loudness contrast between sections — higher = more dynamic.

    Compares RMS of quietest section vs loudest section.

    Args:
        audio_path: Path to audio file.
        segments: Section boundaries from SongFormer.
        sr: Sample rate.

    Returns:
        Dynamic range in dB (difference between loudest and quietest section).
    """
    import librosa

    y, _ = librosa.load(audio_path, sr=sr, mono=True)

    section_rms = []
    for seg in segments:
        start_sample = int(seg["start"] * sr)
        end_sample = int(seg["end"] * sr)
        section = y[start_sample:end_sample]
        if len(section) > 0:
            rms = np.sqrt(np.mean(section ** 2)) + 1e-10
            section_rms.append(20 * np.log10(rms))

    if len(section_rms) < 2:
        return 0.0

    return float(max(section_rms) - min(section_rms))


def score_version(
    audio_path: str,
    segments: list,
    bpm: int,
    key: str,
    original_audio_path: str,
) -> dict:
    """Score a generated version on all metrics.

    Args:
        audio_path: Path to generated audio.
        segments: Section boundaries.
        bpm: BPM.
        key: Key string.
        original_audio_path: Path to original instrumental.

    Returns:
        Dict with scores: qc_pass_rate, spectral_flux, dynamic_range, combined.
    """
    from .post_gen_qc import analyze_generated_audio, get_failing_sections

    # QC
    qc_results = analyze_generated_audio(
        audio_path=audio_path,
        bpm=bpm,
        key=key,
        segments=segments,
        original_audio_path=original_audio_path,
    )
    total = len(qc_results) if qc_results else 1
    failing = len(get_failing_sections(qc_results)) if qc_results else total
    qc_pass_rate = (total - failing) / total

    # Spectral flux
    flux = score_spectral_flux(audio_path)

    # Dynamic range
    dyn_range = score_dynamic_range(audio_path, segments)

    # Combined score (weighted)
    # QC is most important (chords must be right)
    # Then dynamic range (energy contrast)
    # Then spectral flux (variety)
    combined = qc_pass_rate * 0.5 + (dyn_range / 20.0) * 0.3 + (flux / 0.05) * 0.2

    return {
        "qc_pass_rate": qc_pass_rate,
        "spectral_flux": flux,
        "dynamic_range_db": dyn_range,
        "combined": combined,
    }


def select_best_version(versions: list[dict]) -> int:
    """Select the best version from scored candidates.

    Args:
        versions: List of score dicts from score_version().

    Returns:
        Index of the best version.
    """
    best_idx = 0
    best_score = -1.0

    for i, v in enumerate(versions):
        if v["combined"] > best_score:
            best_score = v["combined"]
            best_idx = i

    return best_idx
