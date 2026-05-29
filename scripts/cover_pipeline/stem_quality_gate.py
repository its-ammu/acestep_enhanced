"""Stem quality gate: detect and replace musically incorrect AI stems.

Compares AI-generated stems against original stems using chroma and
rhythm correlation. Swaps only stems that are musically wrong (wrong
notes/rhythm), not stylistically different (different timbre).

Rule: swap at most ONE stem to avoid reverting to the original song.
If multiple stems are bad, flag for regeneration instead.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
from loguru import logger


@dataclass
class StemScore:
    """Quality score for a single stem comparison."""

    name: str
    chroma_correlation: float
    rhythm_correlation: float
    combined_score: float
    needs_swap: bool


@dataclass
class QualityGateResult:
    """Result of the stem quality gate."""

    scores: list[StemScore]
    swap_stem: Optional[str] = None  # Name of stem to swap (None = all OK)
    needs_regeneration: bool = False  # True if multiple stems are bad


def _load_audio_mono(path: str | Path, sr: int = 22050) -> np.ndarray:
    """Load audio as mono at target sample rate."""
    import librosa

    y, _ = librosa.load(str(path), sr=sr, mono=True)
    return y


def _chroma_correlation(audio_a: np.ndarray, audio_b: np.ndarray, sr: int = 22050) -> float:
    """Compute chroma feature correlation between two audio signals.

    Measures harmonic/note similarity. High = same chords/notes.
    """
    import librosa

    # Match lengths
    min_len = min(len(audio_a), len(audio_b))
    audio_a = audio_a[:min_len]
    audio_b = audio_b[:min_len]

    if min_len < sr:  # Less than 1 second
        return 0.5  # Neutral score for very short clips

    chroma_a = librosa.feature.chroma_cqt(y=audio_a, sr=sr)
    chroma_b = librosa.feature.chroma_cqt(y=audio_b, sr=sr)

    # Match time frames
    min_frames = min(chroma_a.shape[1], chroma_b.shape[1])
    chroma_a = chroma_a[:, :min_frames]
    chroma_b = chroma_b[:, :min_frames]

    # Flatten and correlate
    flat_a = chroma_a.flatten()
    flat_b = chroma_b.flatten()

    if np.std(flat_a) < 1e-8 or np.std(flat_b) < 1e-8:
        return 0.5  # Can't compute correlation on silence

    correlation = np.corrcoef(flat_a, flat_b)[0, 1]
    return float(max(0.0, correlation))  # Clamp negative to 0


def _rhythm_correlation(audio_a: np.ndarray, audio_b: np.ndarray, sr: int = 22050) -> float:
    """Compute onset/rhythm pattern correlation between two audio signals.

    Measures timing similarity. High = same beat placement.
    """
    import librosa

    min_len = min(len(audio_a), len(audio_b))
    audio_a = audio_a[:min_len]
    audio_b = audio_b[:min_len]

    if min_len < sr:
        return 0.5

    # Compute onset strength envelopes
    onset_a = librosa.onset.onset_strength(y=audio_a, sr=sr)
    onset_b = librosa.onset.onset_strength(y=audio_b, sr=sr)

    # Match lengths
    min_frames = min(len(onset_a), len(onset_b))
    onset_a = onset_a[:min_frames]
    onset_b = onset_b[:min_frames]

    if np.std(onset_a) < 1e-8 or np.std(onset_b) < 1e-8:
        return 0.5

    correlation = np.corrcoef(onset_a, onset_b)[0, 1]
    return float(max(0.0, correlation))


def evaluate_stems(
    original_stems: dict[str, Path],
    ai_stems: dict[str, Path],
    chroma_threshold: float = 0.3,
    rhythm_threshold: float = 0.25,
) -> QualityGateResult:
    """Compare AI stems against originals and identify bad stems.

    Args:
        original_stems: Dict of stem_name -> path (from original song Demucs).
        ai_stems: Dict of stem_name -> path (from AI instrumental Demucs).
        chroma_threshold: Below this = wrong notes (swap candidate).
        rhythm_threshold: Below this = wrong rhythm (swap candidate).

    Returns:
        QualityGateResult with per-stem scores and swap recommendation.
    """
    scores = []

    for name in ["drums", "bass", "other"]:
        orig_path = original_stems.get(name)
        ai_path = ai_stems.get(name)

        if not orig_path or not ai_path:
            scores.append(StemScore(
                name=name, chroma_correlation=0.5,
                rhythm_correlation=0.5, combined_score=0.5, needs_swap=False,
            ))
            continue

        if not Path(orig_path).exists() or not Path(ai_path).exists():
            scores.append(StemScore(
                name=name, chroma_correlation=0.5,
                rhythm_correlation=0.5, combined_score=0.5, needs_swap=False,
            ))
            continue

        logger.info(f"Evaluating stem: {name}")
        orig_audio = _load_audio_mono(orig_path)
        ai_audio = _load_audio_mono(ai_path)

        chroma_corr = _chroma_correlation(orig_audio, ai_audio)
        rhythm_corr = _rhythm_correlation(orig_audio, ai_audio)

        # Combined score (weighted: chroma matters more for bass/other, rhythm for drums)
        if name == "drums":
            combined = 0.3 * chroma_corr + 0.7 * rhythm_corr
        elif name == "bass":
            combined = 0.6 * chroma_corr + 0.4 * rhythm_corr
        else:
            combined = 0.7 * chroma_corr + 0.3 * rhythm_corr

        # Determine if swap needed
        needs_swap = (chroma_corr < chroma_threshold) or (rhythm_corr < rhythm_threshold)

        scores.append(StemScore(
            name=name,
            chroma_correlation=chroma_corr,
            rhythm_correlation=rhythm_corr,
            combined_score=combined,
            needs_swap=needs_swap,
        ))

        logger.info(
            f"  {name}: chroma={chroma_corr:.3f}, rhythm={rhythm_corr:.3f}, "
            f"combined={combined:.3f}, swap={needs_swap}"
        )

    # Decision logic: swap at most ONE stem
    bad_stems = [s for s in scores if s.needs_swap]

    result = QualityGateResult(scores=scores)

    if len(bad_stems) == 0:
        logger.info("Quality gate: all stems pass ✅")
    elif len(bad_stems) == 1:
        result.swap_stem = bad_stems[0].name
        logger.info(f"Quality gate: swapping '{bad_stems[0].name}' with original")
    else:
        # Multiple bad stems — swap only the worst one
        worst = min(bad_stems, key=lambda s: s.combined_score)
        result.swap_stem = worst.name
        logger.warning(
            f"Quality gate: {len(bad_stems)} stems below threshold "
            f"({[s.name for s in bad_stems]}). Swapping worst: '{worst.name}'"
        )
        if len(bad_stems) == 3:
            result.needs_regeneration = True
            logger.warning("Quality gate: ALL stems bad — recommend regeneration")

    return result


def remix_with_swap(
    vocals_path: str | Path,
    ai_stems: dict[str, Path],
    original_stems: dict[str, Path],
    swap_stem: str,
    output_path: str | Path,
    original_mix_path: Optional[str | Path] = None,
    sr: int = 44100,
) -> Path:
    """Remix using AI stems but swapping one with the original.

    Args:
        vocals_path: Path to original vocal stem.
        ai_stems: Dict of AI stem paths (drums, bass, other).
        original_stems: Dict of original stem paths.
        swap_stem: Which stem to replace with original ("drums", "bass", or "other").
        output_path: Output file path.
        original_mix_path: Original mix for loudness reference.
        sr: Target sample rate.

    Returns:
        Path to the saved mix.
    """
    from pedalboard import Compressor, Limiter, Pedalboard

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load vocals
    vocals, sr_v = sf.read(str(vocals_path))
    if vocals.ndim == 1:
        vocals = np.stack([vocals, vocals], axis=-1)

    # Load stems — use AI for all except the swapped one
    stem_arrays = {}
    for name in ["drums", "bass", "other"]:
        if name == swap_stem:
            path = original_stems.get(name)
            logger.info(f"  Using ORIGINAL {name}")
        else:
            path = ai_stems.get(name)
            logger.info(f"  Using AI {name}")

        if path and Path(path).exists():
            data, _ = sf.read(str(path))
            if data.ndim == 1:
                data = np.stack([data, data], axis=-1)
            stem_arrays[name] = data.astype(np.float32)

    # Match all lengths
    all_arrays = [vocals] + list(stem_arrays.values())
    min_len = min(len(a) for a in all_arrays)
    vocals = vocals[:min_len].astype(np.float32)
    for name in stem_arrays:
        stem_arrays[name] = stem_arrays[name][:min_len]

    # Sum instrumental stems
    instrumental = np.zeros_like(vocals)
    for data in stem_arrays.values():
        instrumental += data

    # Loudness match to original
    if original_mix_path and Path(original_mix_path).exists():
        original, _ = sf.read(str(original_mix_path))
        original = original[:min_len].astype(np.float32)
        if original.ndim == 1:
            original = np.stack([original, original], axis=-1)
        orig_rms = np.sqrt(np.mean(original**2)) + 1e-8
        mix_rms = np.sqrt(np.mean((vocals + instrumental)**2)) + 1e-8
        gain = orig_rms / mix_rms
        gain = np.clip(gain, 0.25, 4.0)
        instrumental = instrumental * gain

    # Mix
    mix = vocals + instrumental

    # Master bus
    master = Pedalboard([
        Compressor(threshold_db=-12.0, ratio=2.0, attack_ms=30.0, release_ms=200.0),
        Limiter(threshold_db=-1.0),
    ])
    mix_pb = mix.T.astype(np.float32)
    mix_pb = master(mix_pb, sr_v)
    mix = mix_pb.T

    sf.write(str(output_path), mix, sr_v)
    logger.info(f"Stem-swap mix saved: {output_path} (swapped: {swap_stem})")
    return output_path
