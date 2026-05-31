"""Chord detection with confidence scoring for hint blending suitability.

Wraps audio_analysis.analyze_chords() with confidence computation and
skip-signal logic. Used by the hint blending stage to determine whether
a song's chord progression is reliable enough for MIDI rendering.
"""

from dataclasses import dataclass, field
from pathlib import Path

from loguru import logger

from .audio_analysis import ChordSegment, analyze_chords


# Recognized chord qualities for confidence scoring
_RECOGNIZED_QUALITIES = {
    "maj", "min", "dim", "aug", "sus2", "sus4",
    "7", "maj7", "min7", "dim7", "hdim7",
    "N",  # No-chord / silence
}


@dataclass
class ChordDetectionResult:
    """Result of chord detection with confidence metadata."""

    chords: list[ChordSegment] = field(default_factory=list)
    confidence: float = 0.0
    should_skip: bool = False
    duration: float = 0.0


def detect_chords(audio_path: str | Path, max_duration: float = 600.0) -> ChordDetectionResult:
    """Detect chords and compute confidence for hint blending suitability.

    Args:
        audio_path: Path to audio file (WAV or FLAC).
        max_duration: Maximum duration to process (seconds). Truncates longer files.

    Returns:
        ChordDetectionResult with chords, confidence, and skip flag.
    """
    audio_path = Path(audio_path)

    if not audio_path.exists():
        logger.warning(f"Chord detection: file not found: {audio_path}")
        return ChordDetectionResult(should_skip=True)

    # Get duration
    try:
        import librosa
        duration = librosa.get_duration(path=str(audio_path))
    except Exception as e:
        logger.warning(f"Chord detection: cannot read duration: {e}")
        return ChordDetectionResult(should_skip=True)

    # Short audio gate
    if duration < 5.0:
        logger.warning(f"Chord detection: audio too short ({duration:.1f}s < 5s), skipping")
        return ChordDetectionResult(duration=duration, should_skip=True)

    # Non-tonal content gate (spectral flatness)
    if _is_non_tonal(audio_path):
        logger.warning("Chord detection: non-tonal content (spectral flatness > 0.9), skipping")
        return ChordDetectionResult(duration=duration, should_skip=True)

    # Truncation warning
    if duration > max_duration:
        logger.warning(
            f"Chord detection: audio exceeds {max_duration}s "
            f"({duration:.1f}s), processing first {max_duration}s only"
        )

    # Run chord detection
    chords = analyze_chords(audio_path)

    if not chords:
        logger.warning("Chord detection: no chords detected")
        return ChordDetectionResult(duration=duration, should_skip=True)

    # Filter to max_duration
    if duration > max_duration:
        chords = [c for c in chords if c.start < max_duration]
        duration = max_duration

    # Compute confidence: proportion of duration covered by recognized chords
    confidence = _compute_confidence(chords, duration)

    should_skip = confidence < 0.5
    if should_skip:
        logger.warning(
            f"Chord detection: low confidence ({confidence:.2f} < 0.5), "
            f"hint blending will be skipped"
        )
    else:
        logger.info(
            f"Chord detection: {len(chords)} segments, "
            f"confidence={confidence:.2f}, duration={duration:.1f}s"
        )

    return ChordDetectionResult(
        chords=chords,
        confidence=confidence,
        should_skip=should_skip,
        duration=duration,
    )


def _compute_confidence(chords: list[ChordSegment], total_duration: float) -> float:
    """Compute confidence as proportion of duration with recognized chord labels.

    Args:
        chords: List of detected chord segments.
        total_duration: Total audio duration in seconds.

    Returns:
        Confidence score between 0.0 and 1.0.
    """
    if total_duration <= 0 or not chords:
        return 0.0

    recognized_duration = 0.0
    for chord in chords:
        if _is_recognized_label(chord.chord):
            recognized_duration += chord.end - chord.start

    return min(recognized_duration / total_duration, 1.0)


def _is_recognized_label(label: str) -> bool:
    """Check if a chord label matches the recognized vocabulary.

    Handles formats: "C:maj", "F#:min7", "N", "Bb:dim", "C", "Am", etc.
    """
    if not label or label.strip() == "":
        return False

    label = label.strip()

    # "N" = no chord (recognized as valid)
    if label == "N":
        return True

    # madmom format: "root:quality" (e.g., "C:maj", "F#:min7")
    if ":" in label:
        parts = label.split(":", 1)
        quality = parts[1].lower() if len(parts) > 1 else ""
        # Check against known qualities
        return quality in _RECOGNIZED_QUALITIES or quality in {
            "min", "maj", "dim", "aug", "7", "maj7", "min7",
            "dim7", "hdim7", "sus2", "sus4", "minmaj7",
        }

    # Simple format: "C", "Am", "F#m", "Bb7"
    # If it starts with a valid note name, consider it recognized
    note_names = {"C", "D", "E", "F", "G", "A", "B"}
    if label[0].upper() in note_names:
        return True

    return False


def _is_non_tonal(audio_path: Path) -> bool:
    """Check if audio is non-tonal using spectral flatness.

    Returns True if average spectral flatness > 0.9 (noise-like content).
    """
    try:
        import librosa
        import numpy as np

        y, sr = librosa.load(str(audio_path), sr=22050, duration=60)
        flatness = librosa.feature.spectral_flatness(y=y)[0]
        avg_flatness = float(np.mean(flatness))
        return avg_flatness > 0.9
    except Exception:
        return False
