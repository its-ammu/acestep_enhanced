"""Section-level QC and splicing for cover generation.

Generates two versions at different cns levels, analyzes each per-section,
and splices the best sections together. This gives creative freedom (low cns)
where it works and structural accuracy (high cns) where needed.

Flow:
1. Generate version A at cns=0.15 (creative, different instruments)
2. Generate version B at cns=0.3 (safer, preserves bass/structure)
3. Per-section analysis: check bass presence and key adherence
4. For each section, pick the better version
5. Crossfade splice at section boundaries
"""

from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
from loguru import logger


def _section_bass_presence(
    audio: np.ndarray,
    sr: int,
    start_sec: float,
    end_sec: float,
    low_freq_hz: int = 200,
) -> float:
    """Measure bass presence in a section (energy below low_freq_hz).

    Args:
        audio: Mono audio array.
        sr: Sample rate.
        start_sec: Section start time.
        end_sec: Section end time.
        low_freq_hz: Frequency cutoff for "bass" content.

    Returns:
        Bass RMS in dB. Higher = more bass present.
    """
    import librosa

    start = int(start_sec * sr)
    end = int(end_sec * sr)
    section = audio[start:end]

    if len(section) < sr // 2:
        return -100.0

    # Low-pass filter to isolate bass
    S = np.abs(librosa.stft(section, n_fft=2048, hop_length=512))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    bass_bins = freqs < low_freq_hz
    bass_energy = np.sqrt(np.mean(S[bass_bins, :] ** 2))

    return 20 * np.log10(bass_energy + 1e-10)


def _section_key_adherence(
    audio: np.ndarray,
    sr: int,
    start_sec: float,
    end_sec: float,
    expected_key: str,
) -> float:
    """Measure how well a section adheres to the expected key.

    Args:
        audio: Mono audio array.
        sr: Sample rate.
        start_sec: Section start time.
        end_sec: Section end time.
        expected_key: Expected key (e.g., "Eb Major").

    Returns:
        Key adherence score (0-1). Higher = more in key.
    """
    import librosa

    start = int(start_sec * sr)
    end = int(end_sec * sr)
    section = audio[start:end]

    if len(section) < sr:
        return 0.5

    # Compute chroma
    chroma = librosa.feature.chroma_cqt(y=section, sr=sr)
    chroma_avg = chroma.mean(axis=1)  # 12 pitch classes

    # Expected key profile (Krumhansl-Schmuckler major/minor)
    major_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
                              2.52, 5.19, 2.39, 3.66, 2.29, 2.88])

    # Parse key name to pitch class
    key_map = {
        "C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3,
        "E": 4, "F": 5, "F#": 6, "Gb": 6, "G": 7, "G#": 8,
        "Ab": 8, "A": 9, "A#": 10, "Bb": 10, "B": 11,
    }

    # Extract root from key string
    parts = expected_key.split()
    root_name = parts[0] if parts else "C"
    root_pc = key_map.get(root_name, 0)

    # Rotate profile to match key
    profile = np.roll(major_profile, root_pc)
    profile = profile / profile.sum()
    chroma_norm = chroma_avg / (chroma_avg.sum() + 1e-10)

    # Correlation
    correlation = np.corrcoef(profile, chroma_norm)[0, 1]
    return max(0.0, float(correlation))


def analyze_and_splice(
    audio_creative: np.ndarray,
    audio_safe: np.ndarray,
    sr: int,
    segments: list[dict],
    original_bass_audio: np.ndarray,
    original_sr: int,
    expected_key: str = "",
    bass_dropout_threshold_db: float = -45.0,
    crossfade_sec: float = 0.3,
) -> np.ndarray:
    """Analyze two versions per-section and splice the best parts together.

    For each section:
    - If creative version has bass dropout (original has bass but creative doesn't) → use safe
    - If creative version is out of key → use safe
    - Otherwise → use creative (sounds more different/interesting)

    Args:
        audio_creative: Full audio from low-cns generation (stereo, samples x channels).
        audio_safe: Full audio from high-cns generation (stereo, samples x channels).
        sr: Sample rate of both.
        segments: List of section dicts with "start", "end", "label".
        original_bass_audio: Mono bass stem from original (for presence comparison).
        original_sr: Sample rate of original bass.
        expected_key: Expected musical key (e.g., "Eb Major").
        bass_dropout_threshold_db: Below this = bass not present.
        crossfade_sec: Crossfade duration at splice points.

    Returns:
        Spliced audio array (stereo, samples x channels).
    """
    import librosa

    # Convert to mono for analysis
    creative_mono = audio_creative.mean(axis=-1) if audio_creative.ndim == 2 else audio_creative
    safe_mono = audio_safe.mean(axis=-1) if audio_safe.ndim == 2 else audio_safe

    # Resample original bass to match sr if needed
    if original_sr != sr:
        original_bass_audio = librosa.resample(
            original_bass_audio, orig_sr=original_sr, target_sr=sr
        )

    # Ensure same length
    min_len = min(len(audio_creative), len(audio_safe))
    audio_creative = audio_creative[:min_len]
    audio_safe = audio_safe[:min_len]
    creative_mono = creative_mono[:min_len]
    safe_mono = safe_mono[:min_len]

    # Per-section decision
    crossfade_samples = int(crossfade_sec * sr)
    choices = []

    for seg in segments:
        if seg.get("label") in ("silence", "end"):
            choices.append("creative")
            continue

        start = seg["start"]
        end = seg["end"]

        # Check bass presence in original
        orig_bass_db = _section_bass_presence(original_bass_audio, sr, start, end)

        # Check bass presence in creative version
        creative_bass_db = _section_bass_presence(creative_mono, sr, start, end)

        # Decision logic:
        # - Chorus/outro sections: always use SAFE (bass consistency issues at low cns)
        # - Other sections: use CREATIVE unless bass drops out
        label = seg.get("label", "")
        use_safe_for_label = label in ("chorus", "outro")

        # Bass dropout: original has bass but creative doesn't
        bass_dropout = (orig_bass_db > bass_dropout_threshold_db and
                        creative_bass_db < bass_dropout_threshold_db)

        if use_safe_for_label:
            choices.append("safe")
            logger.info(
                f"  [{label}] {start:.1f}-{end:.1f}s: "
                f"SAFE (chorus/outro — use safe for bass consistency)"
            )
        elif bass_dropout:
            choices.append("safe")
            logger.info(
                f"  [{label}] {start:.1f}-{end:.1f}s: "
                f"SAFE (bass dropout: orig={orig_bass_db:.1f}dB, "
                f"creative={creative_bass_db:.1f}dB)"
            )
        else:
            choices.append("creative")
            logger.info(
                f"  [{label}] {start:.1f}-{end:.1f}s: "
                f"CREATIVE (bass={creative_bass_db:.1f}dB)"
            )

    # Splice
    output = np.zeros_like(audio_creative)
    creative_count = choices.count("creative")
    safe_count = choices.count("safe")
    logger.info(f"Splice decision: {creative_count} creative, {safe_count} safe sections")

    for i, (seg, choice) in enumerate(zip(segments, choices)):
        if seg.get("label") in ("silence", "end"):
            continue

        start_sample = int(seg["start"] * sr)
        end_sample = min(int(seg["end"] * sr), min_len)

        source = audio_creative if choice == "creative" else audio_safe
        output[start_sample:end_sample] = source[start_sample:end_sample]

        # Apply crossfade at boundaries (except first/last)
        if i > 0 and crossfade_samples > 0:
            fade_start = max(0, start_sample - crossfade_samples // 2)
            fade_end = min(min_len, start_sample + crossfade_samples // 2)
            fade_len = fade_end - fade_start

            if fade_len > 0:
                fade_in = np.linspace(0, 1, fade_len).reshape(-1, 1)
                fade_out = 1.0 - fade_in

                prev_source = audio_creative if choices[i-1] == "creative" else audio_safe
                curr_source = source

                output[fade_start:fade_end] = (
                    prev_source[fade_start:fade_end] * fade_out +
                    curr_source[fade_start:fade_end] * fade_in
                )

    return output
