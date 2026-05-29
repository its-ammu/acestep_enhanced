"""Post-generation quality control: per-bar analysis for cover fidelity.

Analyzes generated audio per-bar and flags sections that have:
1. Bass dropout (bars where bass energy drops to near-zero)
2. Out-of-key notes (pitch classes not in the detected key)
3. Wrong chord roots (chord root doesn't match original)

Used by the pipeline to reject bad sections and retry with new seeds.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from loguru import logger


# Key → valid pitch classes mapping
_KEY_TO_SCALE = {}

# Build all major/minor scales
_NOTE_TO_PC = {
    "C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3,
    "E": 4, "F": 5, "F#": 6, "Gb": 6, "G": 7, "G#": 8,
    "Ab": 8, "A": 9, "A#": 10, "Bb": 10, "B": 11,
}
_PC_TO_NOTE = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]
_MAJOR_INTERVALS = [0, 2, 4, 5, 7, 9, 11]
_MINOR_INTERVALS = [0, 2, 3, 5, 7, 8, 10]


def _get_scale_pcs(key_str: str) -> set[int]:
    """Get valid pitch classes for a key string like 'Eb Major' or 'C Minor'."""
    parts = key_str.strip().split()
    if len(parts) < 2:
        return set(range(12))  # All notes valid if can't parse

    root_name = parts[0]
    mode = parts[1].lower()

    root_pc = _NOTE_TO_PC.get(root_name)
    if root_pc is None:
        return set(range(12))

    intervals = _MAJOR_INTERVALS if "maj" in mode else _MINOR_INTERVALS
    return {(root_pc + i) % 12 for i in intervals}


def _get_chord_root(chroma_frame: np.ndarray) -> int:
    """Extract the dominant chord root from a chroma vector.

    Uses bass-weighted chroma: the root is typically the strongest
    low-frequency pitch class. Returns pitch class index (0-11).

    Args:
        chroma_frame: 12-element chroma vector (mean over a bar).

    Returns:
        Pitch class index of the chord root (0=C, 1=C#, ..., 11=B).
    """
    return int(np.argmax(chroma_frame))


def _chord_root_matches(root_a: int, root_b: int, tolerance: int = 0) -> bool:
    """Check if two chord roots match (with optional semitone tolerance).

    Args:
        root_a: Pitch class of chord root A.
        root_b: Pitch class of chord root B.
        tolerance: Allow this many semitones difference (0=exact match).

    Returns:
        True if roots match within tolerance.
    """
    diff = abs(root_a - root_b)
    # Handle wraparound (e.g., B=11 vs C=0 is 1 semitone, not 11)
    diff = min(diff, 12 - diff)
    return diff <= tolerance


@dataclass
class BarAnalysis:
    """Analysis result for a single bar."""

    bar_index: int
    start_sec: float
    end_sec: float
    bass_rms_db: float
    has_bass: bool
    out_of_key_energy: float  # Energy of out-of-key pitch classes
    out_of_key_notes: list[str] = field(default_factory=list)
    chord_root: int = -1  # Detected chord root (pitch class, -1=unknown)
    expected_root: int = -1  # Expected chord root from original
    chord_root_matches: bool = True  # Does the chord root match original?
    passes: bool = True


@dataclass
class SectionQCResult:
    """QC result for a section."""

    label: str
    start_sec: float
    end_sec: float
    bars: list[BarAnalysis] = field(default_factory=list)
    bass_dropout_bars: int = 0
    out_of_key_bars: int = 0
    chord_mismatch_bars: int = 0
    passes: bool = True


def analyze_generated_audio(
    audio_path: str | Path,
    bpm: int,
    key: str,
    segments: list[dict],
    original_audio_path: Optional[str | Path] = None,
    bass_dropout_threshold_db: float = -45.0,
    out_of_key_multiplier: float = 2.0,
    chord_root_check: bool = True,
) -> list[SectionQCResult]:
    """Analyze generated audio for bass dropout, out-of-key notes, and wrong chords.

    Uses the original instrumental as a baseline — only flags notes that are
    significantly MORE present in the generated output than in the original.
    Also compares chord roots per bar against the original.

    Args:
        audio_path: Path to generated instrumental audio.
        bpm: Detected BPM (for bar boundary calculation).
        key: Detected key (e.g., "Eb Major").
        segments: Section boundaries from SongFormer.
        original_audio_path: Path to original instrumental (for baseline comparison).
        bass_dropout_threshold_db: Below this = bass not playing.
        out_of_key_multiplier: Flag if out-of-key note is this many times
            stronger than in the original. Default 2.0 (double the baseline).
        chord_root_check: Whether to compare chord roots against original.

    Returns:
        List of SectionQCResult with per-bar analysis.
    """
    import librosa

    y, sr = librosa.load(str(audio_path), sr=22050, mono=True)

    # Calculate bar duration from BPM (4 beats per bar)
    bar_duration_sec = 4 * 60.0 / bpm
    bar_duration_samples = int(bar_duration_sec * sr)

    # Get valid pitch classes for this key
    valid_pcs = _get_scale_pcs(key)
    invalid_pcs = set(range(12)) - valid_pcs

    # Load original for baseline and chord root comparison
    y_orig = None
    original_baseline = {}
    original_chord_roots = {}  # {(section_start, bar_idx): root_pc}

    if original_audio_path and Path(original_audio_path).exists():
        y_orig, _ = librosa.load(str(original_audio_path), sr=22050, mono=True)
        for seg in segments:
            start = int(seg["start"] * sr)
            end = min(int(seg["end"] * sr), len(y_orig))
            section_orig = y_orig[start:end]
            if len(section_orig) > sr:
                chroma_orig = librosa.feature.chroma_cqt(y=section_orig, sr=sr)
                chroma_mean_orig = np.mean(chroma_orig, axis=1)
                total_orig = np.sum(chroma_mean_orig) + 1e-10
                # Per-note baseline ratio
                baseline = {pc: chroma_mean_orig[pc] / total_orig for pc in invalid_pcs}
                original_baseline[seg["start"]] = baseline

                # Extract chord roots per bar from original
                if chord_root_check:
                    num_bars_orig = max(1, int(len(section_orig) / bar_duration_samples))
                    for bar_idx in range(num_bars_orig):
                        bar_start = bar_idx * bar_duration_samples
                        bar_end = min((bar_idx + 1) * bar_duration_samples, len(section_orig))
                        bar_orig = section_orig[bar_start:bar_end]
                        if len(bar_orig) > sr // 4:
                            bar_chroma = librosa.feature.chroma_cqt(y=bar_orig, sr=sr)
                            bar_chroma_mean = np.mean(bar_chroma, axis=1)
                            root = _get_chord_root(bar_chroma_mean)
                            original_chord_roots[(seg["start"], bar_idx)] = root

    logger.info(
        f"QC: key={key}, valid notes={[_PC_TO_NOTE[pc] for pc in sorted(valid_pcs)]}, "
        f"invalid={[_PC_TO_NOTE[pc] for pc in sorted(invalid_pcs)]}"
    )
    logger.info(f"QC: bpm={bpm}, bar={bar_duration_sec:.2f}s, {len(segments)} sections")
    if original_baseline:
        logger.info(f"QC: using original as baseline (multiplier={out_of_key_multiplier}x)")
    if original_chord_roots:
        logger.info(f"QC: chord root comparison enabled ({len(original_chord_roots)} bars)")

    results = []

    for seg in segments:
        if seg.get("label") in ("silence", "end"):
            continue

        section_start = seg["start"]
        section_end = seg["end"]
        label = seg.get("label", "unknown")

        start_sample = int(section_start * sr)
        end_sample = int(section_end * sr)
        section_audio = y[start_sample:end_sample]

        if len(section_audio) < sr:
            results.append(SectionQCResult(
                label=label, start_sec=section_start, end_sec=section_end, passes=True
            ))
            continue

        # Get baseline for this section
        baseline = original_baseline.get(section_start, {pc: 0.08 for pc in invalid_pcs})

        # Analyze per-bar
        bars = []
        num_bars = max(1, int(len(section_audio) / bar_duration_samples))

        for bar_idx in range(num_bars):
            bar_start = bar_idx * bar_duration_samples
            bar_end = min((bar_idx + 1) * bar_duration_samples, len(section_audio))
            bar_audio = section_audio[bar_start:bar_end]

            if len(bar_audio) < sr // 4:
                continue

            bar_start_sec = section_start + bar_idx * bar_duration_sec
            bar_end_sec = bar_start_sec + bar_duration_sec

            # Bass check: low-frequency energy (below 200Hz)
            S = np.abs(librosa.stft(bar_audio, n_fft=2048, hop_length=512))
            freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
            bass_bins = freqs < 200
            bass_energy = np.sqrt(np.mean(S[bass_bins, :] ** 2))
            bass_rms_db = 20 * np.log10(bass_energy + 1e-10)
            has_bass = bass_rms_db > bass_dropout_threshold_db

            # Key check: compare against original baseline
            chroma = librosa.feature.chroma_cqt(y=bar_audio, sr=sr)
            chroma_mean = np.mean(chroma, axis=1)
            total_chroma = np.sum(chroma_mean) + 1e-10

            out_of_key_notes = []
            out_of_key_energy = 0.0
            for pc in invalid_pcs:
                note_ratio = chroma_mean[pc] / total_chroma
                baseline_ratio = baseline.get(pc, 0.08)
                # Only flag if significantly above baseline
                if note_ratio > baseline_ratio * out_of_key_multiplier:
                    out_of_key_notes.append(_PC_TO_NOTE[pc])
                    out_of_key_energy += note_ratio

            # Chord root check: compare to original
            chord_root = _get_chord_root(chroma_mean)
            expected_root = original_chord_roots.get((section_start, bar_idx), -1)
            root_matches = True
            if expected_root >= 0 and chord_root_check:
                root_matches = _chord_root_matches(chord_root, expected_root)

            bar_passes = (
                has_bass
                and len(out_of_key_notes) == 0
                and root_matches
            )

            bars.append(BarAnalysis(
                bar_index=bar_idx,
                start_sec=bar_start_sec,
                end_sec=bar_end_sec,
                bass_rms_db=bass_rms_db,
                has_bass=has_bass,
                out_of_key_energy=out_of_key_energy,
                out_of_key_notes=out_of_key_notes,
                chord_root=chord_root,
                expected_root=expected_root,
                chord_root_matches=root_matches,
                passes=bar_passes,
            ))

        # Section-level summary
        bass_dropout_bars = sum(1 for b in bars if not b.has_bass)
        out_of_key_bars = sum(1 for b in bars if b.out_of_key_notes)
        chord_mismatch_bars = sum(1 for b in bars if not b.chord_root_matches)
        section_passes = (
            bass_dropout_bars == 0
            and out_of_key_bars == 0
            and chord_mismatch_bars == 0
        )

        section_result = SectionQCResult(
            label=label,
            start_sec=section_start,
            end_sec=section_end,
            bars=bars,
            bass_dropout_bars=bass_dropout_bars,
            out_of_key_bars=out_of_key_bars,
            chord_mismatch_bars=chord_mismatch_bars,
            passes=section_passes,
        )
        results.append(section_result)

        status = "✅" if section_passes else "❌"
        issues = []
        if bass_dropout_bars > 0:
            issues.append(f"bass dropout in {bass_dropout_bars}/{len(bars)} bars")
        if out_of_key_bars > 0:
            notes = set()
            for b in bars:
                notes.update(b.out_of_key_notes)
            issues.append(f"out-of-key ({', '.join(sorted(notes))}) in {out_of_key_bars}/{len(bars)} bars")
        if chord_mismatch_bars > 0:
            mismatches = []
            for b in bars:
                if not b.chord_root_matches and b.expected_root >= 0:
                    mismatches.append(
                        f"{_PC_TO_NOTE[b.expected_root]}→{_PC_TO_NOTE[b.chord_root]}"
                    )
            unique_mismatches = list(dict.fromkeys(mismatches))[:4]
            issues.append(
                f"wrong chord in {chord_mismatch_bars}/{len(bars)} bars "
                f"({', '.join(unique_mismatches)})"
            )

        logger.info(
            f"  {status} [{label}] {section_start:.1f}-{section_end:.1f}s: "
            f"{len(bars)} bars — {', '.join(issues) if issues else 'clean'}"
        )

    # Summary
    total_sections = len(results)
    passing_sections = sum(1 for r in results if r.passes)
    logger.info(
        f"QC Summary: {passing_sections}/{total_sections} sections pass "
        f"({total_sections - passing_sections} need retry)"
    )

    return results


def get_failing_sections(results: list[SectionQCResult]) -> list[SectionQCResult]:
    """Get sections that failed QC."""
    return [r for r in results if not r.passes]
