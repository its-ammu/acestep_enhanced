"""Render chord progressions as piano WAV for timbre-neutral hint extraction.

Converts detected chord labels (from chord_detector) into a MIDI piano
rendering at 48kHz. The resulting WAV provides harmonically equivalent
audio with neutral timbre, suitable for semantic hint extraction.
"""

from pathlib import Path
from typing import Optional

from loguru import logger

from .audio_analysis import ChordSegment


# Chord quality → interval set (semitones from root)
_QUALITY_INTERVALS: dict[str, list[int]] = {
    "maj": [0, 4, 7],
    "min": [0, 3, 7],
    "dim": [0, 3, 6],
    "aug": [0, 4, 8],
    "7": [0, 4, 7, 10],
    "maj7": [0, 4, 7, 11],
    "min7": [0, 3, 7, 10],
    "dim7": [0, 3, 6, 9],
    "hdim7": [0, 3, 6, 10],
    "sus2": [0, 2, 7],
    "sus4": [0, 5, 7],
    "minmaj7": [0, 3, 7, 11],
}

# Note name → MIDI pitch class (0-11)
_NOTE_TO_PC: dict[str, int] = {
    "C": 0, "C#": 1, "Db": 1,
    "D": 2, "D#": 3, "Eb": 3,
    "E": 4, "Fb": 4,
    "F": 5, "F#": 6, "Gb": 6,
    "G": 7, "G#": 8, "Ab": 8,
    "A": 9, "A#": 10, "Bb": 10,
    "B": 11, "Cb": 11,
}

# Base octave for closed-position voicing (C3 = MIDI 48)
_BASE_MIDI_NOTE = 48  # C3
_MAX_MIDI_NOTE = 72   # C5


def render_chords_to_wav(
    chords: list[ChordSegment],
    output_path: str | Path,
    duration: float,
    sample_rate: int = 48000,
    velocity: int = 80,
) -> Optional[Path]:
    """Render chord progression as piano WAV for hint extraction.

    Args:
        chords: List of ChordSegment with start, end, chord label.
        output_path: Where to write the WAV file.
        duration: Target duration in seconds (matches original audio).
        sample_rate: Output sample rate (48kHz for ACE-Step compatibility).
        velocity: MIDI velocity (0-127). Default 80 for uniform dynamics.

    Returns:
        Path to rendered WAV, or None if chord list is empty.
    """
    if not chords:
        logger.warning("MIDI renderer: empty chord progression, skipping")
        return None

    try:
        import pretty_midi
    except ImportError:
        logger.error("MIDI renderer: pretty_midi not installed. Run: pip install pretty_midi")
        return None

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Create MIDI with piano instrument
    midi = pretty_midi.PrettyMIDI(initial_tempo=120.0)
    piano = pretty_midi.Instrument(program=0, name="Piano")  # Program 0 = Acoustic Grand Piano

    rendered_count = 0
    skipped_count = 0

    for chord in chords:
        # Skip segments beyond target duration
        if chord.start >= duration:
            break

        # Clamp end to target duration
        end_time = min(chord.end, duration)

        # Skip "N" (no-chord) segments
        if chord.chord.strip() == "N":
            continue

        # Parse chord label to MIDI notes
        midi_notes = _chord_to_midi_notes(chord.chord)
        if midi_notes is None:
            logger.warning(
                f"MIDI renderer: unrecognized chord '{chord.chord}' "
                f"at {chord.start:.2f}s, skipping"
            )
            skipped_count += 1
            continue

        # Add notes to piano instrument
        for note_number in midi_notes:
            note = pretty_midi.Note(
                velocity=velocity,
                pitch=note_number,
                start=chord.start,
                end=end_time,
            )
            piano.notes.append(note)

        rendered_count += 1

    if rendered_count == 0:
        logger.warning("MIDI renderer: no chords could be rendered")
        return None

    midi.instruments.append(piano)

    if skipped_count > 0:
        logger.info(
            f"MIDI renderer: rendered {rendered_count} chords, "
            f"skipped {skipped_count} unrecognized"
        )

    # Synthesize to audio
    try:
        audio_data = midi.fluidsynth(fs=sample_rate)
    except (ImportError, Exception) as e:
        logger.warning(f"MIDI renderer: FluidSynth unavailable ({e}), using sine-wave synthesis")
        audio_data = midi.synthesize(fs=sample_rate)

    if audio_data is None or len(audio_data) == 0:
        logger.error("MIDI renderer: synthesis failed")
        return None

    # Pad or trim to match target duration
    target_samples = int(duration * sample_rate)
    import numpy as np

    if len(audio_data) < target_samples:
        audio_data = np.pad(audio_data, (0, target_samples - len(audio_data)))
    elif len(audio_data) > target_samples:
        audio_data = audio_data[:target_samples]

    # Normalize
    peak = np.max(np.abs(audio_data))
    if peak > 0:
        audio_data = audio_data / peak * 0.8

    # Write as mono 16-bit PCM WAV
    import soundfile as sf
    sf.write(str(output_path), audio_data, sample_rate, subtype="PCM_16")

    logger.info(
        f"MIDI renderer: wrote {output_path} "
        f"({duration:.1f}s, {rendered_count} chords, {sample_rate}Hz)"
    )
    return output_path


def _chord_to_midi_notes(label: str) -> Optional[list[int]]:
    """Parse a chord label into MIDI note numbers (closed voicing, C3–C5).

    Handles formats:
    - madmom: "C:maj", "F#:min7", "Bb:dim"
    - simple: "C", "Am", "F#m7", "Bb7"

    Args:
        label: Chord label string.

    Returns:
        List of MIDI note numbers, or None if unrecognized.
    """
    label = label.strip()
    if not label or label == "N":
        return None

    root_name: str
    quality: str

    # madmom format: "root:quality"
    if ":" in label:
        parts = label.split(":", 1)
        root_name = parts[0]
        quality = parts[1].lower() if len(parts) > 1 else "maj"
    else:
        # Simple format: parse root + quality suffix
        root_name, quality = _parse_simple_chord(label)

    # Resolve root pitch class
    root_pc = _NOTE_TO_PC.get(root_name)
    if root_pc is None:
        return None

    # Get intervals for quality
    intervals = _QUALITY_INTERVALS.get(quality)
    if intervals is None:
        # Try common aliases
        aliases = {"m": "min", "m7": "min7", "M7": "maj7", "M": "maj"}
        intervals = _QUALITY_INTERVALS.get(aliases.get(quality, ""))
    if intervals is None:
        # Default to major triad for unknown qualities
        intervals = _QUALITY_INTERVALS["maj"]

    # Build MIDI notes in closed position (C3–C5 range)
    midi_notes = []
    for interval in intervals:
        note = _BASE_MIDI_NOTE + root_pc + interval
        # Keep within C3–C5 range
        while note > _MAX_MIDI_NOTE:
            note -= 12
        while note < _BASE_MIDI_NOTE:
            note += 12
        midi_notes.append(note)

    return sorted(set(midi_notes))


def _parse_simple_chord(label: str) -> tuple[str, str]:
    """Parse simple chord notation like 'Am7', 'F#m', 'Bb', 'C7'.

    Returns:
        Tuple of (root_name, quality).
    """
    # Extract root (1 or 2 characters: note + optional accidental)
    i = 1
    if len(label) > 1 and label[1] in "#b":
        i = 2

    root = label[:i]
    suffix = label[i:].lower()

    # Map suffix to quality
    suffix_map = {
        "": "maj",
        "m": "min",
        "m7": "min7",
        "7": "7",
        "maj7": "maj7",
        "dim": "dim",
        "aug": "aug",
        "sus2": "sus2",
        "sus4": "sus4",
        "dim7": "dim7",
    }

    quality = suffix_map.get(suffix, "maj")
    return root, quality
