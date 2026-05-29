"""Professional mixing using Spotify's pedalboard library.

Applies DAW-grade effects: EQ, compression, reverb, and limiting
to produce a cohesive final mix from vocal + instrumental stems.
"""

from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
from loguru import logger


def _per_stem_process(
    instrumental_path: Path,
    instrumental_pb: np.ndarray,
    sr: int,
) -> np.ndarray:
    """Separate instrumental into stems and apply per-track EQ.

    Splits the generated instrumental via Demucs, applies targeted EQ
    to each stem (drums get punch, bass gets warmth, other gets clarity),
    then recombines.

    Args:
        instrumental_path: Path to the instrumental file (for Demucs).
        instrumental_pb: Instrumental audio array (channels, samples).
        sr: Sample rate.

    Returns:
        Remixed instrumental array (channels, samples).
    """
    import subprocess
    import sys
    import os
    import tempfile

    from pedalboard import (
        Compressor,
        Gain,
        HighpassFilter,
        LowShelfFilter,
        PeakFilter,
        Pedalboard,
    )

    # Run Demucs on the generated instrumental
    with tempfile.TemporaryDirectory() as tmp_dir:
        env = os.environ.copy()
        env["TORCHAUDIO_BACKEND"] = "soundfile"

        result = subprocess.run(
            [sys.executable, "-m", "demucs", "-n", "htdemucs_ft",
             "--out", tmp_dir, "--mp3", str(instrumental_path)],
            capture_output=True, text=True, timeout=300, env=env,
        )

        if result.returncode != 0:
            logger.warning(f"Per-stem Demucs failed, using full instrumental: {result.stderr[:100]}")
            return instrumental_pb

        # Find stems
        stem_dir = Path(tmp_dir) / "htdemucs_ft" / instrumental_path.stem
        if not stem_dir.exists():
            # Try without stem name
            dirs = list(Path(tmp_dir).glob("htdemucs_ft/*"))
            stem_dir = dirs[0] if dirs else None

        if not stem_dir or not stem_dir.exists():
            logger.warning("Per-stem: couldn't find Demucs output")
            return instrumental_pb

        # Load stems
        stems = {}
        for name in ["drums", "bass", "other"]:
            for ext in [".mp3", ".wav"]:
                p = stem_dir / f"{name}{ext}"
                if p.exists():
                    data, _ = sf.read(str(p))
                    if data.ndim == 1:
                        data = np.stack([data, data], axis=-1)
                    stems[name] = data.T.astype(np.float32)
                    break

        if not stems:
            logger.warning("Per-stem: no stems found")
            return instrumental_pb

        # Per-track EQ chains
        drums_chain = Pedalboard([
            HighpassFilter(cutoff_frequency_hz=60.0),
            PeakFilter(cutoff_frequency_hz=100.0, gain_db=3.0, q=1.0),   # Kick punch
            PeakFilter(cutoff_frequency_hz=3000.0, gain_db=2.0, q=1.5),  # Snare crack
            PeakFilter(cutoff_frequency_hz=8000.0, gain_db=1.5, q=1.0),  # Cymbal presence
            Compressor(threshold_db=-12.0, ratio=4.0, attack_ms=5.0, release_ms=50.0),
        ])

        bass_chain = Pedalboard([
            HighpassFilter(cutoff_frequency_hz=30.0),
            LowShelfFilter(cutoff_frequency_hz=80.0, gain_db=2.0, q=0.7),  # Low-end warmth
            PeakFilter(cutoff_frequency_hz=800.0, gain_db=-2.0, q=1.0),    # Reduce mud
            Compressor(threshold_db=-15.0, ratio=3.0, attack_ms=15.0, release_ms=100.0),
        ])

        other_chain = Pedalboard([
            HighpassFilter(cutoff_frequency_hz=100.0),
            PeakFilter(cutoff_frequency_hz=2500.0, gain_db=-2.0, q=0.8),  # Carve vocal space
            PeakFilter(cutoff_frequency_hz=5000.0, gain_db=1.5, q=1.0),   # Air/clarity
            Compressor(threshold_db=-18.0, ratio=2.0, attack_ms=20.0, release_ms=150.0),
        ])

        # Process each stem
        min_len = min(s.shape[1] for s in stems.values())
        mixed = np.zeros((2, min_len), dtype=np.float32)

        if "drums" in stems:
            processed = drums_chain(stems["drums"][:, :min_len], sr)
            mixed += processed
            logger.info("  Per-stem: drums processed (punch + crack)")

        if "bass" in stems:
            processed = bass_chain(stems["bass"][:, :min_len], sr)
            mixed += processed
            logger.info("  Per-stem: bass processed (warmth - mud)")

        if "other" in stems:
            processed = other_chain(stems["other"][:, :min_len], sr)
            mixed += processed
            logger.info("  Per-stem: other processed (clarity + vocal space)")

        logger.info("Per-stem mixing complete")
        return mixed


def mix_with_effects(
    vocal_path: str | Path,
    instrumental_path: str | Path,
    output_path: str | Path,
    original_mix_path: Optional[str | Path] = None,
    vocal_gain_db: float = 0.0,
    instrumental_gain_db: float = 0.0,
    per_stem_mix: bool = False,
) -> Path:
    """Mix vocals + instrumental with professional effects chain.

    Effects applied:
    - Vocal: High-pass filter (80Hz), presence boost (3kHz), de-ess (6-8kHz),
      light compression, short plate reverb
    - Instrumental: Low-cut at 30Hz, low-shelf boost 120Hz, notch at 3.5kHz,
      light compression
    - Master bus: Bus compression, limiter at -1dB

    When per_stem_mix=True, separates the generated instrumental into
    drums/bass/other and applies per-track EQ before mixing.

    Args:
        vocal_path: Path to extracted vocal stem.
        instrumental_path: Path to generated instrumental.
        output_path: Path for the final mixed output.
        original_mix_path: Path to original mix (for loudness reference).
        vocal_gain_db: Additional vocal gain (dB).
        instrumental_gain_db: Additional instrumental gain (dB).
        per_stem_mix: If True, split instrumental into stems for per-track EQ.

    Returns:
        Path to the saved mix file.
    """
    from pedalboard import (
        Compressor,
        Gain,
        HighpassFilter,
        Limiter,
        LowShelfFilter,
        PeakFilter,
        Pedalboard,
        Reverb,
    )
    from pedalboard.io import AudioFile

    vocal_path = Path(vocal_path)
    instrumental_path = Path(instrumental_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load audio
    vocals, sr_v = sf.read(str(vocal_path))
    instrumental, sr_i = sf.read(str(instrumental_path))

    # Match sample rates
    if sr_v != sr_i:
        import librosa
        if sr_v > sr_i:
            instrumental = librosa.resample(instrumental.T, orig_sr=sr_i, target_sr=sr_v).T
        else:
            vocals = librosa.resample(vocals.T, orig_sr=sr_v, target_sr=sr_i).T
            sr_v = sr_i
    sr = sr_v

    # Match lengths
    min_len = min(len(vocals), len(instrumental))
    vocals = vocals[:min_len].astype(np.float32)
    instrumental = instrumental[:min_len].astype(np.float32)

    # Ensure stereo
    if vocals.ndim == 1:
        vocals = np.stack([vocals, vocals], axis=-1)
    if instrumental.ndim == 1:
        instrumental = np.stack([instrumental, instrumental], axis=-1)

    # Transpose to (channels, samples) for pedalboard
    vocals_pb = vocals.T
    instrumental_pb = instrumental.T

    # Per-stem mixing: separate generated instrumental into tracks for individual EQ
    if per_stem_mix:
        instrumental_pb = _per_stem_process(instrumental_path, instrumental_pb, sr)
        # Re-match lengths after per-stem processing (Demucs may trim slightly)
        min_len_pb = min(vocals_pb.shape[1], instrumental_pb.shape[1])
        vocals_pb = vocals_pb[:, :min_len_pb]
        instrumental_pb = instrumental_pb[:, :min_len_pb]

    # === VOCAL CHAIN ===
    # Light processing — vocals come from a mastered track (already compressed)
    vocal_chain = Pedalboard([
        HighpassFilter(cutoff_frequency_hz=80.0),       # Remove rumble
        PeakFilter(cutoff_frequency_hz=3000.0, gain_db=1.0, q=1.0),  # Subtle presence
        PeakFilter(cutoff_frequency_hz=6500.0, gain_db=-3.0, q=1.5),  # De-ess (stronger)
        PeakFilter(cutoff_frequency_hz=8000.0, gain_db=-2.0, q=2.0),  # Sibilance control
        Compressor(
            threshold_db=-12.0,
            ratio=2.0,
            attack_ms=20.0,
            release_ms=150.0,
        ),
        Reverb(room_size=0.15, wet_level=0.06, dry_level=1.0),  # Short plate (subtle)
        Gain(gain_db=vocal_gain_db - 2.0),  # Pull vocals back (Liam: vocals too loud)
    ])

    # === INSTRUMENTAL CHAIN ===
    instrumental_chain = Pedalboard([
        HighpassFilter(cutoff_frequency_hz=20.0),       # Sub cleanup (lower than before)
        LowShelfFilter(cutoff_frequency_hz=80.0, gain_db=3.5, q=0.7),  # Sub/kick weight
        LowShelfFilter(cutoff_frequency_hz=150.0, gain_db=2.0, q=0.7),  # Low-end warmth
        PeakFilter(cutoff_frequency_hz=3500.0, gain_db=-1.5, q=0.8),  # Gentle vocal space carve
        Compressor(
            threshold_db=-15.0,
            ratio=2.5,
            attack_ms=20.0,
            release_ms=150.0,
        ),
        Gain(gain_db=instrumental_gain_db + 5.0),  # Compensate for quieter AI generation
    ])

    # === MASTER BUS ===
    # Light touch — vocals are already from a mastered track (pre-compressed)
    master_chain = Pedalboard([
        Compressor(
            threshold_db=-8.0,
            ratio=1.5,
            attack_ms=50.0,
            release_ms=300.0,
        ),
        Limiter(threshold_db=-1.0),
    ])

    # Process stems
    logger.info("Processing vocal chain...")
    vocals_processed = vocal_chain(vocals_pb, sr)

    logger.info("Processing instrumental chain...")
    instrumental_processed = instrumental_chain(instrumental_pb, sr)

    # Loudness matching: match vocal-to-instrumental ratio from original
    if original_mix_path:
        original, sr_o = sf.read(str(original_mix_path))
        original = original[:min_len].astype(np.float32)
        if original.ndim == 1:
            original = np.stack([original, original], axis=-1)

        # Measure the vocal and instrumental levels
        vocal_rms = np.sqrt(np.mean(vocals_processed**2)) + 1e-10
        inst_rms = np.sqrt(np.mean(instrumental_processed**2)) + 1e-10

        # In the original, instrumental is typically slightly louder than vocals
        # (separated stems show this). Target: instrumental at same level as vocals
        # or slightly above (+1 to +2 dB).
        # Current ratio:
        current_ratio_db = 20 * np.log10(inst_rms / vocal_rms)
        target_ratio_db = 3.0  # Instrumental clearly louder than vocals

        # How much to boost instrumental
        inst_boost_db = target_ratio_db - current_ratio_db
        inst_boost_db = np.clip(inst_boost_db, -3.0, 9.0)  # Clamp to avoid extremes
        inst_boost_linear = 10 ** (inst_boost_db / 20)
        instrumental_processed = instrumental_processed * inst_boost_linear
        logger.info(
            f"Vocal/Inst balance: current={current_ratio_db:+.1f}dB, "
            f"target={target_ratio_db:+.1f}dB, boost={inst_boost_db:+.1f}dB"
        )

        # Now match overall loudness to original
        original_rms = np.sqrt(np.mean(original**2)) + 1e-8
        raw_mix = vocals_processed + instrumental_processed
        raw_mix_rms = np.sqrt(np.mean(raw_mix**2)) + 1e-8
        loudness_gain = original_rms / raw_mix_rms
        loudness_gain = np.clip(loudness_gain, 0.5, 3.0)
        vocals_processed = vocals_processed * loudness_gain
        instrumental_processed = instrumental_processed * loudness_gain
        logger.info(f"Overall loudness match: {20*np.log10(loudness_gain):+.1f} dB")

    # Mix stems
    logger.info("Mixing stems...")
    mix = vocals_processed + instrumental_processed

    # Master bus processing
    logger.info("Applying master bus (compression + limiter)...")
    mix = master_chain(mix, sr)

    # Transpose back to (samples, channels)
    mix = mix.T

    # Apply 50ms fade-in to eliminate any startup transient/click
    fade_in_samples = int(0.05 * sr)
    if len(mix) > fade_in_samples:
        fade = ((1 - np.cos(np.linspace(0, np.pi, fade_in_samples))) / 2).reshape(-1, 1)
        mix[:fade_in_samples] *= fade

    # Save
    sf.write(str(output_path), mix, sr)
    logger.info(f"Final mix saved: {output_path} ({len(mix)/sr:.1f}s, {sr}Hz)")

    return output_path
