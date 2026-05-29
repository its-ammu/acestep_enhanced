"""Final mix: combine original vocals with generated instrumental.

Handles loudness matching, mid-side EQ for vocal space, and export.
"""

from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
from loguru import logger


def mix_vocal_and_instrumental(
    vocal_path: str | Path,
    instrumental_path: str | Path,
    output_path: str | Path,
    original_mix_path: Optional[str | Path] = None,
    vocal_gain_db: float = 0.0,
    instrumental_gain_db: float = 0.0,
    auto_loudness_match: bool = True,
    carve_vocal_space: bool = True,
) -> Path:
    """Mix original vocals with generated instrumental.

    Args:
        vocal_path: Path to extracted vocal stem.
        instrumental_path: Path to generated instrumental.
        output_path: Path for the final mixed output.
        original_mix_path: Path to original full mix (used as loudness reference).
        vocal_gain_db: Additional vocal gain adjustment (dB).
        instrumental_gain_db: Additional instrumental gain adjustment (dB).
        auto_loudness_match: Match mix loudness to original.
        carve_vocal_space: Reduce instrumental center energy for vocal clarity.

    Returns:
        Path to the saved mix file.
    """
    vocal_path = Path(vocal_path)
    instrumental_path = Path(instrumental_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load audio
    vocals, sr_v = sf.read(str(vocal_path))
    instrumental, sr_i = sf.read(str(instrumental_path))

    # Resample if needed
    if sr_v != sr_i:
        import librosa
        if sr_v > sr_i:
            instrumental = librosa.resample(instrumental.T, orig_sr=sr_i, target_sr=sr_v).T
        else:
            vocals = librosa.resample(vocals.T, orig_sr=sr_v, target_sr=sr_i).T
            sr_v = sr_i

    sr = sr_v

    # Match lengths (trim to shorter)
    min_len = min(len(vocals), len(instrumental))
    vocals = vocals[:min_len]
    instrumental = instrumental[:min_len]

    # Ensure stereo
    if vocals.ndim == 1:
        vocals = np.stack([vocals, vocals], axis=-1)
    if instrumental.ndim == 1:
        instrumental = np.stack([instrumental, instrumental], axis=-1)

    # Auto loudness matching
    if auto_loudness_match:
        if original_mix_path:
            # Use original mix as the loudness reference
            original, sr_o = sf.read(str(original_mix_path))
            original = original[:min_len]
            if original.ndim == 1:
                original = np.stack([original, original], axis=-1)

            # Measure original mix loudness
            original_rms = np.sqrt(np.mean(original**2)) + 1e-8

            # Measure vocal-to-mix ratio in original
            vocal_rms = np.sqrt(np.mean(vocals**2)) + 1e-8
            original_inst_rms = original_rms  # Approximate: full mix ≈ vocal + instrumental

            # Scale our instrumental so that vocals + instrumental matches original loudness
            raw_mix = vocals + instrumental
            raw_mix_rms = np.sqrt(np.mean(raw_mix**2)) + 1e-8
            loudness_gain = original_rms / raw_mix_rms

            # Apply gain to instrumental (keep vocals untouched)
            instrumental = instrumental * loudness_gain
            # Re-check: if instrumental is now too quiet relative to vocals, boost it
            inst_rms_after = np.sqrt(np.mean(instrumental**2)) + 1e-8
            if inst_rms_after < vocal_rms * 0.5:
                boost = (vocal_rms * 0.8) / inst_rms_after
                instrumental = instrumental * boost

            logger.info(
                f"Auto-gain (original reference): original_rms={original_rms:.4f}, "
                f"loudness_gain={20*np.log10(loudness_gain):+.1f} dB"
            )
        else:
            # Fallback: match instrumental to vocal level
            vocal_rms = np.sqrt(np.mean(vocals**2)) + 1e-8
            inst_rms = np.sqrt(np.mean(instrumental**2)) + 1e-8

            # Target: instrumental at ~90% of vocal loudness (-1 dB below)
            target_ratio = 0.9
            auto_gain = (vocal_rms * target_ratio) / inst_rms
            auto_gain_db = 20 * np.log10(auto_gain)

            # Clamp to ±12 dB
            auto_gain_db = max(-12.0, min(12.0, auto_gain_db))
            auto_gain = 10 ** (auto_gain_db / 20)

            instrumental = instrumental * auto_gain
            logger.info(
                f"Auto-gain (no reference): vocal_rms={vocal_rms:.4f}, "
                f"inst_rms={inst_rms:.4f}, adjustment={auto_gain_db:+.1f} dB"
            )

    # Carve vocal space (reduce instrumental center by 1.5dB — subtle)
    if carve_vocal_space and instrumental.shape[1] == 2:
        inst_mid = (instrumental[:, 0] + instrumental[:, 1]) / 2
        inst_side = (instrumental[:, 0] - instrumental[:, 1]) / 2
        # Reduce mid by 1.5dB (0.84) — subtle, preserves instrument body
        inst_mid = inst_mid * 0.84
        instrumental[:, 0] = inst_mid + inst_side
        instrumental[:, 1] = inst_mid - inst_side
        logger.info("Applied mid-side EQ: -1.5dB center reduction for vocal space")

    # Apply manual gain adjustments
    if vocal_gain_db != 0.0:
        vocals = vocals * (10 ** (vocal_gain_db / 20))
    if instrumental_gain_db != 0.0:
        instrumental = instrumental * (10 ** (instrumental_gain_db / 20))

    # Mix
    mix = vocals + instrumental

    # Normalize to -1dB peak
    peak = np.max(np.abs(mix))
    if peak > 0:
        mix = mix / peak * 0.891  # -1dB = 10^(-1/20) ≈ 0.891

    # Save
    sf.write(str(output_path), mix, sr)
    logger.info(f"Final mix saved: {output_path} ({len(mix)/sr:.1f}s, {sr}Hz)")

    return output_path
