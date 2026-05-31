"""Blend configuration serialization for reproducibility.

Persists hint blending parameters (blend_factor, chord confidence,
MIDI renderer settings) as JSON alongside pipeline output so that
results can be exactly reproduced.
"""

import json
from pathlib import Path
from typing import Optional, Union

from loguru import logger


def write_blend_metadata(
    output_dir: Path,
    output_stem: str,
    blend_factor: Union[float, list[float]],
    chord_confidence: float,
    midi_settings: dict,
) -> Optional[Path]:
    """Write blend configuration to JSON metadata file.

    Args:
        output_dir: Directory to write the metadata file.
        output_stem: Base name for the output file (without extension).
        blend_factor: Scalar or per-frame blend factor used.
        chord_confidence: Confidence score from chord detection.
        midi_settings: Dict with MIDI renderer parameters (sample_rate, velocity, etc).

    Returns:
        Path to written metadata file, or None if write failed.
    """
    output_dir = Path(output_dir)
    metadata_path = output_dir / f"{output_stem}.blend.json"

    metadata = {
        "blend_factor": blend_factor,
        "chord_detection": {
            "confidence": chord_confidence,
        },
        "midi_renderer": midi_settings,
    }

    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Blend metadata written: {metadata_path}")
        return metadata_path
    except Exception as e:
        logger.warning(f"Failed to write blend metadata: {e}")
        return None


def read_blend_metadata(metadata_path: Path) -> Optional[dict]:
    """Read blend metadata for reproduction.

    Args:
        metadata_path: Path to the .blend.json file.

    Returns:
        Dict with blend configuration, or None if read failed.
    """
    metadata_path = Path(metadata_path)

    if not metadata_path.exists():
        logger.warning(f"Blend metadata not found: {metadata_path}")
        return None

    try:
        with open(metadata_path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.warning(f"Failed to read blend metadata: {e}")
        return None
