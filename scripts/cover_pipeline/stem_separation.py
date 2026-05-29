"""Stem separation using Mel-Band RoFormer and Demucs v4.

Mel-Band RoFormer: highest quality vocal isolation (SDR ~11.2)
Demucs v4 (htdemucs_ft): multi-stem separation for analysis (drums, bass, other)
"""

import gc
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from loguru import logger

from .deps import ensure_dependencies, get_model_path

# Prevent torchcodec/ffmpeg shared lib issues in Demucs
os.environ.setdefault("TORCHAUDIO_BACKEND", "soundfile")


def _free_gpu():
    """Force GPU memory release after separation."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


@dataclass
class SeparationResult:
    """Result of stem separation."""

    vocals: Path
    instrumental: Path
    drums: Optional[Path] = None
    bass: Optional[Path] = None
    other: Optional[Path] = None


def _run_melband_roformer(
    input_audio: Path,
    output_dir: Path,
) -> tuple[Path, Path]:
    """Run Mel-Band RoFormer for vocal/instrumental separation.

    Uses audio-separator package with its default vocal model.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        from audio_separator.separator import Separator

        separator = Separator(
            output_dir=str(output_dir),
            output_format="wav",
        )
        # load_model() with no args uses the default vocal separation model
        separator.load_model()
        output_files = separator.separate(str(input_audio))

        vocals_path = None
        instrumental_path = None
        for f in output_files:
            f_path = Path(f)
            if not f_path.is_absolute():
                f_path = output_dir / f_path.name
            name_lower = f_path.stem.lower()
            if "vocal" in name_lower or "voice" in name_lower:
                vocals_path = f_path
            elif "instrument" in name_lower or "no_vocal" in name_lower or "accompaniment" in name_lower:
                instrumental_path = f_path

        # If naming didn't match, assume first is instrumental, second is vocals
        # (audio-separator convention: [Instrumental, Vocals])
        if not vocals_path and not instrumental_path and len(output_files) == 2:
            instrumental_path = output_dir / Path(output_files[0]).name
            vocals_path = output_dir / Path(output_files[1]).name

        if vocals_path and instrumental_path:
            return vocals_path, instrumental_path

        # If we got files but couldn't identify them
        if output_files:
            logger.warning(f"Could not identify vocal/instrumental from: {output_files}")
            # Return whatever we got
            paths = [Path(f) for f in output_files]
            if len(paths) >= 2:
                return paths[1], paths[0]  # Assume [instrumental, vocals] order

    except ImportError:
        logger.error("audio-separator not installed: pip install 'audio-separator[gpu]'")
    except Exception as e:
        logger.error(f"audio-separator failed: {e}")

    raise RuntimeError(
        "Mel-Band RoFormer separation failed. "
        "Install audio-separator: pip install 'audio-separator[gpu]'"
    )


def _run_demucs(
    input_audio: Path,
    output_dir: Path,
) -> dict[str, Path]:
    """Run Demucs v4 (htdemucs_ft) for 4-stem separation."""
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        import os

        # Force soundfile backend to avoid torchcodec/ffmpeg shared lib issues
        env = os.environ.copy()
        env["TORCHAUDIO_BACKEND"] = "soundfile"

        result = subprocess.run(
            [
                sys.executable, "-m", "demucs",
                "-n", "htdemucs_ft",
                "--out", str(output_dir),
                "--mp3",  # Use mp3 output to avoid torchcodec WAV save issues
                str(input_audio),
            ],
            capture_output=True,
            text=True,
            timeout=600,
            env=env,
        )

        if result.returncode != 0:
            logger.error(f"Demucs failed: {result.stderr}")
            raise RuntimeError(f"Demucs separation failed: {result.stderr}")

        # Demucs outputs to: output_dir/htdemucs_ft/{stem}/{vocals,drums,bass,other}.mp3
        stem = input_audio.stem
        demucs_out = output_dir / "htdemucs_ft" / stem

        stems = {}
        for stem_name in ["vocals", "drums", "bass", "other"]:
            # Check both mp3 and wav
            for ext in [".mp3", ".wav"]:
                stem_path = demucs_out / f"{stem_name}{ext}"
                if stem_path.exists():
                    stems[stem_name] = stem_path
                    break

        return stems

    except FileNotFoundError:
        raise RuntimeError("Demucs not found. Install: pip install demucs")


def separate_stems(
    input_audio: str | Path,
    output_dir: Optional[str | Path] = None,
) -> SeparationResult:
    """Run full stem separation pipeline.

    Uses Mel-Band RoFormer for primary vocal/instrumental split,
    and Demucs v4 for multi-stem analysis.

    Args:
        input_audio: Path to the input audio file.
        output_dir: Output directory. Defaults to input_audio parent / "separated".

    Returns:
        SeparationResult with paths to all separated stems.
    """
    input_audio = Path(input_audio)
    if not input_audio.exists():
        raise FileNotFoundError(f"Input audio not found: {input_audio}")

    if output_dir is None:
        output_dir = input_audio.parent / "separated"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    melband_dir = output_dir / "melband"
    demucs_dir = output_dir / "demucs"

    # Step 1: Mel-Band RoFormer (primary vocal isolation)
    logger.info("Running Mel-Band RoFormer for vocal separation...")
    vocals_path, instrumental_path = _run_melband_roformer(input_audio, melband_dir)
    logger.info(f"Vocals: {vocals_path}")
    logger.info(f"Instrumental: {instrumental_path}")

    # Step 2: Demucs v4 (multi-stem for analysis)
    logger.info("Running Demucs v4 for multi-stem analysis...")
    try:
        demucs_stems = _run_demucs(input_audio, demucs_dir)
        drums_path = demucs_stems.get("drums")
        bass_path = demucs_stems.get("bass")
        other_path = demucs_stems.get("other")
        logger.info(f"Demucs stems: {list(demucs_stems.keys())}")
    except RuntimeError as e:
        logger.warning(f"Demucs failed (non-critical): {e}")
        drums_path = None
        bass_path = None
        other_path = None

    result = SeparationResult(
        vocals=vocals_path,
        instrumental=instrumental_path,
        drums=drums_path,
        bass=bass_path,
        other=other_path,
    )

    # Free GPU after separation models are done
    _free_gpu()
    logger.info("Separation complete, GPU memory freed")

    return result
