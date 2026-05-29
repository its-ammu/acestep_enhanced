"""ACE-Step generation orchestrator for the cover pipeline.

Handles:
1. LM pre-processing (4B planner) — caption refinement + audio codes
2. xl-sft Cover mode generation — high-quality instrumental
"""

import gc
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from loguru import logger


def _free_gpu():
    """Force GPU memory release between pipeline stages."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        # Reset peak memory stats
        torch.cuda.reset_peak_memory_stats()
        allocated = torch.cuda.memory_allocated() / 1e9
        if allocated > 0.1:
            # Force release by resetting CUDA context
            torch.cuda.ipc_collect()
        logger.info(f"GPU memory freed. Allocated: {torch.cuda.memory_allocated() / 1e9:.2f}GB")


@dataclass
class GenerationConfig:
    """Configuration for ACE-Step generation."""

    # Model selection
    dit_model: str = "acestep-v15-xl-sft"
    lm_model: str = "acestep-5Hz-lm-4B"

    # Cover mode parameters
    audio_cover_strength: float = 0.95
    cover_noise_strength: float = 0.9
    guidance_scale: float = 4.0
    inference_steps: int = 50
    shift: float = 6.0
    batch_size: int = 1

    # Metadata (from analysis)
    bpm: Optional[int] = None
    keyscale: Optional[str] = None
    timesignature: str = "4/4"
    duration: Optional[int] = None

    # Paths
    checkpoints_dir: Optional[str] = None


@dataclass
class GenerationResult:
    """Result of the generation pipeline."""

    instrumental_paths: list[Path]
    enhanced_caption: str = ""
    audio_codes: str = ""
    lm_metadata: dict = None

    def __post_init__(self):
        if self.lm_metadata is None:
            self.lm_metadata = {}


def run_lm_preprocess(
    caption: str,
    lyrics: str,
    config: GenerationConfig,
) -> dict:
    """Run the 5Hz LM 4B as a pre-processor to refine caption and generate codes.

    This runs text2music with thinking=True to get:
    - Enhanced caption (CoT-rewritten in DiT's native vocabulary)
    - Audio codes (semantic blueprint for the target style)
    - Validated metadata

    Args:
        caption: Raw caption from the caption builder.
        lyrics: Structural lyrics.
        config: Generation configuration.

    Returns:
        Dict with keys: enhanced_caption, audio_codes, metadata.
    """
    logger.info("Running LM pre-processing (4B planner)...")
    logger.info(f"Input caption: {caption[:100]}...")

    result = {
        "enhanced_caption": caption,  # Fallback to original
        "audio_codes": "",
        "metadata": {},
    }

    try:
        # Import ACE-Step components
        from acestep.llm_inference import LLMHandler

        # Initialize LLM handler
        llm_handler = LLMHandler()

        # Find checkpoint path
        checkpoints_dir = config.checkpoints_dir
        if not checkpoints_dir:
            # Try common locations
            for candidate in [
                Path("checkpoints"),
                Path.home() / ".cache" / "acestep" / "checkpoints",
            ]:
                if candidate.exists():
                    checkpoints_dir = str(candidate)
                    break

        if not checkpoints_dir:
            logger.warning("Checkpoints directory not found, skipping LM pre-processing")
            return result

        # Initialize LLM
        checkpoints_dir_path = Path(checkpoints_dir)
        lm_path = checkpoints_dir_path / config.lm_model
        if not lm_path.exists():
            # Try downloading
            logger.info(f"LM model not found at {lm_path}, attempting download...")
            try:
                from acestep.model_downloader import ensure_dit_model

                ensure_dit_model(config.lm_model, checkpoints_dir)
            except Exception as e:
                logger.warning(f"Could not download LM model: {e}")
                return result

        # Initialize LLM with the correct API
        status_msg, success = llm_handler.initialize(
            checkpoint_dir=checkpoints_dir,
            lm_model_path=config.lm_model,
            backend="pt",
            device="auto",
        )

        if not success or not llm_handler.llm_initialized:
            logger.warning("LLM failed to initialize, skipping pre-processing")
            return result

        # Build user metadata — intentionally omit one field so has_all_metas=False
        # and the LM runs full CoT (which includes caption rewriting).
        # We pass bpm/keyscale/timesignature but NOT duration, forcing CoT to run.
        user_metadata = {}
        if config.bpm:
            user_metadata["bpm"] = str(config.bpm)
        if config.keyscale:
            user_metadata["keyscale"] = config.keyscale
        if config.timesignature:
            user_metadata["timesignature"] = config.timesignature
        # Deliberately NOT passing duration here — this forces has_all_metas=False
        # so the LM runs Phase 1 CoT which rewrites the caption.

        # Run generation (Phase 1: CoT + Phase 2: Codes)
        lm_result = llm_handler.generate_with_stop_condition(
            caption=caption,
            lyrics=lyrics,
            infer_type="llm_dit",
            temperature=0.85,
            cfg_scale=2.0,
            negative_prompt="NO USER INPUT",
            top_k=None,
            top_p=0.9,
            target_duration=float(config.duration) if config.duration else None,
            user_metadata=user_metadata if user_metadata else None,
            use_cot_caption=True,
            use_cot_language=True,
            use_cot_metas=True,
            use_constrained_decoding=True,
            constrained_decoding_debug=False,
            batch_size=1,
        )

        if lm_result.get("success"):
            metadata = lm_result.get("metadata", {})
            audio_codes = lm_result.get("audio_codes", "")

            # Extract enhanced caption
            enhanced_caption = metadata.get("caption", caption)
            if enhanced_caption and len(enhanced_caption) > len(caption) * 0.5:
                result["enhanced_caption"] = enhanced_caption
                logger.info(f"Enhanced caption: {enhanced_caption[:100]}...")

            # Extract audio codes
            if isinstance(audio_codes, list):
                audio_codes = audio_codes[0] if audio_codes else ""
            if audio_codes:
                result["audio_codes"] = audio_codes
                codes_count = audio_codes.count("<|audio_code_")
                logger.info(f"Generated {codes_count} audio codes")

            result["metadata"] = metadata
        else:
            logger.warning(f"LM pre-processing failed: {lm_result.get('error', 'unknown')}")

    except ImportError as e:
        logger.warning(f"ACE-Step imports not available for LM pre-processing: {e}")
    except Exception as e:
        logger.error(f"LM pre-processing error: {e}")
    finally:
        # Free GPU memory before DiT loads
        if "llm_handler" in dir():
            del llm_handler
        _free_gpu()
        logger.info("LM unloaded, GPU memory freed for DiT")

    return result


def _write_cover_config(
    src_audio: Path,
    reference_audio: Path,
    caption: str,
    lyrics: str,
    config: GenerationConfig,
    audio_codes: str,
    output_dir: Path,
) -> Path:
    """Write a TOML config file for cli.py cover generation."""
    import toml

    cover_config = {
        "task_type": "cover",
        "project_root": str(Path(__file__).parent.parent.parent),
        "checkpoint_dir": str(Path(__file__).parent.parent.parent / "checkpoints"),
        "src_audio": str(src_audio),
        "reference_audio": str(reference_audio),
        "caption": caption,
        "lyrics": lyrics,
        "config_path": config.dit_model,
        "audio_cover_strength": config.audio_cover_strength,
        "cover_noise_strength": config.cover_noise_strength,
        "guidance_scale": config.guidance_scale,
        "inference_steps": config.inference_steps,
        "batch_size": config.batch_size,
        "instrumental": True,
        "thinking": False,
        "use_cot_caption": False,
        "use_cot_metas": False,
        "use_cot_language": False,
        "use_cot_lyrics": False,
        "shift": config.shift,
        "cfg_interval_start": 0.0,
        "cfg_interval_end": 1.0,
        "infer_method": "ode",
        "use_random_seed": True,
        "seed": -1,
        "audio_format": "flac",
        "save_dir": str(output_dir),
    }

    if audio_codes:
        cover_config["audio_codes"] = audio_codes
    if config.bpm:
        cover_config["bpm"] = config.bpm
    if config.keyscale:
        cover_config["keyscale"] = config.keyscale
    if config.timesignature:
        cover_config["timesignature"] = config.timesignature

    # Clean lyrics — remove problematic characters that break TOML or confuse DiT
    if "lyrics" in cover_config and cover_config["lyrics"]:
        clean_lyrics = cover_config["lyrics"]
        # Remove content in parentheses (Qwen rambling)
        import re
        clean_lyrics = re.sub(r"\([^)]*\)", "", clean_lyrics)
        # Remove single quotes
        clean_lyrics = clean_lyrics.replace("'", "")
        # Remove double dashes
        clean_lyrics = clean_lyrics.replace("--", "")
        # Collapse multiple spaces/newlines
        clean_lyrics = re.sub(r"\n{3,}", "\n\n", clean_lyrics)
        clean_lyrics = re.sub(r"  +", " ", clean_lyrics)
        cover_config["lyrics"] = clean_lyrics.strip()

    config_path = output_dir / "cover_config.toml"
    config_path.write_text(toml.dumps(cover_config))
    logger.info(f"Cover config written: {config_path}")
    return config_path


def _write_lm_config(
    caption: str,
    lyrics: str,
    config: GenerationConfig,
    output_dir: Path,
) -> Path:
    """Write a TOML config file for cli.py LM pre-processing (text2music with thinking)."""
    import toml

    lm_config = {
        "task_type": "text2music",
        "caption": caption,
        "lyrics": lyrics,
        "config_path": config.dit_model,
        "lm_model": config.lm_model,
        "thinking": True,
        "use_cot_caption": True,
        "use_cot_metas": True,
        "batch_size": 1,
    }

    if config.bpm:
        lm_config["bpm"] = config.bpm
    if config.keyscale:
        lm_config["keyscale"] = config.keyscale
    if config.timesignature:
        lm_config["timesignature"] = config.timesignature
    if config.duration:
        lm_config["duration"] = config.duration

    config_path = output_dir / "lm_preprocess_config.toml"
    config_path.write_text(toml.dumps(lm_config))
    logger.info(f"LM config written: {config_path}")
    return config_path


def _refine_caption_subprocess(
    caption: str,
    lyrics: str,
    config: GenerationConfig,
    output_dir: Path,
) -> str:
    """Run LM caption refinement as a separate subprocess to fully release GPU.

    Writes a small Python script that loads the LM, refines the caption,
    writes the result to a file, and exits (freeing all GPU memory).
    """
    import json
    import tempfile

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Write inputs to temp file
    inputs_path = output_dir / "_lm_refine_input.json"
    result_path = output_dir / "_lm_refine_result.json"
    inputs_path.write_text(json.dumps({
        "caption": caption,
        "lyrics": lyrics,
        "lm_model": config.lm_model,
        "bpm": config.bpm,
        "keyscale": config.keyscale,
        "timesignature": config.timesignature,
        "duration": config.duration,
    }))

    # Build subprocess script
    script = f"""
import json, sys
sys.path.insert(0, '.')
from scripts.cover_pipeline.generate import run_lm_preprocess, GenerationConfig

inputs = json.loads(open('{inputs_path}').read())
config = GenerationConfig(
    lm_model=inputs['lm_model'],
    bpm=inputs.get('bpm'),
    keyscale=inputs.get('keyscale'),
    timesignature=inputs.get('timesignature', '4/4'),
    duration=inputs.get('duration'),
    dit_model='acestep-v15-xl-sft',
)
result = run_lm_preprocess(inputs['caption'], inputs['lyrics'], config)
json.dump({{'enhanced_caption': result.get('enhanced_caption', '')}}, open('{result_path}', 'w'))
"""

    project_root = str(Path(__file__).parent.parent.parent)
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=300,
        cwd=project_root,
    )

    if result.returncode != 0:
        logger.warning(f"LM refinement subprocess failed: {result.stderr[:200]}")
        return ""

    # Read result
    try:
        result_data = json.loads(result_path.read_text())
        return result_data.get("enhanced_caption", "")
    except Exception as e:
        logger.warning(f"Failed to read LM result: {e}")
        return ""
    finally:
        inputs_path.unlink(missing_ok=True)
        result_path.unlink(missing_ok=True)


def run_cover_generation(
    src_audio: str | Path,
    reference_audio: str | Path,
    caption: str,
    lyrics: str,
    config: GenerationConfig,
    audio_codes: str = "",
    output_dir: Optional[str | Path] = None,
    refine_caption: bool = False,
) -> list[Path]:
    """Run xl-sft Cover mode generation.

    Args:
        src_audio: Path to original song (provides structural codes).
        reference_audio: Path to instrumental stem (timbre/mixing anchor).
        caption: Caption from structure timeline.
        lyrics: Structural lyrics.
        config: Generation configuration.
        audio_codes: Audio codes (empty = let DiT extract from src_audio).
        output_dir: Output directory for generated audio.
        refine_caption: If True, run LM 4B to reformat caption before generation.

    Returns:
        List of paths to generated instrumental audio files.
    """
    src_audio = Path(src_audio)
    reference_audio = Path(reference_audio)

    if output_dir is None:
        output_dir = src_audio.parent / "generated"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Optional: LM caption refinement (runs as subprocess to fully release GPU)
    if refine_caption and caption:
        logger.info("Refining caption with LM 4B (subprocess)...")
        refined_caption = _refine_caption_subprocess(caption, lyrics, config, output_dir)
        if refined_caption and len(refined_caption) > 20 and refined_caption != caption:
            logger.info(f"LM refined caption: {refined_caption[:150]}...")
            caption = refined_caption
        else:
            logger.info("LM refinement didn't improve caption, using original")

    logger.info("Running xl-sft Cover generation...")
    logger.info(f"  src_audio: {src_audio}")
    logger.info(f"  reference_audio: {reference_audio}")
    logger.info(f"  audio_cover_strength: {config.audio_cover_strength}")
    logger.info(f"  guidance_scale: {config.guidance_scale}")
    logger.info(f"  batch_size: {config.batch_size}")

    # Write TOML config for cli.py
    config_path = _write_cover_config(
        src_audio=src_audio,
        reference_audio=reference_audio,
        caption=caption,
        lyrics=lyrics,
        config=config,
        audio_codes=audio_codes,
        output_dir=output_dir,
    )

    # Build CLI command using config file
    cmd = [
        sys.executable, "cli.py",
        "-c", str(config_path),
    ]

    logger.info(f"Running: {' '.join(cmd[:10])}...")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=1800,  # 30 min timeout for generation
            cwd=str(Path(__file__).parent.parent.parent),  # Project root
        )

        if result.returncode != 0:
            logger.error(f"Generation failed: {result.stderr[:500]}")
            return _run_generation_direct(
                src_audio, reference_audio, caption, lyrics,
                config, audio_codes, output_dir,
            )

        # Parse stdout for output path (cli.py prints: "Path: output/xxx.flac")
        generated_paths = []
        for line in result.stdout.split("\n"):
            if "Path:" in line and (".flac" in line or ".wav" in line or ".mp3" in line):
                # Extract path from line like "[1] Path: output/abc.flac | Seed: 123"
                path_part = line.split("Path:")[1].split("|")[0].strip()
                p = Path(__file__).parent.parent.parent / path_part
                if p.exists():
                    generated_paths.append(p)

        if generated_paths:
            logger.info(f"Generated {len(generated_paths)} audio files: {generated_paths}")
            return generated_paths

        # Fallback: search output directories
        project_root = Path(__file__).parent.parent.parent
        search_dirs = [output_dir, project_root / "output"]
        import time
        for search_dir in search_dirs:
            if search_dir.exists():
                recent = sorted(
                    list(search_dir.glob("*.flac")) + list(search_dir.glob("*.wav")),
                    key=lambda f: f.stat().st_mtime,
                    reverse=True,
                )
                # Only files created in the last 10 minutes
                recent = [f for f in recent if f.stat().st_mtime > time.time() - 600]
                if recent:
                    logger.info(f"Found {len(recent)} recent files in {search_dir}")
                    return recent

        logger.warning("No output files found after generation")
        logger.debug(f"cli.py stdout: {result.stdout[-500:]}")
        return []

    except subprocess.TimeoutExpired:
        logger.error("Generation timed out (30 min)")
        return []
    except Exception as e:
        logger.error(f"Generation error: {e}")
        return _run_generation_direct(
            src_audio, reference_audio, caption, lyrics,
            config, audio_codes, output_dir,
        )


def _run_generation_direct(
    src_audio: Path,
    reference_audio: Path,
    caption: str,
    lyrics: str,
    config: GenerationConfig,
    audio_codes: str,
    output_dir: Path,
) -> list[Path]:
    """Direct Python invocation of ACE-Step generation (fallback)."""
    try:
        from acestep.inference import GenerationParams, generate_with_progress

        params = GenerationParams(
            task_type="cover",
            caption=caption,
            lyrics=lyrics,
            src_audio=str(src_audio),
            reference_audio=str(reference_audio),
            audio_codes=audio_codes,
            audio_cover_strength=config.audio_cover_strength,
            guidance_scale=config.guidance_scale,
            inference_steps=config.inference_steps,
            bpm=config.bpm,
            keyscale=config.keyscale or "",
            timesignature=config.timesignature,
            thinking=False,
        )

        # This would need the full handler setup — simplified here
        logger.warning("Direct generation fallback not fully implemented. Use CLI mode.")
        return []

    except ImportError as e:
        logger.error(f"Cannot import ACE-Step for direct generation: {e}")
        return []
