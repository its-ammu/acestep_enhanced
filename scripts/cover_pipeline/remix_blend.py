"""Remix blend generation: LM creativity + bass chord accuracy.

The core remix approach:
1. 4B LM generates creative 5Hz codes (dynamic arrangement, genre-specific)
2. Bass stem provides chord-accurate 5Hz codes
3. Blend them at configurable alpha (creativity vs accuracy)
4. Render via xl-turbo cover mode (fast, 8 steps)
5. QC + hint repaint for failing sections
6. Effects + mix with vocals

This module handles steps 1-4. QC/repaint/effects/mix are orchestrated
by the pipeline runner.
"""

import gc
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
import torch
from loguru import logger


def generate_lm_codes(
    caption: str,
    lyrics: str,
    duration: float,
    bpm: int,
    keyscale: str,
    lm_model: str = "acestep-5Hz-lm-4B",
    lm_temperature: float = 0.7,
) -> str:
    """Generate creative 5Hz codes using the LM planner.

    Loads the LM, generates codes via CoT, unloads LM to free VRAM.

    Args:
        caption: Genre-specific instrument/performance description.
        lyrics: Temporal script with energy tags.
        duration: Target duration in seconds.
        bpm: BPM constraint.
        keyscale: Key constraint.
        lm_model: LM checkpoint name.
        lm_temperature: Sampling temperature (0.7=conservative, 0.85=creative).

    Returns:
        Audio codes string, or empty string if failed.
    """
    from acestep.llm_inference import LLMHandler

    logger.info(f"Loading {lm_model} for creative code generation...")
    llm = LLMHandler()
    llm.initialize(
        checkpoint_dir="checkpoints",
        lm_model_path=lm_model,
        backend="vllm",
        device="cuda",
    )

    logger.info("Generating creative codes (CoT planning)...")
    result = llm.generate_with_stop_condition(
        caption=caption,
        lyrics=lyrics,
        infer_type="llm_dit",
        temperature=lm_temperature,
        cfg_scale=2.0,
        negative_prompt="NO USER INPUT",
        top_p=0.9,
        target_duration=duration,
        user_metadata={"bpm": bpm, "keyscale": keyscale, "duration": duration},
        use_cot_caption=True,
        use_cot_metas=False,  # Don't let LM override our detected metadata
        use_cot_language=True,
        use_constrained_decoding=True,
        batch_size=1,
    )

    # Extract codes from result
    codes = ""
    if isinstance(result.get("audio_codes"), list):
        codes = result["audio_codes"][0] if result["audio_codes"] else ""
    else:
        codes = result.get("audio_codes", "")
    if not codes:
        if isinstance(result.get("audio_code_strings"), list):
            codes = result["audio_code_strings"][0] if result["audio_code_strings"] else ""
        else:
            codes = result.get("audio_code_strings", "")

    n_codes = codes.count("audio_code_")
    lm_caption = str(result.get("caption", ""))[:150]
    logger.info(f"LM generated {n_codes} codes")
    logger.info(f"LM plan: {lm_caption}")

    # Unload LM
    del llm
    gc.collect()
    torch.cuda.empty_cache()
    logger.info("LM unloaded, VRAM freed")

    return codes


def blend_codes_with_bass(
    handler,
    lm_codes: str,
    bass_path: str,
    alpha: float = 0.4,
) -> torch.Tensor:
    """Blend LM codes with bass stem hints for chord accuracy.

    Args:
        handler: Initialized AceStepHandler (DiT loaded).
        lm_codes: Audio codes string from LM.
        bass_path: Path to bass stem for chord hints.
        alpha: Blend factor (0=all bass, 1=all LM). Default 0.4.

    Returns:
        Blended 25Hz hints tensor [B, T, D].
    """
    from .semantic_cover import extract_semantic_hints

    # Extract bass hints (chord accuracy)
    logger.info(f"Extracting bass hints from {Path(bass_path).name}...")
    bass_hints = extract_semantic_hints(handler, bass_path)

    # Decode LM codes to 25Hz
    logger.info("Decoding LM codes to 25Hz embeddings...")
    lm_hints = handler._decode_audio_codes_to_latents(lm_codes)

    if lm_hints is None:
        logger.warning("LM code decoding failed, using bass hints only")
        return bass_hints

    logger.info(f"Bass: {bass_hints.shape}, LM: {lm_hints.shape}")

    # Pad shorter to match longer
    max_t = max(bass_hints.shape[1], lm_hints.shape[1])
    if bass_hints.shape[1] < max_t:
        pad = torch.zeros(
            1, max_t - bass_hints.shape[1], 64,
            device=bass_hints.device, dtype=bass_hints.dtype,
        )
        bass_hints = torch.cat([bass_hints, pad], dim=1)
    if lm_hints.shape[1] < max_t:
        pad = torch.zeros(
            1, max_t - lm_hints.shape[1], 64,
            device=lm_hints.device, dtype=lm_hints.dtype,
        )
        lm_hints = torch.cat([lm_hints, pad], dim=1)

    # Blend
    blended = alpha * lm_hints + (1.0 - alpha) * bass_hints
    logger.info(f"Blended hints: {blended.shape} (alpha={alpha})")

    return blended


def render_with_blended_hints(
    handler,
    blended_hints: torch.Tensor,
    target_wavs: torch.Tensor,
    caption: str,
    lyrics: str,
    metas: list,
    cover_noise_strength: float = 0.15,
    inference_steps: int = 8,
    shift: float = 3.0,
) -> Optional[np.ndarray]:
    """Render audio using blended hints in cover mode.

    Args:
        handler: Initialized AceStepHandler.
        blended_hints: Blended 25Hz hints [B, T, D].
        target_wavs: Source audio tensor [B, 2, samples].
        caption: Caption for generation.
        lyrics: Temporal script.
        metas: Metadata dicts.
        cover_noise_strength: How close to start from source.
        inference_steps: Diffusion steps (8 for turbo).
        shift: Timestep shift (3.0 for turbo).

    Returns:
        Audio array (samples, 2) or None if failed.
    """
    # Monkey-patch blended hints
    original_prepare = handler.model.prepare_condition

    def patched_prepare(*args, **kwargs):
        kwargs["precomputed_lm_hints_25Hz"] = blended_hints.to(
            device="cuda:0", dtype=torch.bfloat16
        )
        return original_prepare(*args, **kwargs)

    handler.model.prepare_condition = patched_prepare

    try:
        logger.info(
            f"Rendering (cns={cover_noise_strength}, "
            f"steps={inference_steps}, shift={shift})..."
        )
        result = handler.service_generate(
            captions=caption,
            lyrics=lyrics,
            target_wavs=target_wavs,
            metas=metas,
            audio_cover_strength=0.95,
            guidance_scale=1.0,
            infer_steps=inference_steps,
            shift=shift,
            cover_noise_strength=cover_noise_strength,
            task_type="cover",
            infer_method="ode",
        )

        if "target_latents" not in result:
            logger.error("Render failed — no latents returned")
            return None

        latents = result["target_latents"]
        if latents.shape[-1] == 64:
            latents = latents.movedim(-1, -2)
        latents = latents.to(dtype=torch.bfloat16)

        with torch.no_grad():
            audio_tensor = handler.tiled_decode(latents)

        audio_np = audio_tensor.float().cpu().numpy().squeeze()
        if audio_np.ndim == 2 and audio_np.shape[0] == 2:
            audio_np = audio_np.T
        elif audio_np.ndim == 1:
            audio_np = np.stack([audio_np, audio_np], axis=-1)

        peak = np.max(np.abs(audio_np))
        if peak > 0:
            audio_np = audio_np / peak * 0.891

        return audio_np

    finally:
        handler.model.prepare_condition = original_prepare


def apply_production_effects(audio: np.ndarray, sr: int = 48000) -> np.ndarray:
    """Apply production effects chain for professional polish.

    Args:
        audio: Audio array (samples, 2).
        sr: Sample rate.

    Returns:
        Processed audio (samples, 2).
    """
    from pedalboard import (
        Pedalboard, Compressor, Gain, Reverb, Chorus, Distortion, Delay,
    )

    board = Pedalboard([
        Distortion(drive_db=8.0),
        Compressor(
            threshold_db=-18.0, ratio=3.0, attack_ms=10.0, release_ms=100.0
        ),
        Chorus(rate_hz=1.5, depth=0.4, mix=0.3),
        Delay(delay_seconds=0.125, feedback=0.25, mix=0.15),
        Reverb(room_size=0.4, damping=0.7, wet_level=0.2, dry_level=0.8),
        Gain(gain_db=-2.0),
    ])

    processed = board(audio.T.astype(np.float32), sr)
    peak = np.max(np.abs(processed))
    if peak > 0:
        processed = processed / peak * 0.891

    return processed.T
