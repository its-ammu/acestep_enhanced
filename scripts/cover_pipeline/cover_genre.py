"""Cover-mode genre shift: cover task + genre caption + hints + CFG.

Hybrid approach combining:
- Cover mode (structural skeleton from source audio)
- 25Hz bass hints (precise chord correction at latent level)
- LM-refined genre caption (creative timbre/style via CFG)
- Native repaint (fix failing sections)

This uses xl-sft which supports CFG (guidance_scale > 1.0), allowing
the caption to drive timbre/style while cover mode + hints lock chords.

Flow:
1. Qwen describes original → LM reimagines into new genre (caption)
2. Extract 25Hz hints from bass stem (chord reinforcement)
3. Cover mode with instrumental as source + genre caption + hints
4. Native repaint fixes sections that still drift
"""

import gc
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
import torch
from loguru import logger


def refine_caption_for_genre(
    caption: str,
    lyrics: str,
    bpm: int,
    keyscale: str,
    duration: float,
    lm_model: str = "acestep-5Hz-lm-4B",
    lm_temperature: float = 0.85,
) -> str:
    """Refine genre caption using LM format_sample_from_input.

    The LM takes our base caption and produces a natural, detailed
    music description that the DiT responds well to.

    Args:
        caption: Base genre-shifted caption (from rearrangement or Qwen).
        lyrics: Temporal script (for structural context).
        bpm: BPM.
        keyscale: Key.
        duration: Duration in seconds.
        lm_model: LM model name.
        lm_temperature: Sampling temperature.

    Returns:
        Enhanced caption string, or original if LM fails.
    """
    from acestep.llm_inference import LLMHandler

    logger.info(f"Loading {lm_model} for caption enhancement...")
    llm = LLMHandler()
    llm.initialize(
        checkpoint_dir="checkpoints",
        lm_model_path=lm_model,
        backend="vllm",
        device="cuda",
    )

    try:
        logger.info("Enhancing caption via LM (format_sample)...")
        result_tuple = llm.format_sample_from_input(
            caption=caption,
            lyrics=lyrics,
            user_metadata={"bpm": bpm, "keyscale": keyscale, "duration": int(duration)},
            temperature=lm_temperature,
            use_constrained_decoding=True,
        )

        # format_sample_from_input returns (metadata_dict, status_message)
        if isinstance(result_tuple, tuple) and len(result_tuple) >= 2:
            metadata_dict, status = result_tuple
        else:
            metadata_dict = result_tuple if isinstance(result_tuple, dict) else {}
            status = ""

        if metadata_dict and metadata_dict.get("caption"):
            refined = metadata_dict["caption"]
            logger.info(f"LM enhanced caption: {refined[:150]}")
            return refined
        else:
            logger.info("LM didn't enhance caption, using original")
            return caption
    except Exception as e:
        logger.warning(f"LM caption enhancement failed: {e}")
        return caption
    finally:
        del llm
        gc.collect()
        torch.cuda.empty_cache()
        logger.info("LM unloaded, VRAM freed")


def generate_cover_genre(
    handler,
    source_audio_path: str,
    caption: str,
    lyrics: str,
    hints: Optional[torch.Tensor] = None,
    bpm: Optional[int] = None,
    keyscale: Optional[str] = None,
    audio_cover_strength: float = 0.65,
    cover_noise_strength: float = 0.0,
    guidance_scale: float = 8.5,
    inference_steps: int = 28,
    shift: float = 3.0,
    refer_audios: Optional[list] = None,
    sr: int = 48000,
) -> Optional[np.ndarray]:
    """Generate genre-shifted cover with chord hints.

    Combines cover mode (structural skeleton) with 25Hz hints (chord lock)
    and genre caption via CFG (style shift).

    Args:
        handler: Initialized AceStepHandler.
        source_audio_path: Path to source audio (full instrumental).
        caption: Genre-shifted caption (from LM).
        lyrics: Energy-tagged temporal script.
        hints: Optional 25Hz bass hints tensor [B, T, D] for chord lock.
        bpm: BPM.
        keyscale: Key.
        audio_cover_strength: Structure preservation (0.5-0.7 recommended).
        cover_noise_strength: Additional noise (0.0 recommended).
        guidance_scale: CFG scale (8-10 for style transfer on sft).
        inference_steps: Diffusion steps (28-50 for sft).
        shift: Timestep shift (3.0 recommended).
        refer_audios: Optional timbre reference [[np.ndarray]].
        sr: Sample rate.

    Returns:
        Generated audio (samples, 2) or None.
    """
    import librosa

    # Load source audio
    src_audio, sr_src = sf.read(source_audio_path)
    if src_audio.ndim == 1:
        src_audio = np.stack([src_audio, src_audio], axis=-1)
    if sr_src != sr:
        src_audio = librosa.resample(src_audio.T, orig_sr=sr_src, target_sr=sr).T

    target_wavs = torch.tensor(src_audio.T, dtype=torch.float32).unsqueeze(0)
    duration = len(src_audio) / sr

    metas = [{"audio_duration": duration, "time_signature": "4/4"}]
    if bpm:
        metas[0]["bpm"] = bpm
    if keyscale:
        metas[0]["keyscale"] = keyscale

    logger.info(
        f"Cover genre generation: {duration:.0f}s, "
        f"acs={audio_cover_strength}, cns={cover_noise_strength}, "
        f"guidance={guidance_scale}, steps={inference_steps}, "
        f"hints={'yes' if hints is not None else 'no'}, "
        f"timbre_ref={'yes' if refer_audios else 'no'}"
    )

    # Monkey-patch hints if provided (chord reinforcement at latent level)
    original_prepare = None
    if hints is not None:
        original_prepare = handler.model.prepare_condition

        def patched_prepare(*args, **kwargs):
            kwargs["precomputed_lm_hints_25Hz"] = hints.to(
                device="cuda:0", dtype=torch.bfloat16
            )
            return original_prepare(*args, **kwargs)

        handler.model.prepare_condition = patched_prepare

    try:
        result = handler.service_generate(
            captions=caption,
            lyrics=lyrics,
            target_wavs=target_wavs,
            refer_audios=refer_audios,
            metas=metas,
            audio_cover_strength=audio_cover_strength,
            guidance_scale=guidance_scale,
            infer_steps=inference_steps,
            shift=shift,
            cover_noise_strength=cover_noise_strength,
            task_type="cover",
            infer_method="ode",
        )

        if "target_latents" not in result:
            logger.error("Cover generation failed — no latents returned")
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
        if original_prepare is not None:
            handler.model.prepare_condition = original_prepare


def repaint_failing_sections(
    handler,
    audio_path: str,
    failing_sections: list,
    caption: str,
    lyrics: str,
    source_audio_path: str,
    hints: Optional[torch.Tensor] = None,
    bpm: Optional[int] = None,
    keyscale: Optional[str] = None,
    audio_cover_strength: float = 0.8,
    cover_noise_strength: float = 0.0,
    guidance_scale: float = 8.5,
    inference_steps: int = 28,
    shift: float = 3.0,
    max_attempts: int = 2,
    sr: int = 48000,
) -> Optional[str]:
    """Repaint failing sections using native repaint task.

    Uses HIGHER audio_cover_strength than initial generation to lock
    chords more tightly on retry. Different seed gives different result.

    Args:
        handler: Initialized AceStepHandler.
        audio_path: Path to the generated audio to fix.
        failing_sections: List of SectionQCResult that failed.
        caption: Genre caption.
        lyrics: Temporal script.
        source_audio_path: Original source for cover conditioning.
        hints: Optional 25Hz hints for chord correction.
        bpm: BPM.
        keyscale: Key.
        audio_cover_strength: Higher than initial (0.7-0.8 for chord lock).
        cover_noise_strength: Additional noise.
        guidance_scale: CFG scale.
        inference_steps: Diffusion steps.
        shift: Timestep shift.
        max_attempts: Max retries per section.
        sr: Sample rate.

    Returns:
        Path to fixed audio, or None if all attempts failed.
    """
    import librosa

    current_path = audio_path

    # Monkey-patch hints if provided
    original_prepare = None
    if hints is not None:
        original_prepare = handler.model.prepare_condition

        def patched_prepare(*args, **kwargs):
            kwargs["precomputed_lm_hints_25Hz"] = hints.to(
                device="cuda:0", dtype=torch.bfloat16
            )
            return original_prepare(*args, **kwargs)

        handler.model.prepare_condition = patched_prepare

    try:
        for section in failing_sections:
            fixed = False
            for attempt in range(max_attempts):
                logger.info(
                    f"  Repaint [{section.label}] "
                    f"{section.start_sec:.1f}-{section.end_sec:.1f}s "
                    f"(attempt {attempt + 1}/{max_attempts}, acs={audio_cover_strength})"
                )

                # Load current audio as source for repaint
                src_audio, sr_src = sf.read(current_path)
                if src_audio.ndim == 1:
                    src_audio = np.stack([src_audio, src_audio], axis=-1)
                if sr_src != sr:
                    src_audio = librosa.resample(
                        src_audio.T, orig_sr=sr_src, target_sr=sr
                    ).T

                target_wavs = torch.tensor(
                    src_audio.T, dtype=torch.float32
                ).unsqueeze(0)
                duration = len(src_audio) / sr

                metas = [{"audio_duration": duration, "time_signature": "4/4"}]
                if bpm:
                    metas[0]["bpm"] = bpm
                if keyscale:
                    metas[0]["keyscale"] = keyscale

                result = handler.service_generate(
                    captions=caption,
                    lyrics=lyrics,
                    target_wavs=target_wavs,
                    metas=metas,
                    audio_cover_strength=audio_cover_strength,
                    guidance_scale=guidance_scale,
                    infer_steps=inference_steps,
                    shift=shift,
                    cover_noise_strength=cover_noise_strength,
                    repainting_start=[section.start_sec],
                    repainting_end=[section.end_sec],
                    task_type="repaint",
                    infer_method="ode",
                )

                if "target_latents" not in result:
                    logger.warning(f"  Repaint attempt {attempt + 1} failed")
                    continue

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

                # Save repainted version
                fixed_path = str(Path(current_path).parent / "repainted.flac")
                sf.write(fixed_path, audio_np, sr)
                current_path = fixed_path
                fixed = True
                logger.info(f"  ✅ Section repainted")
                break

            if not fixed:
                logger.warning(
                    f"  ❌ [{section.label}] could not be fixed "
                    f"after {max_attempts} attempts"
                )

        return current_path

    finally:
        if original_prepare is not None:
            handler.model.prepare_condition = original_prepare
