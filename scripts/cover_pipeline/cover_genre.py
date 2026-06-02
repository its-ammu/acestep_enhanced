"""Cover-mode genre shift: use cover task with genre-shifted caption.

Clean approach — no LM codes, no latent blending, no monkey-patching.
Cover mode naturally follows chords from the source audio.
Genre shift comes from the caption driving timbre/style.

Flow:
1. LM refines genre-shifted caption (creativity/complexity)
2. Cover mode with bass/instrumental as source
3. Genre caption + higher cns pushes toward new genre
4. Native repaint fixes any failing sections
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
    """Refine genre caption using LM — constrained to enhance, not replace.

    Uses format_sample with user_metadata constraints to prevent the LM
    from overriding our genre intent. The LM enhances the caption with
    performance detail while respecting our genre choice.

    Args:
        caption: Base genre-shifted caption.
        lyrics: Temporal script.
        bpm: BPM.
        keyscale: Key.
        duration: Duration in seconds.
        lm_model: LM model name.
        lm_temperature: Sampling temperature.

    Returns:
        Enhanced caption (or original if LM fails).
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
        # Use format_sample_from_input which only does caption/metadata
        # refinement without generating audio codes
        logger.info("Enhancing caption via LM (format_sample)...")
        result = llm.format_sample_from_input(
            caption=caption,
            lyrics=lyrics,
            user_metadata={"bpm": bpm, "keyscale": keyscale, "duration": duration},
            temperature=lm_temperature,
            use_constrained_decoding=True,
        )

        if result and result.get("caption"):
            refined = result["caption"]
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
    bpm: Optional[int] = None,
    keyscale: Optional[str] = None,
    audio_cover_strength: float = 0.4,
    cover_noise_strength: float = 0.0,
    guidance_scale: float = 9.0,
    inference_steps: int = 28,
    shift: float = 3.0,
    refer_audios: Optional[list] = None,
    sr: int = 48000,
) -> Optional[np.ndarray]:
    """Generate genre-shifted cover using native cover mode.

    Cover mode encodes the source into latent space (structural skeleton).
    audio_cover_strength blends between structure-preserving and text-driven:
      - 0.8-1.0: subtle genre shift (keeps most of original)
      - 0.4-0.6: balanced (recognizable but transformed)
      - 0.2-0.4: radical reimagining (loose interpretation)

    The genre-shifted caption drives timbre and style via CFG.
    Higher guidance_scale = stronger adherence to caption.

    Args:
        handler: Initialized AceStepHandler.
        source_audio_path: Path to source audio (full instrumental recommended).
        caption: Genre-shifted caption (from LM refinement).
        lyrics: Energy-tagged temporal script.
        bpm: BPM.
        keyscale: Key.
        audio_cover_strength: Structure preservation (0.3-0.5 for genre shift).
        cover_noise_strength: Additional noise (0.0 recommended, use audio_cover_strength instead).
        guidance_scale: CFG scale (8-10 for style transformation).
        inference_steps: Diffusion steps (24-32 for sft).
        shift: Timestep shift (3.0 recommended).
        refer_audios: Optional timbre reference [[np.ndarray]] from target genre.
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
        f"audio_cover_strength={audio_cover_strength}, "
        f"cns={cover_noise_strength}, guidance={guidance_scale}, "
        f"steps={inference_steps}, timbre_ref={'yes' if refer_audios else 'no'}"
    )

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


def repaint_failing_sections(
    handler,
    audio_path: str,
    failing_sections: list,
    caption: str,
    lyrics: str,
    source_audio_path: str,
    bpm: Optional[int] = None,
    keyscale: Optional[str] = None,
    cover_noise_strength: float = 0.25,
    guidance_scale: float = 1.0,
    inference_steps: int = 8,
    shift: float = 3.0,
    max_attempts: int = 2,
    sr: int = 48000,
) -> Optional[str]:
    """Repaint failing sections using native repaint task.

    For each failing section, runs repaint with the same cover conditioning
    but a new seed. Only regenerates the time range of the failing section.

    Args:
        handler: Initialized AceStepHandler.
        audio_path: Path to the generated audio to fix.
        failing_sections: List of SectionQCResult that failed.
        caption: Genre caption.
        lyrics: Temporal script.
        source_audio_path: Original source for cover conditioning.
        bpm: BPM.
        keyscale: Key.
        cover_noise_strength: Same as initial generation.
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

    for section in failing_sections:
        fixed = False
        for attempt in range(max_attempts):
            logger.info(
                f"  Repaint [{section.label}] "
                f"{section.start_sec:.1f}-{section.end_sec:.1f}s "
                f"(attempt {attempt + 1}/{max_attempts})"
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
                audio_cover_strength=0.95,
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
                f"  ❌ [{section.label}] could not be fixed after {max_attempts} attempts"
            )

    return current_path
