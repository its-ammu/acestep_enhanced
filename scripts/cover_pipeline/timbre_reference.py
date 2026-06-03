"""Generate a timbre reference clip for the cover pipeline.

The ACE-Step model has a dedicated timbre encoder that accepts reference audio.
By generating a short clip of the TARGET instruments via text2music, we can
feed it to the timbre encoder during cover generation. This decouples timbre
from the semantic hints (which carry chord structure + original timbre).

Flow:
1. Use the rearranged caption (e.g., "Rhodes Piano. Upright Bass. Brushed Jazz Drums.")
2. Generate a 30s clip via text2music (no cover, no hints — pure caption)
3. Return the raw audio tensor for use as refer_audios in cover generation

The timbre encoder extracts the overall tonal palette from this reference,
then during cover generation the model produces output that lives in that
sound world while following the original's chord progression via hints.
"""

from pathlib import Path
from typing import Optional

import torch
from loguru import logger


def generate_timbre_reference(
    handler,
    caption: str,
    duration: float = 30.0,
    lyrics: str = "[Instrumental]\n",
    bpm: Optional[int] = None,
    keyscale: Optional[str] = None,
    guidance_scale: float = 15.0,
    inference_steps: int = 65,
    shift: float = 6.0,
) -> Optional[torch.Tensor]:
    """Generate a timbre reference clip via text2music.

    Args:
        handler: Initialized AceStepHandler with model loaded.
        caption: Instrument description (from rearrangement).
        duration: Target duration in seconds. Use full song duration for
            dynamic energy matching (quiet verses, loud choruses, solos).
        lyrics: Temporal script with section tags. When provided with full
            duration, the reference matches the song's energy arc.
        bpm: Optional BPM constraint.
        keyscale: Optional key constraint (e.g., "Eb Major").
        guidance_scale: CFG scale for text adherence.
        inference_steps: Diffusion steps.
        shift: Timestep shift.

    Returns:
        Raw audio tensor [2, samples] at 48kHz for use as refer_audios,
        or None if generation failed.
    """
    metas = [{"audio_duration": duration, "time_signature": "4/4"}]
    if bpm:
        metas[0]["bpm"] = bpm
    if keyscale:
        metas[0]["keyscale"] = keyscale

    logger.info(f"Generating timbre reference ({duration:.0f}s): {caption[:80]}")

    # text2music needs a target_wavs tensor for duration calculation
    # Pass silence — the model generates freely from caption
    samples = int(duration * 48000)
    target_wavs = torch.zeros(1, 2, samples)

    result = handler.service_generate(
        captions=caption,
        lyrics=lyrics,
        target_wavs=target_wavs,
        metas=metas,
        guidance_scale=guidance_scale,
        infer_steps=inference_steps,
        shift=shift,
        task_type="text2music",
        infer_method="ode",
    )

    if "target_latents" not in result:
        logger.warning("Timbre reference generation failed — no latents")
        return None

    latents = result["target_latents"]
    if latents.shape[-1] == 64:
        latents = latents.movedim(-1, -2)
    latents = latents.to(dtype=torch.bfloat16)

    with torch.no_grad():
        audio_tensor = handler.tiled_decode(latents)

    # audio_tensor is [1, 2, samples] — squeeze batch dim
    ref_audio = audio_tensor.squeeze(0).float()
    logger.info(f"Timbre reference: {ref_audio.shape} ({ref_audio.shape[-1]/48000:.1f}s)")
    return ref_audio
