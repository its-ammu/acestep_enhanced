"""Hint-guided section repaint using cover mode with selective noise.

Based on RyanOnTheInside's approach: run cover mode with hints active,
but only add noise to the failing section. Passing sections stay clean
(preserved from the creative output). The failing section gets
regenerated with hints guiding the chord progression.

This is the correct way to fix chord drift in specific sections without
affecting the creative output in sections that already sound good.
"""

import numpy as np
import soundfile as sf
import torch
from loguru import logger
from pathlib import Path
from typing import Optional


def repaint_section_with_hints(
    handler,
    creative_audio_path: str,
    start_sec: float,
    end_sec: float,
    hints: torch.Tensor,
    caption: str,
    lyrics: str,
    bpm: Optional[int] = None,
    keyscale: Optional[str] = None,
    guidance_scale: float = 15.0,
    inference_steps: int = 65,
    shift: float = 6.0,
    cover_noise_strength: float = 0.15,
    sr: int = 48000,
) -> Optional[np.ndarray]:
    """Repaint a section using cover mode with hints for chord correction.

    The approach: encode the creative output to latents, run cover mode
    with hints active. The cover_noise_strength controls how much of the
    creative output is preserved vs regenerated. At cns=0.15, the model
    starts close to the creative output but hints guide it toward correct
    chords.

    After generation, splice ONLY the repainted section back into the
    creative output, preserving everything else.

    Args:
        handler: AceStepHandler with model loaded (hints NOT monkey-patched).
        creative_audio_path: Path to the full creative output audio.
        start_sec: Start of failing section.
        end_sec: End of failing section.
        hints: Semantic hints tensor [B, T, D] for chord guidance.
        caption: Caption for generation.
        lyrics: Temporal script.
        bpm: BPM.
        keyscale: Key.
        guidance_scale: CFG scale.
        inference_steps: Diffusion steps.
        shift: Timestep shift.
        cover_noise_strength: How much to regenerate (0.15=mostly preserve).
        sr: Sample rate.

    Returns:
        Full audio with the section repainted, or None if failed.
    """
    # Load creative output
    creative_audio, _ = sf.read(creative_audio_path)
    if creative_audio.ndim == 1:
        creative_audio = np.stack([creative_audio, creative_audio], axis=-1)

    # Prepare as target_wavs (the creative output IS our source)
    target_wavs = torch.tensor(
        creative_audio.T, dtype=torch.float32
    ).unsqueeze(0)

    duration = len(creative_audio) / sr
    metas = [{"audio_duration": duration, "time_signature": "4/4"}]
    if bpm:
        metas[0]["bpm"] = bpm
    if keyscale:
        metas[0]["keyscale"] = keyscale

    # Monkey-patch hints for this generation
    original_prepare = handler.model.prepare_condition

    def patched_prepare(*args, **kwargs):
        kwargs["precomputed_lm_hints_25Hz"] = hints.to(
            device="cuda:0", dtype=torch.bfloat16
        )
        return original_prepare(*args, **kwargs)

    handler.model.prepare_condition = patched_prepare

    try:
        logger.info(
            f"  Hint repaint: {start_sec:.1f}-{end_sec:.1f}s "
            f"(cns={cover_noise_strength})"
        )

        # Run cover mode on the full creative output with hints
        # The model starts from the creative output (cns=0.15) and
        # hints guide it toward correct chords
        result = handler.service_generate(
            captions=caption,
            lyrics=lyrics,
            target_wavs=target_wavs,
            metas=metas,
            audio_cover_strength=1.0,
            guidance_scale=guidance_scale,
            infer_steps=inference_steps,
            shift=shift,
            cover_noise_strength=cover_noise_strength,
            task_type="cover",
            infer_method="ode",
        )

        if "target_latents" not in result:
            logger.warning("Hint repaint: generation failed")
            return None

        latents = result["target_latents"]
        if latents.shape[-1] == 64:
            latents = latents.movedim(-1, -2)
        latents = latents.to(dtype=torch.bfloat16)

        with torch.no_grad():
            audio_tensor = handler.tiled_decode(latents)

        repainted_full = audio_tensor.float().cpu().numpy().squeeze()
        if repainted_full.ndim == 2 and repainted_full.shape[0] == 2:
            repainted_full = repainted_full.T
        elif repainted_full.ndim == 1:
            repainted_full = np.stack([repainted_full, repainted_full], axis=-1)

        # Splice: keep creative output everywhere EXCEPT the failing section
        output = creative_audio.copy()
        start_sample = int(start_sec * sr)
        end_sample = min(int(end_sec * sr), len(output), len(repainted_full))

        # Crossfade edges (300ms)
        crossfade_samples = int(0.3 * sr)

        # Fade in at start of repainted section
        fade_start = max(0, start_sample - crossfade_samples // 2)
        fade_end = min(len(output), start_sample + crossfade_samples // 2)
        if fade_end > fade_start:
            fade_len = fade_end - fade_start
            fade = np.linspace(0, 1, fade_len).reshape(-1, 1)
            output[fade_start:fade_end] = (
                creative_audio[fade_start:fade_end] * (1 - fade)
                + repainted_full[fade_start:fade_end] * fade
            )

        # Replace middle (no crossfade needed)
        mid_start = fade_end
        mid_end = max(mid_start, end_sample - crossfade_samples // 2)
        if mid_end > mid_start and mid_end <= len(repainted_full):
            output[mid_start:mid_end] = repainted_full[mid_start:mid_end]

        # Fade out at end of repainted section
        fade_out_start = mid_end
        fade_out_end = min(len(output), end_sample + crossfade_samples // 2)
        if fade_out_end > fade_out_start and fade_out_end <= len(repainted_full):
            fade_len = fade_out_end - fade_out_start
            fade = np.linspace(1, 0, fade_len).reshape(-1, 1)
            output[fade_out_start:fade_out_end] = (
                repainted_full[fade_out_start:fade_out_end] * fade
                + creative_audio[fade_out_start:fade_out_end] * (1 - fade)
            )

        return output

    finally:
        # Restore original prepare_condition
        handler.model.prepare_condition = original_prepare
