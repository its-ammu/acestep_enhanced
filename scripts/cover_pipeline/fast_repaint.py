"""Fast section-level repaint using mask-based regeneration.

Adapted from RyanOnTheInside's ACE-Step cover guider approach.
Instead of generating a full song and extracting a section (slow),
this uses chunk_masks to regenerate only the failing section while
the model sees surrounding context (fast, better transitions).

Key difference from native ACE-Step repaint: we keep is_covers=True
and semantic hints active, so chord accuracy is preserved during repaint.
Native repaint sets is_covers=False which disables hints → silence.
"""

import numpy as np
import torch
import soundfile as sf
from loguru import logger
from pathlib import Path
from typing import Optional


def _load_silence_latent(handler) -> torch.Tensor:
    """Get the silence_latent from the handler (already loaded).

    The silence_latent is a learned tensor that tells the model
    "generate new content here." It's NOT zeros.

    Args:
        handler: Initialized AceStepHandler.

    Returns:
        Silence latent tensor [T, D] on handler device.
    """
    # The handler loads silence_latent during initialization
    if hasattr(handler, "silence_latent") and handler.silence_latent is not None:
        sl = handler.silence_latent
        # Handler stores it as [1, T, D] after transpose
        if sl.dim() == 3:
            sl = sl[0]  # [T, D]
        return sl.to(device=handler.device, dtype=torch.bfloat16)

    # Fallback: load from file
    silence_path = Path("checkpoints") / "silence_latent.pt"
    if silence_path.exists():
        sl = torch.load(str(silence_path), map_location="cpu", weights_only=True)
        # File format is [1, D, T], transpose to [T, D]
        sl = sl.squeeze(0).transpose(0, 1)
        return sl.to(device=handler.device, dtype=torch.bfloat16)

    raise RuntimeError(
        "silence_latent not found. Ensure handler is initialized or "
        "checkpoints/silence_latent.pt exists."
    )


def repaint_section_fast(
    handler,
    clean_latents: torch.Tensor,
    start_sec: float,
    end_sec: float,
    caption: str,
    lyrics: str,
    metas: list[dict],
    hints: Optional[torch.Tensor] = None,
    guidance_scale: float = 15.0,
    infer_steps: int = 65,
    shift: float = 6.0,
    sr: int = 48000,
    fps: float = 25.0,
) -> Optional[torch.Tensor]:
    """Repaint a section using mask-based regeneration (fast).

    Uses chunk_masks to tell the model which frames to regenerate.
    Keeps is_covers=True and semantic hints active for chord accuracy.
    The model sees surrounding clean frames as context.

    Args:
        handler: Initialized AceStepHandler with model loaded.
        clean_latents: Clean latents from initial generation [B, D, T].
        start_sec: Start of section to repaint (seconds).
        end_sec: End of section to repaint (seconds).
        caption: Caption for generation.
        lyrics: Lyrics/temporal script.
        metas: Metadata dicts.
        hints: Semantic hints tensor (already on device). If None, uses
               whatever hints are monkey-patched on the handler.
        guidance_scale: CFG scale.
        infer_steps: Diffusion steps.
        shift: Timestep shift.
        sr: Sample rate.
        fps: Latent frames per second (48000/1920 = 25).

    Returns:
        New latents with the section repainted [B, D, T], or None if failed.
    """
    # Calculate frame indices
    start_frame = int(start_sec * fps)
    end_frame = int(end_sec * fps)
    total_frames = clean_latents.shape[-1]
    start_frame = max(0, min(start_frame, total_frames - 1))
    end_frame = max(start_frame + 1, min(end_frame, total_frames))

    logger.info(
        f"Fast repaint: frames {start_frame}-{end_frame} "
        f"({start_sec:.1f}-{end_sec:.1f}s) of {total_frames} total"
    )

    device = clean_latents.device
    dtype = clean_latents.dtype
    batch_size = clean_latents.shape[0]

    try:
        # Load silence latent
        silence_latent = _load_silence_latent(handler)

        # Tile silence_latent to match total_frames
        # silence_latent is [T_silence, D], we need [total_frames, D]
        if silence_latent.shape[0] < total_frames:
            num_tiles = (total_frames // silence_latent.shape[0]) + 1
            silence_tiled = silence_latent.repeat(num_tiles, 1)[:total_frames]
        else:
            silence_tiled = silence_latent[:total_frames]

        # Build src_latents: clean everywhere, silence in repaint region
        # clean_latents is [B, D, T], src_latents needs same shape
        src_latents = clean_latents.clone()
        # Transpose silence to [D, T] for assignment
        silence_region = silence_tiled[start_frame:end_frame].transpose(0, 1)
        src_latents[:, :, start_frame:end_frame] = silence_region.unsqueeze(0)

        # Build chunk_masks: 0 = preserve, 1 = regenerate
        chunk_masks = torch.zeros(
            batch_size, 1, total_frames, device=device, dtype=dtype
        )
        chunk_masks[:, :, start_frame:end_frame] = 1.0

        # Monkey-patch the handler to inject our masks
        # Override _build_chunk_masks_and_src_latents to return our custom masks
        original_build = handler._build_chunk_masks_and_src_latents

        def patched_build(*args, **kwargs):
            """Override mask building to use our repaint masks."""
            result = original_build(*args, **kwargs)
            # result is (chunk_masks_tensor, spans, is_covers_tensor,
            #            src_latents_out, repaint_mask)
            result_list = list(result)

            # Override chunk_masks (index 0)
            cm = chunk_masks.squeeze(1).to(dtype=torch.bool)
            result_list[0] = cm

            # Override is_covers to True (index 2) — keep hints active!
            result_list[2] = torch.ones(
                batch_size, dtype=torch.bool, device=device
            )

            # Override src_latents (index 3)
            # src_latents in handler format is [B, T, D]
            src_t = src_latents.movedim(-1, -2)  # [B, D, T] → [B, T, D]
            result_list[3] = src_t

            return tuple(result_list)

        handler._build_chunk_masks_and_src_latents = patched_build

        # Convert clean_latents to target_wavs format for service_generate
        # We need to pass target_wavs so the handler encodes them to latents
        # But we already have latents — so we'll decode to audio first
        # Actually, simpler: pass the src_latents directly via the patch above
        # and provide a dummy target_wavs that triggers cover mode

        # Decode clean latents to get target_wavs
        # Ensure latents are on the correct device for VAE decode
        decode_latents = clean_latents.to(device=handler.device)
        with torch.no_grad():
            clean_audio = handler.tiled_decode(decode_latents)
        target_wavs = clean_audio.to(dtype=torch.float32)

        # Duration from latents
        duration = total_frames / fps

        # Generate with our patched masks
        result = handler.service_generate(
            captions=caption,
            lyrics=lyrics,
            target_wavs=target_wavs,
            metas=metas,
            audio_cover_strength=1.0,
            guidance_scale=guidance_scale,
            infer_steps=infer_steps,
            shift=shift,
            cover_noise_strength=0.25,
            task_type="cover",
            infer_method="ode",
        )

        if "target_latents" not in result:
            logger.warning("Fast repaint: generation returned no latents")
            return None

        new_latents = result["target_latents"]
        if new_latents.shape[-1] == 64:
            new_latents = new_latents.movedim(-1, -2)
        new_latents = new_latents.to(dtype=dtype)

        # Blend: keep clean latents for preserved region, new for repainted
        output_latents = clean_latents.clone()
        output_latents[:, :, start_frame:end_frame] = (
            new_latents[:, :, start_frame:end_frame]
        )

        logger.info(
            f"Fast repaint complete: replaced frames {start_frame}-{end_frame}"
        )
        return output_latents

    except Exception as e:
        logger.error(f"Fast repaint failed: {e}")
        import traceback
        traceback.print_exc()
        return None
    finally:
        # Restore original mask builder
        handler._build_chunk_masks_and_src_latents = original_build


def repaint_section_fast_audio(
    handler,
    clean_audio: np.ndarray,
    start_sec: float,
    end_sec: float,
    target_wavs: torch.Tensor,
    caption: str,
    lyrics: str,
    metas: list[dict],
    hints: Optional[torch.Tensor] = None,
    guidance_scale: float = 15.0,
    infer_steps: int = 65,
    shift: float = 6.0,
    sr: int = 48000,
    cover_noise_strength: float = 0.25,
) -> Optional[np.ndarray]:
    """Fast repaint using latent cropping for speed.

    Instead of encoding/decoding the full song, crops to just the section
    plus context padding, runs diffusion on the small tensor, then splices
    the result back into the clean audio.

    Args:
        handler: Initialized AceStepHandler.
        clean_audio: Clean audio array (samples, channels).
        start_sec: Section start.
        end_sec: Section end.
        target_wavs: Source audio tensor (for cover conditioning).
        caption: Caption.
        lyrics: Lyrics.
        metas: Metadata.
        hints: Semantic hints (optional).
        guidance_scale: CFG scale.
        infer_steps: Diffusion steps.
        shift: Timestep shift.
        sr: Sample rate.
        cover_noise_strength: Noise strength for repaint.

    Returns:
        Audio with the section repainted (samples, channels), or None.
    """
    fps = sr / 1920  # 48000/1920 = 25
    context_sec = 3.0  # seconds of context on each side

    # Calculate crop boundaries (section + context padding)
    crop_start_sec = max(0.0, start_sec - context_sec)
    crop_end_sec = min(len(clean_audio) / sr, end_sec + context_sec)

    crop_start_sample = int(crop_start_sec * sr)
    crop_end_sample = int(crop_end_sec * sr)

    # Crop audio to section + context
    cropped_audio = clean_audio[crop_start_sample:crop_end_sample]

    logger.info(
        f"  Fast repaint (cropped): {start_sec:.1f}-{end_sec:.1f}s "
        f"(crop: {crop_start_sec:.1f}-{crop_end_sec:.1f}s, "
        f"{crop_end_sec - crop_start_sec:.1f}s vs {len(clean_audio)/sr:.1f}s full)"
    )

    # Encode cropped audio to latents
    audio_tensor = torch.tensor(
        cropped_audio.T, dtype=torch.float32
    ).unsqueeze(0).to(handler.device)

    with torch.no_grad():
        cropped_latents = handler.tiled_encode(audio_tensor)
    cropped_latents = cropped_latents.to(device=handler.device, dtype=torch.bfloat16)

    # Calculate repaint region within the cropped latent
    local_start_sec = start_sec - crop_start_sec
    local_end_sec = end_sec - crop_start_sec
    start_frame = int(local_start_sec * fps)
    end_frame = int(local_end_sec * fps)
    total_frames = cropped_latents.shape[-1]
    start_frame = max(0, min(start_frame, total_frames - 1))
    end_frame = max(start_frame + 1, min(end_frame, total_frames))

    # Run fast repaint on cropped latents
    new_latents = repaint_section_fast(
        handler=handler,
        clean_latents=cropped_latents,
        start_sec=local_start_sec,
        end_sec=local_end_sec,
        caption=caption,
        lyrics=lyrics,
        metas=metas,
        hints=hints,
        guidance_scale=guidance_scale,
        infer_steps=infer_steps,
        shift=shift,
        sr=sr,
        fps=fps,
    )

    if new_latents is None:
        return None

    # Decode only the cropped latents (much smaller than full song)
    new_latents_device = new_latents.to(device=handler.device)
    with torch.no_grad():
        audio_out = handler.tiled_decode(new_latents_device)

    new_audio_cropped = audio_out.float().cpu().numpy().squeeze()
    if new_audio_cropped.ndim == 2 and new_audio_cropped.shape[0] == 2 and new_audio_cropped.shape[1] > 2:
        new_audio_cropped = new_audio_cropped.T
    elif new_audio_cropped.ndim == 1:
        new_audio_cropped = np.stack([new_audio_cropped, new_audio_cropped], axis=-1)

    # Normalize the cropped output
    peak = np.max(np.abs(new_audio_cropped))
    if peak > 0:
        new_audio_cropped = new_audio_cropped / peak * 0.891

    # Splice the repainted section back into clean audio with crossfade
    output = clean_audio.copy()
    crossfade_samples = int(0.3 * sr)  # 300ms crossfade

    # The repainted region in the cropped output
    repaint_start_in_crop = int(local_start_sec * sr)
    repaint_end_in_crop = min(int(local_end_sec * sr), len(new_audio_cropped))

    # Map back to full audio coordinates
    start_sample = int(start_sec * sr)
    end_sample = min(int(end_sec * sr), len(clean_audio))

    # Ensure we have enough repainted audio
    repaint_len = min(
        repaint_end_in_crop - repaint_start_in_crop,
        end_sample - start_sample,
    )

    if repaint_len <= 0:
        logger.warning("Fast repaint: no audio to splice")
        return None

    # Crossfade in
    fade_start = max(0, start_sample - crossfade_samples)
    if fade_start < start_sample and fade_start >= crop_start_sample:
        fade_len = start_sample - fade_start
        fade = (1 - np.cos(np.linspace(0, np.pi, fade_len))) / 2
        fade = fade.reshape(-1, 1)
        # Get the corresponding region from cropped output
        crop_fade_start = int((fade_start - crop_start_sample) / sr * sr)
        crop_fade_start = fade_start - crop_start_sample
        crop_fade_end = crop_fade_start + fade_len
        if crop_fade_end <= len(new_audio_cropped):
            output[fade_start:start_sample] = (
                clean_audio[fade_start:start_sample] * (1 - fade) +
                new_audio_cropped[crop_fade_start:crop_fade_end] * fade
            )

    # Replace section
    output[start_sample:start_sample + repaint_len] = (
        new_audio_cropped[repaint_start_in_crop:repaint_start_in_crop + repaint_len]
    )

    # Crossfade out
    fade_end = min(len(output), end_sample + crossfade_samples)
    if fade_end > end_sample:
        fade_len = fade_end - end_sample
        fade = (1 + np.cos(np.linspace(0, np.pi, fade_len))) / 2
        fade = fade.reshape(-1, 1)
        crop_out_start = repaint_end_in_crop
        crop_out_end = crop_out_start + fade_len
        if crop_out_end <= len(new_audio_cropped):
            output[end_sample:fade_end] = (
                new_audio_cropped[crop_out_start:crop_out_end] * fade +
                clean_audio[end_sample:fade_end] * (1 - fade)
            )

    logger.info(f"  Fast repaint: {start_sec:.1f}-{end_sec:.1f}s replaced")
    return output
