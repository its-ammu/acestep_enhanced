"""Lego-based stem replacement for remix generation.

Instead of generating a full instrumental from scratch (cover mode),
this approach keeps the original drums + bass and uses ACE-Step's Lego
task to generate a new melodic/lead track that fits over them.

The Lego task hears the context audio (drums + bass) and generates a
complementary track in the specified style. Generation is done per-section
to stay within the model's effective range (~60s per call).
"""

from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
import torch
from loguru import logger


def _mix_stems_to_context(
    drums_path: Path,
    bass_path: Path,
    target_sr: int = 48000,
) -> np.ndarray:
    """Mix drums + bass stems into a single context audio array.

    Args:
        drums_path: Path to drums stem.
        bass_path: Path to bass stem.
        target_sr: Target sample rate.

    Returns:
        Stereo audio array (samples, 2) at target_sr.
    """
    import librosa

    drums, _ = librosa.load(str(drums_path), sr=target_sr, mono=False)
    bass, _ = librosa.load(str(bass_path), sr=target_sr, mono=False)

    if drums.ndim == 1:
        drums = np.stack([drums, drums])
    if bass.ndim == 1:
        bass = np.stack([bass, bass])

    min_len = min(drums.shape[1], bass.shape[1])
    drums = drums[:, :min_len]
    bass = bass[:, :min_len]

    mixed = drums + bass
    peak = np.max(np.abs(mixed))
    if peak > 0:
        mixed = mixed / peak * 0.9

    return mixed.T  # (samples, 2)


def generate_lego_stem(
    handler,
    context_audio: np.ndarray,
    track_name: str,
    caption: str,
    duration: float,
    segments: Optional[list] = None,
    lyrics: str = "[Instrumental]\n",
    global_caption: str = "",
    bpm: Optional[int] = None,
    keyscale: Optional[str] = None,
    guidance_scale: float = 15.0,
    inference_steps: int = 65,
    shift: float = 6.0,
) -> Optional[np.ndarray]:
    """Generate a stem using Lego task, section by section.

    Processes the song in sections (from SongFormer segments) to stay
    within the model's effective range. Each section sees the full
    context audio but only generates within its time range.

    Args:
        handler: Initialized AceStepHandler.
        context_audio: Context audio (drums+bass mix) as (samples, 2).
        track_name: Track to generate (e.g., "synth", "guitar").
        caption: Local description of desired track character.
        duration: Total duration in seconds.
        segments: Section boundaries from SongFormer. If None, generates
            in fixed 60s chunks.
        lyrics: Temporal script with section tags and energy hints.
        global_caption: Full song description for context.
        bpm: BPM constraint.
        keyscale: Key constraint.
        guidance_scale: CFG scale.
        inference_steps: Diffusion steps.
        shift: Timestep shift.

    Returns:
        Generated audio as (samples, 2) array, or None if failed.
    """
    if context_audio.ndim == 1:
        context_audio = np.stack([context_audio, context_audio], axis=-1)

    audio_tensor = torch.tensor(
        context_audio.T, dtype=torch.float32
    ).unsqueeze(0)

    metas = [{"audio_duration": duration, "time_signature": "4/4"}]
    if bpm:
        metas[0]["bpm"] = bpm
    if keyscale:
        metas[0]["keyscale"] = keyscale

    instruction = f"Generate the {track_name} track based on the audio context:"

    # Build section ranges (merge small sections, cap at 60s)
    if segments:
        ranges = _build_generation_ranges(segments, duration)
    else:
        ranges = _build_fixed_ranges(duration, chunk_sec=60.0)

    logger.info(
        f"Lego: generating '{track_name}' in {len(ranges)} sections "
        f"({duration:.0f}s total)"
    )

    # Generate section by section
    output_audio = np.zeros_like(context_audio)
    sr = 48000

    for i, (start_sec, end_sec) in enumerate(ranges):
        logger.info(
            f"  Section {i+1}/{len(ranges)}: "
            f"{start_sec:.1f}-{end_sec:.1f}s ({end_sec-start_sec:.1f}s)"
        )

        result = handler.service_generate(
            captions=caption,
            global_captions=[global_caption] if global_caption else None,
            lyrics=lyrics,
            target_wavs=audio_tensor,
            metas=metas,
            instructions=[instruction],
            guidance_scale=guidance_scale,
            infer_steps=inference_steps,
            shift=shift,
            task_type="lego",
            infer_method="ode",
            repainting_start=[start_sec],
            repainting_end=[end_sec],
        )

        if "target_latents" not in result:
            logger.warning(f"  Section {i+1} failed — skipping")
            continue

        latents = result["target_latents"]
        if latents.shape[-1] == 64:
            latents = latents.movedim(-1, -2)
        latents = latents.to(dtype=torch.bfloat16)

        with torch.no_grad():
            audio_out = handler.tiled_decode(latents)

        section_audio = audio_out.float().cpu().numpy().squeeze()
        if section_audio.ndim == 2 and section_audio.shape[0] == 2:
            section_audio = section_audio.T  # (samples, 2)

        # Extract just the generated section and splice into output
        start_sample = int(start_sec * sr)
        end_sample = min(int(end_sec * sr), len(output_audio))
        section_start_sample = start_sample
        section_end_sample = min(
            start_sample + (end_sample - start_sample),
            len(section_audio),
        )
        copy_len = min(
            end_sample - start_sample,
            section_end_sample - section_start_sample,
        )

        if copy_len > 0 and section_start_sample + copy_len <= len(section_audio):
            output_audio[start_sample:start_sample + copy_len] = (
                section_audio[section_start_sample:section_start_sample + copy_len]
            )

    # Normalize
    peak = np.max(np.abs(output_audio))
    if peak > 0:
        output_audio = output_audio / peak * 0.891

    return output_audio


def _build_generation_ranges(
    segments: list, duration: float, max_sec: float = 60.0
) -> list:
    """Build generation ranges from song segments.

    Merges adjacent small sections to reduce the number of Lego calls.
    Caps each range at max_sec.

    Args:
        segments: Section boundaries [{"start": float, "end": float}].
        duration: Total song duration.
        max_sec: Maximum seconds per generation call.

    Returns:
        List of (start_sec, end_sec) tuples.
    """
    ranges = []
    current_start = 0.0

    for seg in segments:
        seg_end = seg.get("end", duration)
        if seg_end - current_start > max_sec:
            # Current accumulated range is too long, flush it
            ranges.append((current_start, seg.get("start", current_start)))
            current_start = seg.get("start", current_start)

    # Final range
    if current_start < duration:
        ranges.append((current_start, duration))

    # Split any ranges that are still > max_sec
    final_ranges = []
    for start, end in ranges:
        while end - start > max_sec:
            final_ranges.append((start, start + max_sec))
            start += max_sec
        if end > start:
            final_ranges.append((start, end))

    return final_ranges


def _build_fixed_ranges(duration: float, chunk_sec: float = 60.0) -> list:
    """Build fixed-size generation ranges.

    Args:
        duration: Total duration.
        chunk_sec: Seconds per chunk.

    Returns:
        List of (start_sec, end_sec) tuples.
    """
    ranges = []
    start = 0.0
    while start < duration:
        end = min(start + chunk_sec, duration)
        ranges.append((start, end))
        start = end
    return ranges
