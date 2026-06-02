"""Complete-task remix: generate new instrumental around vocal + bass.

Professional remixer approach:
1. LM refines genre-shifted caption (creativity/complexity)
2. Mix vocal + bass at controlled ratio (harmonic anchor)
3. DiT "complete" task generates new instruments conditioned on:
   - The vocal+bass mix (harmonic/timing reference)
   - LM-refined caption (instrument style + performance detail)
   - Energy-tagged temporal script (dynamic arc)
   - Instruction (which tracks to generate)

The model hears the vocal melody and bass chords, then writes
complementary parts in the target genre. Chord accuracy is natural
because the model generates around what it hears.
"""

import gc
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
import torch
from loguru import logger


def refine_caption_with_lm(
    caption: str,
    lyrics: str,
    bpm: int,
    keyscale: str,
    duration: float,
    lm_model: str = "acestep-5Hz-lm-4B",
    lm_temperature: float = 0.85,
) -> str:
    """Refine genre-shifted caption using LM CoT for richer detail.

    The LM takes our base caption and enhances it with performance
    nuance, instrument interactions, and production detail. This drives
    the musical complexity of the output.

    Args:
        caption: Base genre-shifted caption from rearrangement.
        lyrics: Temporal script (for context about song structure).
        bpm: BPM.
        keyscale: Key.
        duration: Song duration in seconds.
        lm_model: LM checkpoint name.
        lm_temperature: Sampling temperature.

    Returns:
        Refined caption string, or original if LM fails.
    """
    from acestep.inference import format_sample
    from acestep.llm_inference import LLMHandler

    logger.info(f"Loading {lm_model} for caption refinement...")
    llm = LLMHandler()
    llm.initialize(
        checkpoint_dir="checkpoints",
        lm_model_path=lm_model,
        backend="vllm",
        device="cuda",
    )

    try:
        logger.info("Refining caption via LM CoT...")
        result = format_sample(
            llm_handler=llm,
            caption=caption,
            lyrics=lyrics,
            user_metadata={"bpm": bpm, "keyscale": keyscale, "duration": duration},
            temperature=lm_temperature,
        )

        if result.success and result.caption:
            refined = result.caption
            logger.info(f"LM refined caption: {refined[:150]}")
            return refined
        else:
            logger.warning("LM caption refinement failed, using base caption")
            return caption
    finally:
        del llm
        gc.collect()
        torch.cuda.empty_cache()
        logger.info("LM unloaded, VRAM freed")


def mix_vocal_and_bass(
    vocal_path: str,
    bass_path: str,
    bass_level_db: float = -6.0,
    sr: int = 48000,
) -> np.ndarray:
    """Mix vocal and bass stems at controlled ratio.

    The bass level controls how strongly chords constrain generation:
    - -3dB: strong harmonic constraint
    - -6dB: balanced (recommended)
    - -12dB: loose guidance, model has more freedom

    Args:
        vocal_path: Path to vocal stem.
        bass_path: Path to bass stem.
        bass_level_db: Bass level relative to vocal (negative = quieter).
        sr: Target sample rate.

    Returns:
        Mixed audio array (samples, 2) at target sample rate.
    """
    import librosa

    # Load vocal
    vocal, sr_v = sf.read(vocal_path)
    if vocal.ndim == 1:
        vocal = np.stack([vocal, vocal], axis=-1)
    if sr_v != sr:
        vocal = librosa.resample(vocal.T, orig_sr=sr_v, target_sr=sr).T

    # Load bass
    bass, sr_b = sf.read(bass_path)
    if bass.ndim == 1:
        bass = np.stack([bass, bass], axis=-1)
    if sr_b != sr:
        bass = librosa.resample(bass.T, orig_sr=sr_b, target_sr=sr).T

    # Match lengths
    min_len = min(len(vocal), len(bass))
    vocal = vocal[:min_len]
    bass = bass[:min_len]

    # Apply bass level
    bass_gain = 10 ** (bass_level_db / 20.0)
    mixed = vocal + bass * bass_gain

    # Normalize
    peak = np.max(np.abs(mixed))
    if peak > 0:
        mixed = mixed / peak * 0.891

    logger.info(
        f"Mixed vocal + bass (bass at {bass_level_db:.0f}dB): "
        f"{min_len / sr:.1f}s, peak normalized"
    )
    return mixed


def build_complete_instruction(instruments: dict[str, str]) -> str:
    """Build the instruction string for the complete task.

    Tells the model which tracks to generate (excluding vocal and bass
    which are already in the source audio).

    Args:
        instruments: Dict with "drums", "bass", "melodic" keys.

    Returns:
        Instruction string for the complete task.
    """
    # Map instrument names to track types the model understands
    melodic = instruments.get("melodic", "synth")
    drums = instruments.get("drums", "drums")

    # Determine track type for melodic instrument
    melodic_lower = melodic.lower()
    if any(k in melodic_lower for k in ("piano", "rhodes", "wurlitzer", "keyboard", "organ")):
        melodic_track = "keyboard"
    elif any(k in melodic_lower for k in ("guitar", "nylon", "steel")):
        melodic_track = "guitar"
    elif any(k in melodic_lower for k in ("trumpet", "brass", "horn")):
        melodic_track = "brass"
    elif any(k in melodic_lower for k in ("violin", "cello", "string")):
        melodic_track = "strings"
    elif any(k in melodic_lower for k in ("flute", "clarinet", "sax", "woodwind")):
        melodic_track = "woodwinds"
    else:
        melodic_track = "synth"

    instruction = (
        f"Complete the input track with drums, {melodic_track}, percussion:"
    )
    logger.info(f"Complete instruction: {instruction}")
    return instruction


def generate_complete_remix(
    handler,
    source_mix: np.ndarray,
    caption: str,
    lyrics: str,
    instruction: str,
    bpm: Optional[int] = None,
    keyscale: Optional[str] = None,
    guidance_scale: float = 7.0,
    inference_steps: int = 50,
    shift: float = 3.0,
    sr: int = 48000,
) -> Optional[np.ndarray]:
    """Run the complete task to generate new instrumental parts.

    Args:
        handler: Initialized AceStepHandler.
        source_mix: Vocal+bass mix array (samples, 2).
        caption: LM-refined genre-shifted caption.
        lyrics: Energy-tagged temporal script.
        instruction: Which tracks to generate.
        bpm: BPM.
        keyscale: Key.
        guidance_scale: CFG scale (7-9 for sft).
        inference_steps: Diffusion steps (32-65 for sft).
        shift: Timestep shift.
        sr: Sample rate.

    Returns:
        Generated audio array (samples, 2) or None.
    """
    # Prepare source as target_wavs tensor
    target_wavs = torch.tensor(
        source_mix.T, dtype=torch.float32
    ).unsqueeze(0)

    duration = len(source_mix) / sr
    metas = [{"audio_duration": duration, "time_signature": "4/4"}]
    if bpm:
        metas[0]["bpm"] = bpm
    if keyscale:
        metas[0]["keyscale"] = keyscale

    logger.info(
        f"Generating complete remix ({duration:.0f}s, "
        f"steps={inference_steps}, guidance={guidance_scale})..."
    )

    result = handler.service_generate(
        captions=caption,
        lyrics=lyrics,
        target_wavs=target_wavs,
        metas=metas,
        instructions=[instruction],
        guidance_scale=guidance_scale,
        infer_steps=inference_steps,
        shift=shift,
        task_type="complete",
        infer_method="ode",
    )

    if "target_latents" not in result:
        logger.error("Complete generation failed — no latents returned")
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

    logger.info(f"Complete remix generated: {len(audio_np) / sr:.1f}s")
    return audio_np
