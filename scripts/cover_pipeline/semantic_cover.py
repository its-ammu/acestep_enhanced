"""Semantic cover generation: extract hints from full mix, generate from instrumental.

Based on RyanOnTheInside's ACEStep15NativeCoverGuider approach:
- Extract semantic hints (chords/melody/rhythm) from the full mix
- Use instrumental latents as source (no vocal leakage)
- Pass both separately to DiT for generation

This gives the model correct harmonic information (from full mix)
while generating instrumental-only output (from instrumental source).
"""

import gc
from pathlib import Path
from typing import Optional

import torch
from loguru import logger


def extract_semantic_hints(handler, audio_path: str) -> torch.Tensor:
    """Extract semantic hints from audio using the model's tokenizer.

    Replicates RyanOnTheInside's extract_semantic_hints:
    1. VAE-encode audio → latents
    2. Tokenize latents → 5Hz quantized codes
    3. Detokenize → 25Hz semantic hints

    Args:
        handler: Initialized AceStepHandler with model loaded.
        audio_path: Path to audio file.

    Returns:
        Semantic hints tensor [B, T, D] (25Hz resolution).
    """
    import soundfile as sf
    import numpy as np

    # Load audio
    audio_data, sr = sf.read(audio_path)
    if audio_data.ndim == 1:
        audio_data = np.stack([audio_data, audio_data], axis=-1)

    # Convert to tensor [channels, samples]
    audio_tensor = torch.tensor(audio_data.T, dtype=torch.float32)

    # Resample to 48kHz if needed
    if sr != 48000:
        import torchaudio
        audio_tensor = torchaudio.functional.resample(audio_tensor, sr, 48000)

    # Ensure stereo [2, samples]
    if audio_tensor.shape[0] == 1:
        audio_tensor = audio_tensor.repeat(2, 1)

    # Add batch dimension [1, 2, samples]
    audio_tensor = audio_tensor.unsqueeze(0)

    device = next(handler.model.parameters()).device
    dtype = handler._get_vae_dtype()

    # VAE encode → latents using handler's tiled_encode
    logger.info(f"VAE encoding {audio_path}...")
    audio_tensor = audio_tensor.to(device=device, dtype=dtype)
    with torch.no_grad():
        latents = handler.tiled_encode(audio_tensor)

    logger.info(f"Latents shape: {latents.shape}")  # [B, D, T] = [1, 64, T]

    # Tokenize → 5Hz quantized codes
    # Model's tokenizer expects [B, T, D]
    latents_transposed = latents.movedim(-1, -2)  # [B, T, D]

    # Access tokenizer/detokenizer on handler.model
    tokenizer = handler.model.tokenizer
    detokenizer = handler.model.detokenizer

    # Move latents to model device
    model_device = next(tokenizer.parameters()).device
    model_dtype = next(tokenizer.parameters()).dtype
    latents_transposed = latents_transposed.to(device=model_device, dtype=model_dtype)

    with torch.no_grad():
        # Use the model's own tokenize method (handles padding + reshaping)
        # It needs silence_latent and attention_mask
        silence_latent = handler.silence_latent.to(device=model_device, dtype=model_dtype)
        # Expand silence_latent to match latents length if needed
        if silence_latent.shape[1] < latents_transposed.shape[1]:
            repeats = (latents_transposed.shape[1] // silence_latent.shape[1]) + 1
            silence_latent = silence_latent.repeat(1, repeats, 1)[:, :latents_transposed.shape[1], :]

        attention_mask = torch.ones(
            latents_transposed.shape[0], latents_transposed.shape[1],
            device=model_device,
        )

        # Call the model's tokenize (not the tokenizer module directly)
        quantized, indices, mask = handler.model.tokenize(
            latents_transposed, silence_latent, attention_mask,
        )

    logger.info(f"Quantized shape: {quantized.shape}, indices shape: {indices.shape}")

    # Detokenize → 25Hz semantic hints
    with torch.no_grad():
        lm_hints = handler.model.detokenize(quantized)

    logger.info(f"Semantic hints shape: {lm_hints.shape}")

    return lm_hints


def generate_with_separate_hints(
    handler,
    src_audio: str,
    full_mix_audio: str,
    caption: str,
    lyrics: str,
    output_dir: str,
    audio_cover_strength: float = 1.0,
    guidance_scale: float = 4.0,
    inference_steps: int = 65,
    shift: float = 6.0,
    cover_noise_strength: float = 0.9,
    seed: Optional[int] = None,
) -> Optional[str]:
    """Generate cover with semantic hints from full mix, source from instrumental.

    Args:
        handler: Initialized AceStepHandler.
        src_audio: Path to instrumental stem (no vocals).
        full_mix_audio: Path to original full mix (correct chords).
        caption: Style caption.
        lyrics: Structural lyrics.
        output_dir: Output directory.
        audio_cover_strength: Cover strength (0-1).
        guidance_scale: CFG scale.
        inference_steps: Diffusion steps.
        shift: Timestep shift.
        cover_noise_strength: How close to start from source (0=noise, 1=source).
        seed: Random seed.

    Returns:
        Path to generated audio file, or None if failed.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Extract semantic hints from full mix (correct chords)
    logger.info("Extracting semantic hints from full mix...")
    full_mix_hints = extract_semantic_hints(handler, full_mix_audio)

    # Generate using instrumental as src_audio but full mix hints
    logger.info("Generating with separate hints...")
    result = handler.service_generate(
        captions=caption,
        lyrics=lyrics,
        src_audio=src_audio,
        audio_cover_strength=audio_cover_strength,
        guidance_scale=guidance_scale,
        inference_steps=inference_steps,
        shift=shift,
        cover_noise_strength=cover_noise_strength,
        seed=seed or -1,
        use_random_seed=(seed is None),
        precomputed_lm_hints_25Hz=full_mix_hints,
        task_type="cover",
        infer_method="ode",
    )

    if result and result.get("success"):
        # Save audio
        import soundfile as sf
        audio_data = result["audios"][0]
        output_path = Path(output_dir) / f"semantic_cover_{seed or 'random'}.flac"
        sf.write(str(output_path), audio_data.T.cpu().numpy(), 48000)
        logger.info(f"Saved: {output_path}")
        return str(output_path)

    logger.error("Generation failed")
    return None
