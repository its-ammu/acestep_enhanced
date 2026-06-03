"""Extract 5Hz audio codes from source audio for loose chord guidance.

The 5Hz codes are the quantized indices from the tokenizer — they encode
chord-level harmonic information without frame-level timbre detail (unlike
25Hz hints which entangle both).

Usage: Pass these codes as audio_codes to text2music task for loose
structural guidance while letting the DiT generate freely from caption.

At 5Hz resolution (vs 25Hz hints), codes capture ~200ms windows of
harmonic content — enough for chord roots but not instrument timbre.
"""

import torch
from loguru import logger


def extract_5hz_indices(handler, audio_path: str) -> torch.Tensor:
    """Extract raw 5Hz quantized indices from audio.

    Flow: audio → VAE encode → latents [B,D,T] → tokenize → indices [B, T//5]

    Args:
        handler: Initialized AceStepHandler with model loaded.
        audio_path: Path to audio file (bass stem recommended).

    Returns:
        Indices tensor [B, T_5hz] (integer codebook indices at 5Hz).
    """
    import numpy as np
    import soundfile as sf

    audio_data, sr = sf.read(audio_path)
    if audio_data.ndim == 1:
        audio_data = np.stack([audio_data, audio_data], axis=-1)

    audio_tensor = torch.tensor(audio_data.T, dtype=torch.float32)

    if sr != 48000:
        import torchaudio
        audio_tensor = torchaudio.functional.resample(audio_tensor, sr, 48000)

    if audio_tensor.shape[0] == 1:
        audio_tensor = audio_tensor.repeat(2, 1)

    audio_tensor = audio_tensor.unsqueeze(0)

    device = next(handler.model.parameters()).device
    dtype = handler._get_vae_dtype()

    logger.info(f"VAE encoding {audio_path} for 5Hz codes...")
    audio_tensor = audio_tensor.to(device=device, dtype=dtype)
    with torch.no_grad():
        latents = handler.tiled_encode(audio_tensor)

    logger.info(f"Latents shape: {latents.shape}")

    latents_transposed = latents.movedim(-1, -2)  # [B, T, D]

    model_device = next(handler.model.tokenizer.parameters()).device
    model_dtype = next(handler.model.tokenizer.parameters()).dtype
    latents_transposed = latents_transposed.to(device=model_device, dtype=model_dtype)

    with torch.no_grad():
        silence_latent = handler.silence_latent.to(
            device=model_device, dtype=model_dtype
        )
        if silence_latent.shape[1] < latents_transposed.shape[1]:
            repeats = (
                latents_transposed.shape[1] // silence_latent.shape[1]
            ) + 1
            silence_latent = silence_latent.repeat(1, repeats, 1)[
                :, : latents_transposed.shape[1], :
            ]

        attention_mask = torch.ones(
            latents_transposed.shape[0],
            latents_transposed.shape[1],
            device=model_device,
        )

        _quantized, indices, _mask = handler.model.tokenize(
            latents_transposed, silence_latent, attention_mask
        )

    logger.info(f"5Hz indices shape: {indices.shape}, range: [{indices.min()}, {indices.max()}]")
    return indices


def indices_to_code_string(indices: torch.Tensor) -> str:
    """Convert index tensor to <|audio_code_XXXXX|> string format.

    Args:
        indices: Tensor of integer indices [B, T] or [T].

    Returns:
        Serialized code string for GenerationParams.audio_codes.
    """
    flat = indices.flatten().cpu().tolist()
    return "".join(f"<|audio_code_{idx}|>" for idx in flat)


def extract_audio_codes_string(handler, audio_path: str) -> str:
    """One-shot: extract 5Hz codes from audio and return as string.

    Args:
        handler: Initialized AceStepHandler.
        audio_path: Path to source audio (bass stem recommended).

    Returns:
        Audio codes string ready for GenerationParams.
    """
    indices = extract_5hz_indices(handler, audio_path)
    code_str = indices_to_code_string(indices)
    n_codes = indices.numel()
    duration_approx = n_codes / 5.0
    logger.info(
        f"Extracted {n_codes} codes (~{duration_approx:.1f}s at 5Hz)"
    )
    return code_str
