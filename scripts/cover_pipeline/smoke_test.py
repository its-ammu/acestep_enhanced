"""Quick smoke test: generate 30s cover and save to check for garbled output.

Run on SageMaker:
    uv run --no-sync python -m scripts.cover_pipeline.smoke_test

Tests 3 configs to isolate the garble:
  A) Base model only (no LoRA, no ScragVAE)
  B) Base + ScragVAE
  C) Base + ScragVAE + LoRA (full pipeline config)

Listen to all 3 — whichever is garbled tells you the culprit.
"""

import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from loguru import logger


def run_smoke_test():
    """Generate 3 short covers to isolate garble source."""
    # Use the same instrumental stem from v20
    src_path = "data/output/just_the_way_it_is_v20/stems/melband/just the way it is, baby_(Instrumental)_model_bs_roformer_ep_317_sdr_12.wav"
    if not Path(src_path).exists():
        # Fallback: find any instrumental
        import glob
        candidates = glob.glob("data/output/*/stems/melband/*Instrumental*")
        if not candidates:
            logger.error("No instrumental stem found. Put one at the expected path.")
            return
        src_path = candidates[0]
    logger.info(f"Source: {src_path}")

    # Load source
    import librosa
    inst_audio, sr = sf.read(src_path)
    if inst_audio.ndim == 1:
        inst_audio = np.stack([inst_audio, inst_audio], axis=-1)
    if sr != 48000:
        inst_audio = librosa.resample(inst_audio.T, orig_sr=sr, target_sr=48000).T
        sr = 48000

    # Trim to 30s for speed
    max_samples = 30 * sr
    inst_audio = inst_audio[:max_samples]

    target_wavs = torch.tensor(inst_audio.T, dtype=torch.float32).unsqueeze(0)
    duration = len(inst_audio) / sr

    metas = [{"audio_duration": duration, "time_signature": "4/4", "bpm": 112, "keyscale": "Eb Major"}]
    caption = "electronic synth pop, warm pads, driving bass, crisp drums"
    lyrics = "[Instrumental]"

    out_dir = Path("data/output/smoke_test")
    out_dir.mkdir(parents=True, exist_ok=True)

    configs = [
        ("A_base_only", False, None),
        ("B_scragvae", True, None),
        ("C_scragvae_lora", True, "digital-acoustic"),
    ]

    from acestep.handler import AceStepHandler

    for label, use_scrag, lora_name in configs:
        logger.info(f"\n{'='*50}")
        logger.info(f"CONFIG {label}: scrag={use_scrag}, lora={lora_name}")
        logger.info("=" * 50)

        handler = AceStepHandler()
        handler.initialize_service(project_root=".", config_path="acestep-v15-xl-sft")

        # ScragVAE
        if use_scrag:
            from scripts.cover_pipeline.generate_semantic import _load_scrag_vae
            _load_scrag_vae(handler)

        # LoRA
        if lora_name:
            from scripts.cover_pipeline.generate_semantic import ensure_slider_lora
            lora_dir = ensure_slider_lora(lora_name)
            if lora_dir and lora_dir.exists():
                handler.add_lora(str(lora_dir))
                handler.set_lora_scale(0.7)
                handler.use_lora = True
                logger.info(f"LoRA loaded: {lora_dir}")

        # Generate
        result = handler.service_generate(
            captions=caption,
            lyrics=lyrics,
            target_wavs=target_wavs,
            metas=metas,
            audio_cover_strength=1.0,
            guidance_scale=12.0,
            infer_steps=65,
            shift=6.0,
            cover_noise_strength=0.15,
            task_type="cover",
            infer_method="ode",
        )

        if "target_latents" not in result:
            logger.error(f"{label}: generation failed — {list(result.keys())}")
            del handler
            torch.cuda.empty_cache()
            continue

        latents = result["target_latents"]
        logger.info(f"  latents shape: {latents.shape}, dtype: {latents.dtype}")

        if latents.shape[-1] == 64:
            latents = latents.movedim(-1, -2)

        # Cast to bfloat16 (the fix)
        latents = latents.to(dtype=torch.bfloat16)

        with torch.no_grad():
            audio_tensor = handler.tiled_decode(latents)

        audio_np = audio_tensor.float().cpu().numpy().squeeze()
        peak = np.max(np.abs(audio_np))
        if peak > 0:
            audio_np = audio_np / peak * 0.891

        out_path = out_dir / f"{label}.flac"
        sf.write(str(out_path), audio_np.T, 48000)

        # Quick stats
        rms = np.sqrt(np.mean(audio_np ** 2))
        logger.info(f"  ✅ Saved: {out_path}")
        logger.info(f"  RMS={rms:.4f}, peak={peak:.4f}, shape={audio_np.shape}")

        del handler
        import gc
        gc.collect()
        torch.cuda.empty_cache()

    logger.info(f"\n{'='*50}")
    logger.info(f"Done! Listen to files in: {out_dir}")
    logger.info("If A is clean but B/C garbled → ScragVAE issue")
    logger.info("If A+B clean but C garbled → LoRA issue")
    logger.info("If all garbled → base model or target_wavs issue")


if __name__ == "__main__":
    run_smoke_test()
