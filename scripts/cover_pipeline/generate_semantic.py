"""Semantic cover generation — integrated pipeline version.

Uses semantic hints from the full mix for chord accuracy while
generating new instrumental content guided by caption.

This is the preferred generation method when chord accuracy matters.
Falls back to generate.py (cli.py subprocess) if this approach fails.

Key parameters:
- cover_noise_strength=0.3: Balance of creativity vs source fidelity
- shift=6.0: Better compositional quality
- guidance_scale=7.0: Strong caption adherence for different instruments
- audio_cover_strength=1.0: Full cover conditioning
- Semantic hints from full mix: Correct chord structure
- reference_audio: Timbre reference (different instruments). When None,
  model uses silence_latent and relies purely on caption for timbre.
- hints_source: Which audio to extract hints from. Use bass stem for
  chord roots without full timbre leakage.
"""

import gc
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
import torch
from loguru import logger

from .semantic_cover import extract_semantic_hints


def _degrade_hints(
    hints: torch.Tensor,
    strength: float = 0.6,
    temporal_smooth_window: int = 5,
) -> torch.Tensor:
    """Degrade semantic hints to reduce timbre leakage while preserving chords.

    Applies two techniques:
    1. Temporal smoothing — averages hints over a window. Chords change slowly
       (every 1-4 beats at 25Hz = 25-100 frames), but timbre transients are
       frame-by-frame. Smoothing blurs timbre while keeping harmony.
    2. Noise mixing — blends hints with gaussian noise at (1-strength) ratio.
       The model can still "read" coarse harmonic patterns through moderate noise.

    Args:
        hints: Semantic hints tensor [B, T, D] (25Hz resolution).
        strength: 1.0=full hints (sounds same), 0.0=pure noise (no chord help).
            Recommended: 0.5-0.7 for different instruments with correct chords.
        temporal_smooth_window: Frames to average over. Higher = more chord-only.
            At 25Hz: 5 frames = 200ms (good default), 13 frames = ~1 beat at 120bpm.

    Returns:
        Degraded hints tensor, same shape and dtype.
    """
    if strength >= 1.0:
        return hints

    if strength <= 0.0:
        return torch.randn_like(hints) * hints.std()

    orig_dtype = hints.dtype
    # Work in float32 for numerical stability
    degraded = hints.clone().float()

    # Step 1: Temporal smoothing (blur timbre transients, keep slow-moving chords)
    if temporal_smooth_window > 1:
        B, T, D = degraded.shape
        pad = temporal_smooth_window // 2
        padded = torch.nn.functional.pad(
            degraded.permute(0, 2, 1),  # [B, D, T] for conv1d
            (pad, pad),
            mode="reflect",
        )
        kernel = torch.ones(
            1, 1, temporal_smooth_window, device=hints.device, dtype=torch.float32
        ) / temporal_smooth_window
        smoothed = torch.nn.functional.conv1d(
            padded.reshape(B * D, 1, -1),
            kernel,
            padding=0,
        ).reshape(B, D, T).permute(0, 2, 1)  # Back to [B, T, D]
        degraded = smoothed

    # Step 2: Mix with noise based on strength
    noise = torch.randn_like(degraded) * degraded.std()
    degraded = strength * degraded + (1.0 - strength) * noise

    # Cast back to original dtype (bfloat16)
    return degraded.to(dtype=orig_dtype)


def patch_repaint_for_cover(handler) -> None:
    """Monkey-patch the handler so repaint mode keeps is_covers=True.

    By default, ACE-Step sets is_covers=False during repaint, which disables
    semantic hints. This patch overrides that behavior for cover task so
    hints remain active during repainted sections.

    Args:
        handler: Initialized AceStepHandler.
    """
    original_build = handler._build_chunk_masks_and_src_latents

    def patched_build(*args, **kwargs):
        result = original_build(*args, **kwargs)
        # result is a tuple: (chunk_masks_tensor, is_covers_tensor, spans, ...)
        # Force is_covers to True for all items (we're always in cover mode)
        if hasattr(result, '__len__') and len(result) >= 2:
            # The is_covers tensor is typically the 2nd or 3rd element
            # Find it by checking for a boolean/long tensor
            result_list = list(result)
            for idx, item in enumerate(result_list):
                if isinstance(item, torch.Tensor) and item.dtype in (torch.bool, torch.long):
                    if item.dim() == 1 and item.shape[0] <= 4:  # batch-sized
                        result_list[idx] = torch.ones_like(item)
                        break
            return tuple(result_list)
        return result

    handler._build_chunk_masks_and_src_latents = patched_build
    logger.info("Patched repaint to keep is_covers=True (hints active during repaint)")


def ensure_slider_lora(slider_name: str) -> Optional[Path]:
    """Download and convert a concept slider LoRA to PEFT format.

    Auto-downloads from Xanthius/Ace-Step-1.5-XL-Concept-Sliders on first use,
    converts to PEFT-compatible directory with adapter_config.json.

    Available sliders: digital-acoustic, danceability, aggressive-gentle,
    bass, drum, choir-solo, closeness, age, oldradio.

    Args:
        slider_name: Name of the slider (e.g. "digital-acoustic", "danceability").

    Returns:
        Path to the PEFT adapter directory, or None if failed.
    """
    import json

    lora_dir = Path("checkpoints/lora_sliders") / slider_name
    config_file = lora_dir / "adapter_config.json"

    # Already converted
    if config_file.exists():
        return lora_dir

    try:
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file, save_file

        # Download from HuggingFace
        filename = f"ace-step_1-5_xl_{slider_name.replace('-', '-')}_slider.safetensors"
        cache_dir = "/home/ec2-user/SageMaker/.cache/huggingface"

        logger.info(f"Downloading concept slider: {slider_name}")
        src_path = hf_hub_download(
            "Xanthius/Ace-Step-1.5-XL-Concept-Sliders",
            filename,
            cache_dir=cache_dir,
        )

        # Convert: remap keys from diffusion_model.decoder.* to base_model.model.*
        tensors = load_file(src_path)
        remapped = {
            k.replace("diffusion_model.decoder.", "base_model.model."): v
            for k, v in tensors.items()
        }

        # Save PEFT-compatible files
        lora_dir.mkdir(parents=True, exist_ok=True)
        save_file(remapped, str(lora_dir / "adapter_model.safetensors"))

        # Detect rank from first lora_A tensor
        rank = 8  # default
        for k, v in remapped.items():
            if "lora_A.weight" in k:
                rank = v.shape[0]
                break

        # Detect target modules
        modules = set()
        for k in remapped.keys():
            parts = k.replace("base_model.model.", "").split(".lora_")[0]
            module_name = parts.split(".")[-1]
            modules.add(module_name)

        config = {
            "alpha_pattern": {},
            "auto_mapping": None,
            "base_model_name_or_path": "",
            "bias": "none",
            "fan_in_fan_out": False,
            "inference_mode": True,
            "init_lora_weights": True,
            "layers_pattern": None,
            "layers_to_transform": None,
            "lora_alpha": rank,
            "lora_dropout": 0.0,
            "modules_to_save": None,
            "peft_type": "LORA",
            "r": rank,
            "rank_pattern": {},
            "revision": None,
            "target_modules": sorted(modules),
            "task_type": None,
            "use_rslora": False,
        }
        config_file.write_text(json.dumps(config, indent=2))

        logger.info(f"Slider '{slider_name}' ready: {lora_dir} (rank={rank}, targets={sorted(modules)})")
        return lora_dir

    except Exception as e:
        logger.warning(f"Failed to setup slider '{slider_name}': {e}")
        return None


def _load_scrag_vae(handler) -> bool:
    """Swap in ScragVAE decoder for higher fidelity audio output.

    Downloads the ScragVAE weights on first use and replaces only the
    decoder portion of the VAE. Encoder stays the same so all DiT
    checkpoints remain compatible.

    Args:
        handler: Initialized AceStepHandler.

    Returns:
        True if successfully loaded, False otherwise.
    """
    try:
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file

        cache_dir = "/home/ec2-user/SageMaker/.cache/huggingface"
        scrag_path = hf_hub_download(
            "scragnog/Ace-Step-1.5-ScragVAE",
            "diffusion_pytorch_model.safetensors",
            cache_dir=cache_dir,
        )

        # Load only decoder keys
        scrag_weights = load_file(scrag_path)
        decoder_keys = {k: v for k, v in scrag_weights.items() if k.startswith("decoder.")}

        if not decoder_keys:
            logger.warning("ScragVAE: no decoder keys found")
            return False

        # Apply to the VAE (match dtype)
        vae = handler.vae
        vae_dtype = next(vae.parameters()).dtype
        decoder_keys = {k: v.to(dtype=vae_dtype) for k, v in decoder_keys.items()}
        vae.load_state_dict(decoder_keys, strict=False)
        logger.info(f"ScragVAE loaded: {len(decoder_keys)} decoder weights swapped (higher fidelity)")
        return True

    except ImportError:
        logger.warning("ScragVAE: huggingface_hub not available, skipping")
        return False
    except Exception as e:
        logger.warning(f"ScragVAE: failed to load ({e}), using default VAE")
        return False


def run_semantic_cover(
    src_audio: str | Path,
    full_mix_audio: str | Path,
    caption: str,
    lyrics: str,
    output_dir: str | Path,
    bpm: Optional[int] = None,
    keyscale: Optional[str] = None,
    timesignature: str = "4/4",
    cover_noise_strength: float = 0.3,
    guidance_scale: float = 7.0,
    inference_steps: int = 65,
    shift: float = 6.0,
    audio_cover_strength: float = 1.0,
    dit_model: str = "acestep-v15-xl-sft",
    hints_strength: float = 1.0,
    reference_audio: Optional[str | Path] = None,
    hints_source: Optional[str | Path] = None,
    lora_path: Optional[str | Path] = None,
    lora_scale: float = 0.7,
    use_scrag_vae: bool = True,
) -> Optional[Path]:
    """Generate cover using semantic hints for chord accuracy.

    The three key audio inputs control different aspects:
    - src_audio: Structure/timing (what gets noised and denoised)
    - hints source (full_mix_audio or hints_source): Chord information
    - reference_audio: Timbre/instrument character

    When reference_audio is None, the model uses silence_latent for timbre
    and relies purely on the caption for instrument choice. This is the
    cleanest way to get different-sounding instruments.

    When hints_source is provided, hints are extracted from that audio
    instead of full_mix_audio. Use the bass stem for chord roots without
    full arrangement timbre leakage.

    Args:
        src_audio: Path to instrumental stem (timing/structure source).
        full_mix_audio: Path to original full mix (default chord source).
        caption: Style description for the new instrumental.
        lyrics: Structural lyrics with energy hints.
        output_dir: Output directory.
        bpm: BPM override.
        keyscale: Key override.
        timesignature: Time signature.
        cover_noise_strength: 0.0=pure noise, 1.0=copy source. Default 0.3.
        guidance_scale: Caption adherence. Higher=more different instruments.
        inference_steps: Diffusion steps.
        shift: Timestep shift (6.0 recommended).
        audio_cover_strength: Cover conditioning strength.
        dit_model: DiT model name.
        hints_strength: How much of the hints to preserve (1.0=full,
            0.5=smoothed/degraded, 0.0=no hints). Default 1.0.
        reference_audio: Path to audio with desired instrument timbre.
            None=use caption only for timbre (most different from original).
        hints_source: Path to audio for hint extraction (overrides full_mix_audio).
            Use bass stem for chord-only hints without timbre leakage.
        lora_path: Path to PEFT LoRA adapter directory. None=no LoRA.
            Use concept sliders (e.g. checkpoints/lora_sliders/digital-acoustic).
        lora_scale: LoRA strength (0.7=subtle, 1.0=full, 1.5=strong/artifacts).
        use_scrag_vae: If True, swap in ScragVAE decoder for higher fidelity audio.
            Auto-downloads on first use. Default True.

    Returns:
        Path to generated audio file, or None if failed.
    """
    import librosa

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    src_audio = Path(src_audio)
    full_mix_audio = Path(full_mix_audio)

    # Determine which audio to extract hints from
    if isinstance(hints_source, (list, tuple)):
        hints_audio = None  # Will use blend logic below
    else:
        hints_audio = Path(hints_source) if hints_source else full_mix_audio

    handler = None
    try:
        from acestep.handler import AceStepHandler

        # Initialize handler
        logger.info(f"Loading DiT model: {dit_model}")
        handler = AceStepHandler()
        handler.initialize_service(project_root=".", config_path=dit_model)

        # Load ScragVAE decoder for higher fidelity audio
        if use_scrag_vae:
            _load_scrag_vae(handler)

        # Load LoRA concept slider if specified
        if lora_path:
            lora_dir = Path(lora_path)
            # If it's a slider name (not a path), auto-download and convert
            if not lora_dir.exists() and "/" not in str(lora_path):
                lora_dir = ensure_slider_lora(str(lora_path))
            if lora_dir and lora_dir.exists():
                logger.info(f"Loading LoRA slider: {lora_dir} (scale={lora_scale})")
                result = handler.add_lora(str(lora_dir))
                logger.info(f"LoRA: {result}")
                handler.set_lora_scale(lora_scale)
                handler.use_lora = True

        # Extract semantic hints
        # If hints_source is a list/tuple of paths, blend them (bass+other approach)
        if isinstance(hints_source, (list, tuple)) and len(hints_source) >= 2:
            # Blend hints from multiple stems
            logger.info(f"Extracting and blending hints from {len(hints_source)} stems...")
            all_hints = []
            for stem_path in hints_source:
                if stem_path and Path(stem_path).exists():
                    h = extract_semantic_hints(handler, str(stem_path))
                    all_hints.append(h)
                    logger.info(f"  Hints from {Path(stem_path).name}: {h.shape}")

            if len(all_hints) >= 2:
                # Match lengths (trim to shortest)
                min_t = min(h.shape[1] for h in all_hints)
                all_hints = [h[:, :min_t, :] for h in all_hints]
                # Equal-weight blend
                weight = 1.0 / len(all_hints)
                hints = sum(h * weight for h in all_hints)
                logger.info(f"Blended hints: {hints.shape} ({len(all_hints)} stems, equal weight)")
            elif len(all_hints) == 1:
                hints = all_hints[0]
            else:
                logger.warning("No valid stems for hint extraction, using full mix")
                hints = extract_semantic_hints(handler, str(full_mix_audio))
        else:
            # Single source (bass stem or full mix)
            source = hints_audio if hints_audio else full_mix_audio
            logger.info(f"Extracting semantic hints from: {source.name}")
            hints = extract_semantic_hints(handler, str(source))

        logger.info(f"Hints shape: {hints.shape}")

        # Optional: degrade hints to reduce timbre leakage
        if hints_strength < 1.0:
            logger.info(f"Degrading hints: strength={hints_strength}")
            hints = _degrade_hints(hints, strength=hints_strength)

        # Monkey-patch to inject hints
        original_prepare = handler.model.prepare_condition

        def patched_prepare(*args, **kwargs):
            kwargs["precomputed_lm_hints_25Hz"] = hints.to(
                device="cuda:0", dtype=torch.bfloat16
            )
            return original_prepare(*args, **kwargs)

        handler.model.prepare_condition = patched_prepare

        # Load instrumental as target_wavs (structure/timing source)
        inst_audio, sr = sf.read(str(src_audio))
        if inst_audio.ndim == 1:
            inst_audio = np.stack([inst_audio, inst_audio], axis=-1)
        if sr != 48000:
            inst_audio = librosa.resample(
                inst_audio.T, orig_sr=sr, target_sr=48000
            ).T
            sr = 48000
        target_wavs = torch.tensor(inst_audio.T, dtype=torch.float32).unsqueeze(0)
        duration = len(inst_audio) / sr

        # Prepare reference audio for timbre conditioning
        refer_audios = None
        if reference_audio and Path(reference_audio).exists():
            logger.info(f"Using reference audio for timbre: {Path(reference_audio).name}")
            ref_data, ref_sr = sf.read(str(reference_audio))
            if ref_data.ndim == 1:
                ref_data = np.stack([ref_data, ref_data], axis=-1)
            if ref_sr != 48000:
                ref_data = librosa.resample(
                    ref_data.T, orig_sr=ref_sr, target_sr=48000
                ).T
            ref_tensor = torch.tensor(ref_data.T, dtype=torch.float32)
            refer_audios = [[ref_tensor]]
        else:
            # None = silence_latent = caption-only timbre (most different)
            logger.info("No reference audio — timbre from caption only")

        # Build metas
        metas = [{"audio_duration": duration, "time_signature": timesignature}]
        if bpm:
            metas[0]["bpm"] = bpm
        if keyscale:
            metas[0]["keyscale"] = keyscale

        # Generate
        logger.info(
            f"Generating: cns={cover_noise_strength}, cfg={guidance_scale}, "
            f"steps={inference_steps}, shift={shift}, "
            f"hints_strength={hints_strength}, "
            f"ref_audio={'yes' if refer_audios else 'no (caption timbre)'}"
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

        # Decode latents
        if "target_latents" in result:
            latents = result["target_latents"]
            if latents.shape[-1] == 64:
                latents = latents.movedim(-1, -2)

            logger.info(f"Decoding latents: {latents.shape}")
            with torch.no_grad():
                audio_tensor = handler.tiled_decode(latents)

            audio_np = audio_tensor.float().cpu().numpy().squeeze()

            # Normalize
            peak = np.max(np.abs(audio_np))
            if peak > 0:
                audio_np = audio_np / peak * 0.891  # -1dB

            output_path = output_dir / "semantic_cover.flac"
            sf.write(str(output_path), audio_np.T, 48000)
            logger.info(f"Saved: {output_path}")
            return output_path
        else:
            logger.error(
                f"Generation returned unexpected result: {list(result.keys())}"
            )
            return None

    except Exception as e:
        logger.error(f"Semantic cover generation failed: {e}")
        return None
    finally:
        # Aggressively cleanup GPU
        del handler
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()


def _spectral_complexity(audio: np.ndarray, sr: int = 48000) -> float:
    """Measure spectral complexity of audio (anti-bland metric).

    Higher values = more interesting/varied spectral content.
    Low values = flat/static/bland output.

    Combines:
    - Spectral flux (how much the spectrum changes over time)
    - Spectral bandwidth variation (dynamic range of frequencies)

    Args:
        audio: Audio array (samples, channels) or (samples,).
        sr: Sample rate.

    Returns:
        Complexity score (0-1 range, higher = more interesting).
    """
    import librosa

    if audio.ndim == 2:
        audio = audio.mean(axis=-1)  # Mono

    # Spectral flux (frame-to-frame spectral change)
    S = np.abs(librosa.stft(audio, n_fft=2048, hop_length=512))
    flux = np.sqrt(np.mean(np.diff(S, axis=1) ** 2))

    # Spectral bandwidth variation
    bandwidth = librosa.feature.spectral_bandwidth(S=S, sr=sr)[0]
    bw_variation = np.std(bandwidth) / (np.mean(bandwidth) + 1e-10)

    # Normalize to 0-1 range (empirical bounds)
    flux_score = min(flux / 0.5, 1.0)
    bw_score = min(bw_variation / 0.5, 1.0)

    return 0.6 * flux_score + 0.4 * bw_score


def run_semantic_cover_multi_seed(
    src_audio: str | Path,
    full_mix_audio: str | Path,
    caption: str,
    lyrics: str,
    output_dir: str | Path,
    num_variants: int = 4,
    bpm: Optional[int] = None,
    keyscale: Optional[str] = None,
    timesignature: str = "4/4",
    cover_noise_strength: float = 0.15,
    guidance_scale: float = 12.0,
    inference_steps: int = 65,
    shift: float = 6.0,
    audio_cover_strength: float = 1.0,
    dit_model: str = "acestep-v15-xl-sft",
    hints_strength: float = 1.0,
    hints_source: Optional[str | Path] = None,
    lora_path: Optional[str | Path] = None,
    lora_scale: float = 0.7,
    use_scrag_vae: bool = True,
) -> Optional[Path]:
    """Generate multiple cover variants and auto-select the best one.

    Scores each variant on:
    - Chroma correlation (chord accuracy vs original)
    - Spectral complexity (anti-bland: musical interest/variation)

    Picks the variant with the best combined score.

    Args:
        num_variants: Number of seeds to try (default 4).
        All other args: same as run_semantic_cover.

    Returns:
        Path to the best generated audio file, or None if all failed.
    """
    import librosa as lr
    import random

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    src_audio = Path(src_audio)
    full_mix_audio = Path(full_mix_audio)

    # Determine hints source
    if isinstance(hints_source, (list, tuple)):
        hints_audio = None
    else:
        hints_audio = Path(hints_source) if hints_source else full_mix_audio

    handler = None
    try:
        from acestep.handler import AceStepHandler

        logger.info(f"Multi-seed generation: {num_variants} variants")
        handler = AceStepHandler()
        handler.initialize_service(project_root=".", config_path=dit_model)

        # Load ScragVAE
        if use_scrag_vae:
            _load_scrag_vae(handler)

        # Load LoRA
        if lora_path:
            lora_dir = Path(lora_path)
            if not lora_dir.exists() and "/" not in str(lora_path):
                lora_dir = ensure_slider_lora(str(lora_path))
            if lora_dir and lora_dir.exists():
                logger.info(f"Loading LoRA: {lora_dir} (scale={lora_scale})")
                handler.add_lora(str(lora_dir))
                handler.set_lora_scale(lora_scale)
                handler.use_lora = True

        # Extract hints
        if isinstance(hints_source, (list, tuple)) and len(hints_source) >= 2:
            all_hints = []
            for stem_path in hints_source:
                if stem_path and Path(stem_path).exists():
                    h = extract_semantic_hints(handler, str(stem_path))
                    all_hints.append(h)
            if len(all_hints) >= 2:
                min_t = min(h.shape[1] for h in all_hints)
                all_hints = [h[:, :min_t, :] for h in all_hints]
                weight = 1.0 / len(all_hints)
                hints = sum(h * weight for h in all_hints)
            elif all_hints:
                hints = all_hints[0]
            else:
                hints = extract_semantic_hints(handler, str(full_mix_audio))
        else:
            source = hints_audio if hints_audio else full_mix_audio
            hints = extract_semantic_hints(handler, str(source))

        if hints_strength < 1.0:
            hints = _degrade_hints(hints, strength=hints_strength)

        # Monkey-patch hints
        original_prepare = handler.model.prepare_condition

        def patched_prepare(*args, **kwargs):
            kwargs["precomputed_lm_hints_25Hz"] = hints.to(
                device="cuda:0", dtype=torch.bfloat16
            )
            return original_prepare(*args, **kwargs)

        handler.model.prepare_condition = patched_prepare

        # Load source audio
        inst_audio, sr = sf.read(str(src_audio))
        if inst_audio.ndim == 1:
            inst_audio = np.stack([inst_audio, inst_audio], axis=-1)
        if sr != 48000:
            inst_audio = lr.resample(inst_audio.T, orig_sr=sr, target_sr=48000).T
            sr = 48000
        target_wavs = torch.tensor(inst_audio.T, dtype=torch.float32).unsqueeze(0)
        duration = len(inst_audio) / sr

        metas = [{"audio_duration": duration, "time_signature": timesignature}]
        if bpm:
            metas[0]["bpm"] = bpm
        if keyscale:
            metas[0]["keyscale"] = keyscale

        # Load original for scoring
        orig_mono = lr.load(str(src_audio), sr=22050, mono=True)[0]

        # Generate variants in batches of 2 for speed
        seeds = random.sample(range(1, 10000), num_variants)
        variants = []
        batch_size = 2  # 2 at a time fits in 24GB VRAM

        for batch_start in range(0, len(seeds), batch_size):
            batch_seeds = seeds[batch_start:batch_start + batch_size]
            logger.info(f"Batch {batch_start//batch_size + 1}: seeds={batch_seeds}")

            # Duplicate inputs for batch
            batch_target_wavs = target_wavs.repeat(len(batch_seeds), 1, 1)
            batch_metas = metas * len(batch_seeds)
            batch_captions = [caption] * len(batch_seeds)
            batch_lyrics = [lyrics] * len(batch_seeds)

            result = handler.service_generate(
                captions=batch_captions,
                lyrics=batch_lyrics,
                target_wavs=batch_target_wavs,
                metas=batch_metas,
                audio_cover_strength=audio_cover_strength,
                guidance_scale=guidance_scale,
                infer_steps=inference_steps,
                shift=shift,
                cover_noise_strength=cover_noise_strength,
                task_type="cover",
                infer_method="ode",
                seed=batch_seeds,
            )

            if "target_latents" not in result:
                logger.warning(f"Batch failed: {list(result.keys())}")
                continue

            latents = result["target_latents"]
            if latents.shape[-1] == 64:
                latents = latents.movedim(-1, -2)

            # Decode and score each in the batch
            for idx, seed in enumerate(batch_seeds):
                single_latent = latents[idx:idx+1]
                with torch.no_grad():
                    audio_tensor = handler.tiled_decode(single_latent)

                audio_np = audio_tensor.float().cpu().numpy().squeeze()
                peak = np.max(np.abs(audio_np))
                if peak > 0:
                    audio_np = audio_np / peak * 0.891

                var_path = output_dir / f"variant_{seed}.flac"
                sf.write(str(var_path), audio_np.T, 48000)

                # Score
                gen_mono = lr.load(str(var_path), sr=22050, mono=True)[0]
                min_len = min(len(orig_mono), len(gen_mono))

                from .stem_quality_gate import _chroma_correlation, _rhythm_correlation
                chroma = _chroma_correlation(orig_mono[:min_len], gen_mono[:min_len])
                complexity = _spectral_complexity(audio_np.T, sr=48000)

                chroma_score = 1.0 - abs(chroma - 0.55) / 0.55
                chroma_score = max(0.0, chroma_score)
                combined = 0.4 * chroma_score + 0.6 * complexity

                variants.append({
                    "seed": seed,
                    "path": var_path,
                    "chroma": chroma,
                    "complexity": complexity,
                    "combined": combined,
                })
                logger.info(
                    f"  Seed {seed}: chroma={chroma:.3f}, complexity={complexity:.3f}, "
                    f"combined={combined:.3f}"
                )

        if not variants:
            logger.error("All variants failed")
            return None

        # Pick best
        best = max(variants, key=lambda v: v["combined"])
        logger.info(
            f"\nBest variant: seed={best['seed']}, "
            f"chroma={best['chroma']:.3f}, complexity={best['complexity']:.3f}"
        )

        # Copy best to final output path
        final_path = output_dir / "semantic_cover.flac"
        import shutil
        shutil.copy2(str(best["path"]), str(final_path))
        logger.info(f"Selected: {final_path}")

        # Log all variants for reference
        logger.info("All variants:")
        for v in sorted(variants, key=lambda x: x["combined"], reverse=True):
            marker = " ← BEST" if v["seed"] == best["seed"] else ""
            logger.info(
                f"  seed={v['seed']}: chroma={v['chroma']:.3f}, "
                f"complexity={v['complexity']:.3f}, combined={v['combined']:.3f}{marker}"
            )

        return final_path

    except Exception as e:
        logger.error(f"Multi-seed generation failed: {e}")
        import traceback
        traceback.print_exc()
        return None
    finally:
        del handler
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
