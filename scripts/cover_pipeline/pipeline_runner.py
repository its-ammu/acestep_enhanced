"""Single entry point for the cover pipeline with all options.

Orchestrates: separation → analysis → timeline → generation → quality gate → mix.
"""

import glob
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from loguru import logger


@dataclass
class PipelineConfig:
    """All pipeline configuration in one place."""

    # Input/output
    input_song: str
    output_dir: str

    # Models
    dit_model: str = "acestep-v15-xl-sft"
    lm_model: str = "acestep-5Hz-lm-4B"
    qwen_model: str = "Qwen/Qwen2.5-Omni-7B"

    # Generation parameters
    audio_cover_strength: float = 0.95
    cover_noise_strength: float = 0.15
    guidance_scale: float = 12.0
    inference_steps: int = 65
    shift: float = 6.0
    batch_size: int = 1

    # Generation mode: "semantic" (bass hints) or "cli" (subprocess)
    generation_mode: str = "semantic"

    # Caption
    refine_caption: bool = True

    # Quality gate
    quality_gate: bool = True
    bad_stem_action: str = "hints"  # "hints", "swap", or "keep"

    # Mix
    per_stem_mix: bool = False

    # Hints degradation (for semantic route)
    hints_strength: float = 1.0  # 1.0=full (same timbre), 0.5=chords only, 0.0=no hints

    # LoRA concept slider (name or path)
    lora_path: Optional[str] = "digital-acoustic"
    lora_scale: float = 0.7

    # ScragVAE (improved audio fidelity)
    use_scrag_vae: bool = True


def run_pipeline(cfg: PipelineConfig) -> Optional[Path]:
    """Run the full cover pipeline.

    Args:
        cfg: Pipeline configuration.

    Returns:
        Path to final cover file, or None if failed.
    """
    from .deps import ensure_dependencies
    from .stem_separation import separate_stems
    from .audio_analysis import analyze_bpm_and_structure, analyze_key
    from .structure_timeline import generate_structure_timeline
    from .mix_daw import mix_with_effects
    from .stem_quality_gate import evaluate_stems, remix_with_swap

    ensure_dependencies(include_optional=True)

    input_song = cfg.input_song
    output_dir = cfg.output_dir
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # === STEP 1: Stem Separation ===
    logger.info("=" * 50)
    logger.info("STEP 1: Stem Separation")
    stems = separate_stems(input_song, f"{output_dir}/stems")

    # === STEP 2: Metadata ===
    logger.info("=" * 50)
    logger.info("STEP 2: Metadata Detection")
    bpm_result = analyze_bpm_and_structure(input_song)
    key = analyze_key(input_song, bass_stem_path=stems.bass)
    metadata = {
        "bpm": int(round(bpm_result["bpm"])),
        "keyscale": key,
        "timesignature": "4/4",
    }
    logger.info(f"BPM: {metadata['bpm']}, Key: {metadata['keyscale']}")

    # === STEP 3: Structure Timeline ===
    logger.info("=" * 50)
    logger.info("STEP 3: Structure Timeline")
    timeline = generate_structure_timeline(
        audio_path=input_song,  # Full mix for SongFormer (needs vocals for section detection)
        metadata=metadata,
        qwen_model=cfg.qwen_model,
        stem_paths={
            "drums": stems.drums,
            "bass": stems.bass,
            "other": stems.other,
        },
    )
    logger.info(f"Caption: {timeline.caption[:150]}")

    # === STEP 4: Cover Generation ===
    logger.info("=" * 50)
    logger.info("STEP 4: Cover Generation")

    instrumental_path = None

    if cfg.generation_mode == "semantic":
        # Semantic generation with section-level QC:
        # 1. Generate "creative" version (low cns, sounds different)
        # 2. Generate "safe" version (higher cns, preserves bass/structure)
        # 3. Per-section analysis: splice best parts together
        from .generate_semantic import (
            _load_scrag_vae, ensure_slider_lora, extract_semantic_hints, _degrade_hints,
        )

        logger.info("Using semantic generation with section-level QC")

        # Single handler load for both generations
        from acestep.handler import AceStepHandler
        import torch
        import soundfile as sf_gen

        handler = AceStepHandler()
        handler.initialize_service(project_root=".", config_path=cfg.dit_model)

        # Load ScragVAE
        if cfg.use_scrag_vae:
            _load_scrag_vae(handler)

        # Extract hints BEFORE LoRA (LoRA backup uses GPU memory during load)
        hints = extract_semantic_hints(handler, str(stems.bass))
        if cfg.hints_strength < 1.0:
            hints = _degrade_hints(hints, strength=cfg.hints_strength)
        logger.info(f"Hints: {hints.shape}")

        # Load LoRA (after hints extraction to avoid OOM)
        if cfg.lora_path:
            lora_dir = Path(cfg.lora_path)
            if not lora_dir.exists() and "/" not in str(cfg.lora_path):
                lora_dir = ensure_slider_lora(str(cfg.lora_path))
            if lora_dir and lora_dir.exists():
                handler.add_lora(str(lora_dir))
                handler.set_lora_scale(cfg.lora_scale)
                handler.use_lora = True
                logger.info(f"LoRA: {lora_dir} at scale {cfg.lora_scale}")

        # Monkey-patch hints
        original_prepare = handler.model.prepare_condition

        def patched_prepare(*args, **kwargs):
            kwargs["precomputed_lm_hints_25Hz"] = hints.to(
                device="cuda:0", dtype=torch.bfloat16
            )
            return original_prepare(*args, **kwargs)

        handler.model.prepare_condition = patched_prepare

        # Load source audio
        import librosa as lr
        import numpy as np

        inst_audio, sr_inst = sf_gen.read(str(stems.instrumental))
        if inst_audio.ndim == 1:
            inst_audio = np.stack([inst_audio, inst_audio], axis=-1)
        if sr_inst != 48000:
            inst_audio = lr.resample(inst_audio.T, orig_sr=sr_inst, target_sr=48000).T
            sr_inst = 48000
        target_wavs = torch.tensor(inst_audio.T, dtype=torch.float32).unsqueeze(0)
        duration = len(inst_audio) / sr_inst
        metas = [{"audio_duration": duration, "time_signature": "4/4"}]
        if metadata.get("bpm"):
            metas[0]["bpm"] = metadata["bpm"]
        if metadata.get("keyscale"):
            metas[0]["keyscale"] = metadata["keyscale"]

        # Generate creative version (low cns)
        logger.info(f"Generating creative version (cns={cfg.cover_noise_strength})...")
        creative_result = handler.service_generate(
            captions=timeline.caption,
            lyrics=timeline.lyrics,
            target_wavs=target_wavs,
            metas=metas,
            audio_cover_strength=cfg.audio_cover_strength,
            guidance_scale=cfg.guidance_scale,
            infer_steps=cfg.inference_steps,
            shift=cfg.shift,
            cover_noise_strength=cfg.cover_noise_strength,
            task_type="cover",
            infer_method="ode",
        )

        # Generate safe version (higher cns for bass consistency)
        logger.info("Generating safe version (cns=0.3)...")
        safe_result = handler.service_generate(
            captions=timeline.caption,
            lyrics=timeline.lyrics,
            target_wavs=target_wavs,
            metas=metas,
            audio_cover_strength=cfg.audio_cover_strength,
            guidance_scale=cfg.guidance_scale,
            infer_steps=cfg.inference_steps,
            shift=cfg.shift,
            cover_noise_strength=0.3,
            task_type="cover",
            infer_method="ode",
        )

        # Decode both versions
        gen_dir = Path(f"{output_dir}/generated")
        gen_dir.mkdir(parents=True, exist_ok=True)
        creative_path = None
        safe_path = None

        for label, result in [("creative", creative_result), ("safe", safe_result)]:
            if "target_latents" not in result:
                logger.warning(f"{label} generation failed")
                continue
            latents = result["target_latents"]
            if latents.shape[-1] == 64:
                latents = latents.movedim(-1, -2)
            latents = latents.to(dtype=torch.bfloat16)
            with torch.no_grad():
                audio_tensor = handler.tiled_decode(latents)
            audio_np = audio_tensor.float().cpu().numpy().squeeze()
            peak = np.max(np.abs(audio_np))
            if peak > 0:
                audio_np = audio_np / peak * 0.891
            out_path = gen_dir / f"{label}.flac"
            sf_gen.write(str(out_path), audio_np.T, 48000)
            logger.info(f"{label}: {out_path}")
            if label == "creative":
                creative_path = out_path
            else:
                safe_path = out_path

        # Cleanup handler AFTER QC step (not here)
        # del handler — moved to after QC

        # Splice: creative for verses, safe for choruses/outros
        if creative_path and safe_path:
            from .section_splice import analyze_and_splice

            creative_audio, sr_c = sf_gen.read(str(creative_path))
            safe_audio, _ = sf_gen.read(str(safe_path))
            bass_mono = lr.load(str(stems.bass), sr=sr_c, mono=True)[0]

            logger.info("Analyzing sections and splicing...")
            spliced = analyze_and_splice(
                audio_creative=creative_audio,
                audio_safe=safe_audio,
                sr=sr_c,
                segments=timeline.segments,
                original_bass_audio=bass_mono,
                original_sr=sr_c,
                expected_key=metadata.get("keyscale", ""),
            )

            spliced_path = gen_dir / "semantic_cover.flac"
            sf_gen.write(str(spliced_path), spliced, sr_c)
            instrumental_path = spliced_path
            logger.info(f"Spliced output: {spliced_path}")
        elif creative_path:
            instrumental_path = creative_path
        elif safe_path:
            instrumental_path = safe_path
        else:
            instrumental_path = None
    else:
        # CLI subprocess generation (original approach)
        from .generate import run_cover_generation, GenerationConfig

        gen_config = GenerationConfig(
            dit_model=cfg.dit_model,
            lm_model=cfg.lm_model,
            audio_cover_strength=cfg.audio_cover_strength,
            cover_noise_strength=cfg.cover_noise_strength,
            guidance_scale=cfg.guidance_scale,
            inference_steps=cfg.inference_steps,
            shift=cfg.shift,
            batch_size=cfg.batch_size,
            bpm=metadata["bpm"],
            keyscale=metadata["keyscale"],
            timesignature="4/4",
            duration=None,
        )
        generated = run_cover_generation(
            src_audio=stems.instrumental,
            reference_audio=stems.instrumental,
            caption=timeline.caption,
            lyrics=timeline.lyrics,
            config=gen_config,
            audio_codes="",
            output_dir=f"{output_dir}/generated",
            refine_caption=cfg.refine_caption,
        )

        # Find generated file
        if not generated:
            flacs = sorted(
                glob.glob("output/*.flac") + glob.glob(f"{output_dir}/generated/*.flac"),
                key=os.path.getmtime,
                reverse=True,
            )
            generated = [Path(f) for f in flacs[:1]]

        if generated:
            instrumental_path = generated[0]

    if not instrumental_path:
        logger.error("No instrumental generated")
        return None

    logger.info(f"Generated: {instrumental_path}")

    # === STEP 5: Post-Generation QC ===
    if cfg.generation_mode == "semantic":
        logger.info("=" * 50)
        logger.info("STEP 5: Post-Generation QC")

        from .qc_retry_loop import run_qc_retry_loop

        try:
            # Reuse the handler from generation (still loaded with hints + LoRA)
            fixed_path = run_qc_retry_loop(
                handler=handler,
                generated_audio_path=str(instrumental_path),
                safe_audio_path=str(safe_path) if safe_path else None,
                original_instrumental_path=str(stems.instrumental),
                bpm=metadata.get("bpm", 100),
                key=metadata.get("keyscale", ""),
                segments=timeline.segments,
                target_wavs=target_wavs,
                metas=metas,
                caption=timeline.caption,
                lyrics=timeline.lyrics,
                hints=hints,
            )
            instrumental_path = fixed_path

        except Exception as e:
            logger.warning(f"QC retry loop failed: {e}. Using spliced generation.")

        # NOW cleanup handler (after QC is done)
        del handler
        import gc as gc_mod
        gc_mod.collect()
        torch.cuda.empty_cache()

    # === STEP 5b: Legacy Quality Gate (stem-level) ===
    if cfg.quality_gate and cfg.bad_stem_action != "keep":
        logger.info("=" * 50)
        logger.info("STEP 5: Quality Gate")

        # Demucs the AI instrumental
        ai_stem_dir = Path(output_dir) / "ai_stems"
        ai_stem_dir.mkdir(exist_ok=True)
        env = os.environ.copy()
        env["TORCHAUDIO_BACKEND"] = "soundfile"

        subprocess.run(
            [sys.executable, "-m", "demucs", "-n", "htdemucs_ft",
             "--out", str(ai_stem_dir), "--mp3", str(instrumental_path)],
            capture_output=True, text=True, timeout=300, env=env,
        )

        # Find AI stems
        ai_stem_base = ai_stem_dir / "htdemucs_ft"
        ai_dirs = list(ai_stem_base.iterdir()) if ai_stem_base.exists() else []

        if ai_dirs:
            ai_dir = ai_dirs[0]
            ai_stems = {}
            original_stems = {}

            for name in ["drums", "bass", "other"]:
                for ext in [".mp3", ".wav"]:
                    ai_path = ai_dir / f"{name}{ext}"
                    if ai_path.exists():
                        ai_stems[name] = ai_path
                        break

                # Original stems from initial Demucs
                if stems.drums and name == "drums":
                    original_stems["drums"] = stems.drums
                elif stems.bass and name == "bass":
                    original_stems["bass"] = stems.bass
                elif stems.other and name == "other":
                    original_stems["other"] = stems.other

            if ai_stems and original_stems:
                gate_result = evaluate_stems(original_stems, ai_stems)

                if gate_result.swap_stem:
                    logger.info(f"Bad stem detected: {gate_result.swap_stem}")

                    if cfg.bad_stem_action == "hints":
                        # Regenerate bad stem using semantic hints (correct chords)
                        final = _hints_regenerate_and_mix(
                            bad_stem=gate_result.swap_stem,
                            ai_stems=ai_stems,
                            original_stems=original_stems,
                            vocals_path=stems.vocals,
                            full_mix_path=input_song,
                            caption=timeline.caption,
                            metadata=metadata,
                            output_dir=output_dir,
                            input_song=input_song,
                            dit_model=cfg.dit_model,
                            hints_strength=cfg.hints_strength,
                        )
                        if final:
                            return final
                        # Fallback to swap if hints approach fails
                        logger.warning("Hints regeneration failed, falling back to swap")
                        final = remix_with_swap(
                            vocals_path=stems.vocals,
                            ai_stems=ai_stems,
                            original_stems=original_stems,
                            swap_stem=gate_result.swap_stem,
                            output_path=f"{output_dir}/final_cover.wav",
                            original_mix_path=input_song,
                        )
                        return final

                    elif cfg.bad_stem_action == "swap":
                        # Swap with original
                        final = remix_with_swap(
                            vocals_path=stems.vocals,
                            ai_stems=ai_stems,
                            original_stems=original_stems,
                            swap_stem=gate_result.swap_stem,
                            output_path=f"{output_dir}/final_cover.wav",
                            original_mix_path=input_song,
                        )
                        logger.info(f"Final (swapped {gate_result.swap_stem}): {final}")
                        return final

                    elif cfg.bad_stem_action in ("lego", "lego_then_swap"):
                        # Regenerate bad stem via Lego
                        final = _lego_regenerate_and_mix(
                            bad_stem=gate_result.swap_stem,
                            good_ai_stems=ai_stems,
                            original_stems=original_stems,
                            vocals_path=stems.vocals,
                            caption=timeline.caption,
                            metadata=metadata,
                            output_dir=output_dir,
                            input_song=input_song,
                        )

                        if final and cfg.bad_stem_action == "lego_then_swap":
                            # Re-check: run quality gate on the Lego output
                            lego_stem = Path(output_dir) / "lego_output"
                            lego_files = sorted(lego_stem.glob("*.flac"), key=lambda f: f.stat().st_mtime, reverse=True)
                            if lego_files:
                                # Compare Lego bass against original bass
                                from .stem_quality_gate import _load_audio_mono, _chroma_correlation, _rhythm_correlation
                                orig_audio = _load_audio_mono(original_stems[gate_result.swap_stem])
                                lego_audio = _load_audio_mono(lego_files[0])
                                lego_chroma = _chroma_correlation(orig_audio, lego_audio)
                                lego_rhythm = _rhythm_correlation(orig_audio, lego_audio)
                                logger.info(f"Lego re-check: chroma={lego_chroma:.3f}, rhythm={lego_rhythm:.3f}")

                                if lego_chroma < 0.3 or lego_rhythm < 0.25:
                                    logger.warning("Lego stem still fails quality gate — falling back to swap")
                                    final = remix_with_swap(
                                        vocals_path=stems.vocals,
                                        ai_stems=ai_stems,
                                        original_stems=original_stems,
                                        swap_stem=gate_result.swap_stem,
                                        output_path=f"{output_dir}/final_cover_swap.wav",
                                        original_mix_path=input_song,
                                    )
                                    logger.info(f"Final (swap fallback): {final}")
                                    return final
                                else:
                                    logger.info("Lego stem passes re-check ✅")

                        if final:
                            return final

                        # Fallback to swap if Lego fails entirely
                        logger.warning("Lego failed, falling back to swap")
                        final = remix_with_swap(
                            vocals_path=stems.vocals,
                            ai_stems=ai_stems,
                            original_stems=original_stems,
                            swap_stem=gate_result.swap_stem,
                            output_path=f"{output_dir}/final_cover.wav",
                            original_mix_path=input_song,
                        )
                        return final

    # === STEP 6: Final Mix ===
    logger.info("=" * 50)
    logger.info("STEP 6: Final Mix")
    final = mix_with_effects(
        vocal_path=stems.vocals,
        instrumental_path=instrumental_path,
        output_path=f"{output_dir}/final_cover_daw.wav",
        original_mix_path=input_song,
        per_stem_mix=cfg.per_stem_mix,
    )
    logger.info(f"Final cover: {final}")
    return final


def _lego_regenerate_and_mix(
    bad_stem: str,
    good_ai_stems: dict[str, Path],
    original_stems: dict[str, Path],
    vocals_path: str | Path,
    caption: str,
    metadata: dict,
    output_dir: str,
    input_song: str,
) -> Optional[Path]:
    """Regenerate a bad stem using ACE-Step Lego task.

    Combines the good AI stems as context audio, then generates
    the bad stem using xl-base Lego mode.

    Args:
        bad_stem: Name of stem to regenerate ("drums", "bass", "other").
        good_ai_stems: Dict of AI stem paths.
        original_stems: Dict of original stem paths.
        vocals_path: Path to vocal stem.
        caption: Style caption.
        metadata: BPM/key/timesig dict.
        output_dir: Output directory.
        input_song: Original song path for loudness reference.

    Returns:
        Path to final mix, or None if Lego fails.
    """
    import tempfile
    import soundfile as sf
    import numpy as np
    import toml

    output_dir = Path(output_dir)

    # Combine good stems into context audio
    good_stems_audio = []
    sr = None
    for name, path in good_ai_stems.items():
        if name == bad_stem:
            continue
        if path and Path(path).exists():
            data, stem_sr = sf.read(str(path))
            sr = stem_sr
            if data.ndim == 1:
                data = np.stack([data, data], axis=-1)
            good_stems_audio.append(data)

    if not good_stems_audio or sr is None:
        logger.warning("No good stems to use as Lego context")
        return None

    # Mix good stems together as context
    min_len = min(len(a) for a in good_stems_audio)
    context = np.zeros((min_len, 2), dtype=np.float32)
    for a in good_stems_audio:
        context += a[:min_len].astype(np.float32)

    # Normalize context
    peak = np.max(np.abs(context))
    if peak > 0:
        context = context / peak * 0.8

    # Save context audio
    context_path = output_dir / "lego_context.wav"
    sf.write(str(context_path), context, sr)

    # Build Lego config
    track_name = bad_stem if bad_stem != "other" else "guitar"

    # Use the ORIGINAL bad stem as src_audio — gives Lego the correct
    # notes/rhythm as structural reference while generating new timbre
    original_bad_stem = original_stems.get(bad_stem)
    src_audio_for_lego = str(context_path)
    if original_bad_stem and Path(original_bad_stem).exists():
        # Use original bad stem directly as src_audio (strongest guidance)
        # The good AI stems are the context that Lego "hears"
        src_audio_for_lego = str(original_bad_stem)
        logger.info(f"Lego: using original {bad_stem} directly as src_audio (high cover_strength)")

    lego_config = {
        "task_type": "lego",
        "project_root": str(Path(__file__).parent.parent.parent),
        "checkpoint_dir": str(Path(__file__).parent.parent.parent / "checkpoints"),
        "src_audio": src_audio_for_lego,
        "caption": caption,
        "config_path": "acestep-v15-xl-base",
        "inference_steps": 50,
        "guidance_scale": 7.0,
        "shift": 3.0,
        "batch_size": 1,
        "audio_cover_strength": 0.95,
        "instrumental": True,
        "thinking": False,
        "use_cot_caption": False,
        "use_cot_metas": False,
        "infer_method": "ode",
        "use_random_seed": True,
        "seed": -1,
        "audio_format": "flac",
        "save_dir": str(output_dir / "lego_output"),
        "lego_track": track_name,
        "repainting_start": 0,
        "repainting_end": -1,
    }
    if metadata.get("bpm"):
        lego_config["bpm"] = metadata["bpm"]
    if metadata.get("keyscale"):
        lego_config["keyscale"] = metadata["keyscale"]

    config_path = output_dir / "lego_config.toml"
    config_path.write_text(toml.dumps(lego_config))

    # Run Lego generation
    logger.info(f"Lego: regenerating '{bad_stem}' track (xl-base)...")
    (output_dir / "lego_output").mkdir(exist_ok=True)

    result = subprocess.run(
        [sys.executable, "cli.py", "-c", str(config_path)],
        capture_output=True, text=True, timeout=600,
        cwd=str(Path(__file__).parent.parent.parent),
    )

    # Find Lego output
    lego_output_dir = output_dir / "lego_output"
    lego_files = sorted(
        list(lego_output_dir.glob("*.flac")) + list(Path("output").glob("*.flac")),
        key=lambda f: f.stat().st_mtime,
        reverse=True,
    )

    if not lego_files:
        logger.warning("Lego generation produced no output")
        return None

    lego_stem_path = lego_files[0]
    logger.info(f"Lego generated: {lego_stem_path}")

    # Mix: good AI stems + Lego stem + vocals
    from .stem_quality_gate import remix_with_swap

    # Replace the bad stem with the Lego output
    final_ai_stems = dict(good_ai_stems)
    final_ai_stems[bad_stem] = lego_stem_path

    final = remix_with_swap(
        vocals_path=vocals_path,
        ai_stems=final_ai_stems,
        original_stems=original_stems,
        swap_stem="__none__",  # Don't swap anything — use all AI stems including Lego
        output_path=str(output_dir / "final_cover_lego.wav"),
        original_mix_path=input_song,
    )
    return final


def _hints_regenerate_and_mix(
    bad_stem: str,
    ai_stems: dict[str, Path],
    original_stems: dict[str, Path],
    vocals_path: str | Path,
    full_mix_path: str | Path,
    caption: str,
    metadata: dict,
    output_dir: str,
    input_song: str,
    dit_model: str = "acestep-v15-xl-sft",
    hints_strength: float = 0.6,
) -> Optional[Path]:
    """Regenerate a bad stem using semantic hints from the full mix.

    Generates a new version of just the bad stem's frequency range
    using semantic hints (correct chords) while keeping the good AI stems.

    The hints-regenerated stem will have correct chords but may sound
    closer to the original — acceptable since it's just one instrument.

    Args:
        bad_stem: Name of stem to fix ("drums", "bass", "other").
        ai_stems: Dict of AI stem paths (from Demucs of generated instrumental).
        original_stems: Dict of original stem paths.
        vocals_path: Path to vocal stem.
        full_mix_path: Path to original full mix (for hint extraction).
        caption: Style caption.
        metadata: BPM/key/timesig dict.
        output_dir: Output directory.
        input_song: Original song path for loudness reference.
        dit_model: DiT model name.
        hints_strength: How much of the hints to preserve (1.0=full, 0.5=chords only).

    Returns:
        Path to final mix, or None if failed.
    """
    from .generate_semantic import run_semantic_cover

    output_dir = Path(output_dir)

    # Get the original bad stem as source for the hints regeneration
    original_bad_stem = original_stems.get(bad_stem)
    if not original_bad_stem or not Path(original_bad_stem).exists():
        logger.warning(f"No original {bad_stem} stem for hints regeneration")
        return None

    # Generate a new version using semantic hints
    # This will have correct chords (from hints) even if timbre is similar
    logger.info(f"Regenerating '{bad_stem}' using semantic hints...")
    hints_output = run_semantic_cover(
        src_audio=str(original_bad_stem),
        full_mix_audio=str(full_mix_path),
        caption=caption,
        lyrics="[Instrumental]",
        output_dir=str(output_dir / "hints_regen"),
        bpm=metadata.get("bpm"),
        keyscale=metadata.get("keyscale"),
        timesignature=metadata.get("timesignature", "4/4"),
        cover_noise_strength=0.3,
        guidance_scale=7.0,
        dit_model=dit_model,
        hints_strength=hints_strength,
    )

    if not hints_output:
        logger.warning("Hints regeneration failed")
        return None

    # Demucs the hints output to get just the bad stem
    import os
    env = os.environ.copy()
    env["TORCHAUDIO_BACKEND"] = "soundfile"
    hints_stem_dir = output_dir / "hints_stems"
    hints_stem_dir.mkdir(exist_ok=True)

    subprocess.run(
        [sys.executable, "-m", "demucs", "-n", "htdemucs_ft",
         "--out", str(hints_stem_dir), "--mp3", str(hints_output)],
        capture_output=True, text=True, timeout=300, env=env,
    )

    # Find the regenerated bad stem
    hints_demucs_dir = hints_stem_dir / "htdemucs_ft"
    if not hints_demucs_dir.exists():
        logger.warning("Demucs of hints output failed")
        return None

    hints_dirs = list(hints_demucs_dir.iterdir())
    if not hints_dirs:
        return None

    regen_stem_path = None
    for ext in [".mp3", ".wav"]:
        p = hints_dirs[0] / f"{bad_stem}{ext}"
        if p.exists():
            regen_stem_path = p
            break

    if not regen_stem_path:
        logger.warning(f"Could not find {bad_stem} in hints Demucs output")
        return None

    logger.info(f"Hints-regenerated {bad_stem}: {regen_stem_path}")

    # Mix: good AI stems + hints-regenerated bad stem + vocals using DAW effects
    from .mix_daw import mix_with_effects

    # Combine all AI stems with the regenerated bad stem into one instrumental
    import numpy as np
    import soundfile as sf_io
    final_ai_stems = dict(ai_stems)
    final_ai_stems[bad_stem] = regen_stem_path

    # Sum stems into a single instrumental file
    stem_arrays = []
    stem_sr = None
    for name in ["drums", "bass", "other"]:
        path = final_ai_stems.get(name)
        if path and Path(path).exists():
            data, sr = sf_io.read(str(path))
            stem_sr = sr
            if data.ndim == 1:
                data = np.stack([data, data], axis=-1)
            stem_arrays.append(data.astype(np.float32))

    if not stem_arrays or stem_sr is None:
        logger.warning("No stems to mix after hints regeneration")
        return None

    min_len = min(len(a) for a in stem_arrays)
    combined_instrumental = np.zeros((min_len, 2), dtype=np.float32)
    for a in stem_arrays:
        combined_instrumental += a[:min_len]

    # Save combined instrumental
    combined_path = output_dir / "hints_combined_instrumental.wav"
    sf_io.write(str(combined_path), combined_instrumental, stem_sr)

    # Mix with DAW effects (per_stem_mix=False for clean output)
    final = mix_with_effects(
        vocal_path=vocals_path,
        instrumental_path=combined_path,
        output_path=str(output_dir / "final_cover_daw.wav"),
        original_mix_path=input_song,
        per_stem_mix=False,
    )
    logger.info(f"Final (hints-regenerated {bad_stem}, DAW mix): {final}")
    return final
