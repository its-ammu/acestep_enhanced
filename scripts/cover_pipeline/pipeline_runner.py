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
    guidance_scale: float = 15.0
    inference_steps: int = 65
    shift: float = 6.0
    batch_size: int = 1

    # Generation mode: "semantic" (bass hints), "remix_blend" (LM codes + bass blend),
    # "lego_stems" (replace melodic only), "complete_vocals" (vocals→full backing),
    # "complete_remix" (vocal+bass → complete task, pro remixer approach),
    # "cover_genre" (cover task + genre caption + native repaint),
    # "audio_codes" (5Hz loose guide), "text2music_free" (pure text), or "cli" (subprocess)
    generation_mode: str = "semantic"

    # complete_remix mode settings
    bass_mix_db: float = -6.0  # Bass level in source mix (-3=strong, -6=balanced, -12=loose)
    refine_caption_lm: bool = True  # Use LM to enhance caption (creativity lever)
    include_vocal_in_source: bool = True  # Include vocal in source mix (False = bass only)

    # remix_blend mode settings
    blend_alpha: float = 0.4  # 0.0=all bass chords, 1.0=all LM creativity
    lm_model: str = "acestep-5Hz-lm-4B"  # LM for creative code planning
    lm_temperature: float = 0.7  # LM sampling temp (0.7=conservative, 0.85=creative)
    apply_effects: bool = True  # Post-processing production effects

    # audio_codes mode settings (5Hz loose chord guidance via text2music)
    codes_cover_strength: float = 0.3  # 0.1-0.5 for loose interpretation

    # Caption
    refine_caption: bool = True

    # Re-arrangement (v25): generate different instruments for the cover
    rearrange: bool = True

    # Quality gate
    quality_gate: bool = True
    bad_stem_action: str = "hints"  # "hints", "swap", or "keep"

    # Mix
    per_stem_mix: bool = False

    # Hints degradation (for semantic route)
    hints_strength: float = 1.0  # 1.0=full (correct chords), 0.5=chords only, 0.0=no hints

    # LoRA concept slider (name or path)
    lora_path: Optional[str] = "digital-acoustic"
    lora_scale: float = 0.7

    # Skip safe generation (rely on QC repaint instead)
    skip_safe_generation: bool = False

    # QC retry cap (per section)
    max_repaint_attempts: int = 3

    # Hint blending (v26): blend original hints with MIDI piano hints
    use_hint_blending: bool = False
    blend_factor: float = 0.5  # 0.0=original timbre, 1.0=neutral timbre

    # ScragVAE (improved audio fidelity)
    use_scrag_vae: bool = True

    # Timbre reference: generate a reference clip of target instruments
    # to feed the timbre encoder (decouples timbre from semantic hints)
    use_timbre_reference: bool = False
    timbre_ref_duration: float = 30.0  # seconds of reference to generate


def run_pipeline(cfg: PipelineConfig) -> Optional[Path]:
    """Run the full cover pipeline.

    Args:
        cfg: Pipeline configuration.

    Returns:
        Path to final cover file, or None if failed.
    """
    import time as _time

    from .deps import ensure_dependencies
    from .stem_separation import separate_stems
    from .audio_analysis import analyze_bpm_and_structure, analyze_key
    from .structure_timeline import generate_structure_timeline
    from .mix_daw import mix_with_effects
    from .stem_quality_gate import evaluate_stems, remix_with_swap

    ensure_dependencies(include_optional=True)

    _timings = {}
    _t_pipeline_start = _time.time()

    input_song = cfg.input_song
    output_dir = cfg.output_dir
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # === STEP 1: Stem Separation ===
    logger.info("=" * 50)
    logger.info("STEP 1: Stem Separation")
    _t0 = _time.time()
    stems = separate_stems(input_song, f"{output_dir}/stems")
    _timings["stem_separation"] = _time.time() - _t0

    # === STEP 2: Metadata ===
    logger.info("=" * 50)
    logger.info("STEP 2: Metadata Detection")
    _t0 = _time.time()
    bpm_result = analyze_bpm_and_structure(input_song)
    key = analyze_key(input_song, bass_stem_path=stems.bass)
    metadata = {
        "bpm": int(round(bpm_result["bpm"])),
        "keyscale": key,
        "timesignature": "4/4",
    }
    _timings["metadata"] = _time.time() - _t0
    logger.info(f"BPM: {metadata['bpm']}, Key: {metadata['keyscale']}")

    # === STEP 3: Structure Timeline ===
    logger.info("=" * 50)
    logger.info("STEP 3: Structure Timeline")
    _t0 = _time.time()
    timeline = generate_structure_timeline(
        audio_path=input_song,  # Full mix for SongFormer (needs vocals for section detection)
        metadata=metadata,
        qwen_model=cfg.qwen_model,
        stem_paths={
            "drums": stems.drums,
            "bass": stems.bass,
            "other": stems.other,
        },
        rearrange=cfg.rearrange,
    )
    _timings["structure_timeline"] = _time.time() - _t0
    logger.info(f"Caption: {timeline.caption[:150]}")

    # === STEP 4: Cover Generation ===
    logger.info("=" * 50)
    logger.info("STEP 4: Cover Generation")

    instrumental_path = None
    _qc_section_log = []

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
        hints_stem_path = stems.bass

        if cfg.use_hint_blending:
            hints = _run_hint_blending(handler, hints_stem_path, output_dir, cfg)
        else:
            # Hardcode bass stem for hints — proven to give best chord accuracy.
            # Auto-selector experiments showed "other" stem causes chord drift
            # even when it scores higher on simplicity metrics.
            logger.info(f"Hints source: bass ({hints_stem_path})")
            hints = extract_semantic_hints(handler, str(hints_stem_path))

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
        import time as _time

        # Timbre reference: generate a clip of target instruments to feed
        # the timbre encoder. This decouples timbre from the semantic hints.
        refer_audios = None
        if cfg.use_timbre_reference and cfg.rearrange:
            from .timbre_reference import generate_timbre_reference

            logger.info("Generating timbre reference clip...")
            _t_ref_start = _time.time()
            ref_audio = generate_timbre_reference(
                handler=handler,
                caption=timeline.caption,
                duration=duration,
                lyrics=timeline.lyrics,
                bpm=metadata.get("bpm"),
                keyscale=metadata.get("keyscale"),
                guidance_scale=cfg.guidance_scale,
                inference_steps=cfg.inference_steps,
                shift=cfg.shift,
            )
            if ref_audio is not None:
                refer_audios = [[ref_audio]]
                _timings["timbre_reference"] = _time.time() - _t_ref_start
                logger.info(
                    f"⏱️  Timbre reference: {_timings['timbre_reference']:.1f}s"
                )
            else:
                logger.warning("Timbre reference failed — using caption-only timbre")

        logger.info(f"Generating creative version (cns={cfg.cover_noise_strength})...")
        _t_creative_start = _time.time()
        creative_result = handler.service_generate(
            captions=timeline.caption,
            lyrics=timeline.lyrics,
            target_wavs=target_wavs,
            refer_audios=refer_audios,
            metas=metas,
            audio_cover_strength=cfg.audio_cover_strength,
            guidance_scale=cfg.guidance_scale,
            infer_steps=cfg.inference_steps,
            shift=cfg.shift,
            cover_noise_strength=cfg.cover_noise_strength,
            task_type="cover",
            infer_method="ode",
        )
        _t_creative_end = _time.time()
        _timings["creative_generation"] = _t_creative_end - _t_creative_start
        logger.info(f"⏱️  Creative generation: {_timings['creative_generation']:.1f}s")

        # Generate safe version (higher cns for bass consistency)
        safe_result = None
        if not cfg.skip_safe_generation:
            logger.info("Generating safe version (cns=0.3)...")
            _t_safe_start = _time.time()
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
            _t_safe_end = _time.time()
            _timings["safe_generation"] = _t_safe_end - _t_safe_start
            logger.info(f"⏱️  Safe generation: {_timings['safe_generation']:.1f}s")
        else:
            logger.info("⏭️  Skipping safe generation (skip_safe_generation=True)")

        # Decode versions
        gen_dir = Path(f"{output_dir}/generated")
        gen_dir.mkdir(parents=True, exist_ok=True)
        creative_path = None
        safe_path = None

        versions = [("creative", creative_result)]
        if safe_result is not None:
            versions.append(("safe", safe_result))

        for label, result in versions:
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
    elif cfg.generation_mode == "audio_codes":
        # Audio codes mode: extract 5Hz indices from bass stem, use as loose
        # chord guide via text2music task. The model auto-switches to cover
        # internally when audio_codes are provided, but uses the coarser 5Hz
        # representation (chord-level) instead of 25Hz hints (timbre-level).
        from .extract_audio_codes import extract_audio_codes_string
        from acestep.handler import AceStepHandler
        from .generate_semantic import _load_scrag_vae, ensure_slider_lora
        import torch
        import soundfile as sf_gen
        import numpy as np
        import librosa as lr
        import time as _time

        logger.info("Using audio_codes mode (5Hz loose chord guidance)")

        handler = AceStepHandler()
        handler.initialize_service(project_root=".", config_path=cfg.dit_model)

        if cfg.use_scrag_vae:
            _load_scrag_vae(handler)

        # Extract 5Hz codes from bass stem (chord roots without timbre)
        _t_codes_start = _time.time()
        codes_string = extract_audio_codes_string(handler, str(stems.bass))
        _timings["codes_extraction"] = _time.time() - _t_codes_start
        logger.info(f"⏱️  Codes extraction: {_timings['codes_extraction']:.1f}s")

        # Load LoRA
        if cfg.lora_path:
            lora_dir = Path(cfg.lora_path)
            if not lora_dir.exists() and "/" not in str(cfg.lora_path):
                lora_dir = ensure_slider_lora(str(cfg.lora_path))
            if lora_dir and lora_dir.exists():
                handler.add_lora(str(lora_dir))
                handler.set_lora_scale(cfg.lora_scale)
                handler.use_lora = True
                logger.info(f"LoRA: {lora_dir} at scale {cfg.lora_scale}")

        # Prepare source audio (instrumental) for duration/metas
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

        # Generate with audio_codes — the handler auto-switches text2music→cover
        # when codes are provided, but uses 5Hz quantizer path (not 25Hz hints)
        logger.info(
            f"Generating with audio_codes (strength={cfg.codes_cover_strength}, "
            f"guidance={cfg.guidance_scale})..."
        )
        _t_gen_start = _time.time()
        result = handler.service_generate(
            captions=timeline.caption,
            lyrics=timeline.lyrics,
            target_wavs=target_wavs,
            metas=metas,
            audio_cover_strength=cfg.codes_cover_strength,
            guidance_scale=cfg.guidance_scale,
            infer_steps=cfg.inference_steps,
            shift=cfg.shift,
            cover_noise_strength=0.0,  # Start from noise for max creativity
            audio_code_hints=[codes_string],
            task_type="text2music",
            infer_method="ode",
        )
        _timings["codes_generation"] = _time.time() - _t_gen_start
        logger.info(f"⏱️  Codes generation: {_timings['codes_generation']:.1f}s")

        # Decode output
        gen_dir = Path(f"{output_dir}/generated")
        gen_dir.mkdir(parents=True, exist_ok=True)

        if "target_latents" in result:
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
            out_path = gen_dir / "audio_codes_cover.flac"
            sf_gen.write(str(out_path), audio_np.T, 48000)
            instrumental_path = out_path
            logger.info(f"Audio codes output: {out_path}")
        else:
            logger.error("Audio codes generation failed — no latents returned")
            instrumental_path = None

        # Cleanup
        del handler
        import gc as gc_mod
        gc_mod.collect()
        torch.cuda.empty_cache()

    elif cfg.generation_mode == "remix_blend":
        # Remix blend: LM plans creative codes, blend with bass hints, render.
        # This produces genre-shifted remixes with musical complexity.
        from .remix_blend import (
            generate_lm_codes, blend_codes_with_bass,
            render_with_blended_hints, apply_production_effects,
        )
        from .generate_semantic import _load_scrag_vae, ensure_slider_lora
        from acestep.handler import AceStepHandler
        import torch
        import soundfile as sf_gen
        import numpy as np
        import librosa as lr
        import time as _time

        logger.info("Using remix_blend mode (LM creativity + bass chords)")

        # Phase 1: LM generates creative codes
        _t_lm_start = _time.time()
        lm_codes = generate_lm_codes(
            caption=timeline.caption,
            lyrics=timeline.lyrics,
            duration=sf_gen.info(str(stems.instrumental)).duration,
            bpm=metadata.get("bpm", 112),
            keyscale=metadata.get("keyscale", ""),
            lm_model=cfg.lm_model,
            lm_temperature=cfg.lm_temperature,
        )
        _timings["lm_codes"] = _time.time() - _t_lm_start

        if not lm_codes:
            logger.error("LM code generation failed")
            instrumental_path = None
        else:
            # Phase 2: Load DiT, blend, render
            _t_render_start = _time.time()
            handler = AceStepHandler()
            handler.initialize_service(
                project_root=".", config_path=cfg.dit_model
            )

            if cfg.use_scrag_vae:
                _load_scrag_vae(handler)

            # Load LoRA
            if cfg.lora_path:
                lora_dir = Path(cfg.lora_path)
                if not lora_dir.exists() and "/" not in str(cfg.lora_path):
                    lora_dir = ensure_slider_lora(str(cfg.lora_path))
                if lora_dir and lora_dir.exists():
                    handler.add_lora(str(lora_dir))
                    handler.set_lora_scale(cfg.lora_scale)
                    handler.use_lora = True
                    logger.info(f"LoRA: {lora_dir} at scale {cfg.lora_scale}")

            # Blend LM codes with bass hints
            blended_hints = blend_codes_with_bass(
                handler=handler,
                lm_codes=lm_codes,
                bass_path=str(stems.bass),
                alpha=cfg.blend_alpha,
            )

            # Prepare source audio
            inst_audio, sr_inst = sf_gen.read(str(stems.instrumental))
            if inst_audio.ndim == 1:
                inst_audio = np.stack([inst_audio, inst_audio], axis=-1)
            if sr_inst != 48000:
                inst_audio = lr.resample(
                    inst_audio.T, orig_sr=sr_inst, target_sr=48000
                ).T
            target_wavs = torch.tensor(
                inst_audio.T, dtype=torch.float32
            ).unsqueeze(0)
            duration = len(inst_audio) / 48000
            metas = [{"audio_duration": duration, "time_signature": "4/4"}]
            if metadata.get("bpm"):
                metas[0]["bpm"] = metadata["bpm"]
            if metadata.get("keyscale"):
                metas[0]["keyscale"] = metadata["keyscale"]

            # Render
            audio_np = render_with_blended_hints(
                handler=handler,
                blended_hints=blended_hints,
                target_wavs=target_wavs,
                caption=timeline.caption,
                lyrics=timeline.lyrics,
                metas=metas,
                cover_noise_strength=cfg.cover_noise_strength,
                inference_steps=cfg.inference_steps,
                shift=cfg.shift,
            )
            _timings["render"] = _time.time() - _t_render_start

            if audio_np is not None:
                gen_dir = Path(f"{output_dir}/generated")
                gen_dir.mkdir(parents=True, exist_ok=True)

                # Apply effects if enabled
                if cfg.apply_effects:
                    audio_np = apply_production_effects(audio_np)
                    logger.info("Production effects applied")

                out_path = gen_dir / "remix_blend.flac"
                sf_gen.write(str(out_path), audio_np, 48000)
                instrumental_path = out_path
                logger.info(f"Remix blend output: {out_path}")

                # QC + Hint Repaint for failing sections
                if cfg.max_repaint_attempts > 0:
                    from .post_gen_qc import (
                        analyze_generated_audio, get_failing_sections,
                    )
                    from .hint_repaint import repaint_section_with_hints
                    from .semantic_cover import extract_semantic_hints

                    logger.info("QC: checking sections for chord accuracy...")
                    qc_results = analyze_generated_audio(
                        audio_path=str(out_path),
                        bpm=metadata.get("bpm", 100),
                        key=metadata.get("keyscale", ""),
                        segments=timeline.segments,
                        original_audio_path=str(stems.instrumental),
                    )
                    failing = get_failing_sections(qc_results)

                    if failing:
                        logger.info(
                            f"QC: {len(failing)} sections need repaint"
                        )
                        # Use blended hints at lower alpha for repaint
                        # (more bass = better chords, still some LM character)
                        repaint_alpha = max(0.15, cfg.blend_alpha - 0.2)
                        bass_hints_pure = extract_semantic_hints(
                            handler, str(stems.bass)
                        )
                        # Re-decode LM codes for repaint blend
                        lm_hints_repaint = handler._decode_audio_codes_to_latents(
                            lm_codes
                        )
                        if lm_hints_repaint is not None:
                            max_t_r = max(
                                bass_hints_pure.shape[1],
                                lm_hints_repaint.shape[1],
                            )
                            if bass_hints_pure.shape[1] < max_t_r:
                                bass_hints_pure = torch.cat([
                                    bass_hints_pure,
                                    torch.zeros(1, max_t_r - bass_hints_pure.shape[1], 64, device=bass_hints_pure.device, dtype=bass_hints_pure.dtype),
                                ], dim=1)
                            if lm_hints_repaint.shape[1] < max_t_r:
                                lm_hints_repaint = torch.cat([
                                    lm_hints_repaint,
                                    torch.zeros(1, max_t_r - lm_hints_repaint.shape[1], 64, device=lm_hints_repaint.device, dtype=lm_hints_repaint.dtype),
                                ], dim=1)
                            repaint_hints = (
                                repaint_alpha * lm_hints_repaint
                                + (1.0 - repaint_alpha) * bass_hints_pure
                            )
                        else:
                            repaint_hints = bass_hints_pure

                        logger.info(
                            f"Repaint blend alpha={repaint_alpha:.2f}"
                        )

                        current_path = str(out_path)
                        for section in failing:
                            for attempt in range(cfg.max_repaint_attempts):
                                logger.info(
                                    f"  Repainting [{section.label}] "
                                    f"{section.start_sec:.1f}-"
                                    f"{section.end_sec:.1f}s "
                                    f"(attempt {attempt+1})"
                                )
                                fixed = repaint_section_with_hints(
                                    handler=handler,
                                    creative_audio_path=current_path,
                                    start_sec=section.start_sec,
                                    end_sec=section.end_sec,
                                    hints=repaint_hints,
                                    caption=timeline.caption,
                                    lyrics=timeline.lyrics,
                                    bpm=metadata.get("bpm"),
                                    keyscale=metadata.get("keyscale"),
                                    guidance_scale=1.0,
                                    inference_steps=cfg.inference_steps,
                                    shift=cfg.shift,
                                    cover_noise_strength=0.20,
                                )
                                if fixed is not None:
                                    fixed_path = gen_dir / "remix_blend_fixed.flac"
                                    sf_gen.write(
                                        str(fixed_path), fixed, 48000
                                    )
                                    current_path = str(fixed_path)
                                    instrumental_path = fixed_path
                                    logger.info(f"  ✅ Section repainted")
                                    break
                                else:
                                    logger.warning(
                                        f"  ❌ Repaint attempt {attempt+1} failed"
                                    )
                    else:
                        logger.info("QC: all sections pass ✅")

            else:
                instrumental_path = None

            # Cleanup
            del handler
            import gc as gc_mod
            gc_mod.collect()
            torch.cuda.empty_cache()

    elif cfg.generation_mode == "lego_stems":
        # Lego stem replacement: keep original drums + bass, generate new
        # melodic track using Lego task. The model hears drums+bass as context
        # and generates a complementary track in the requested style.
        from .lego_generate import _mix_stems_to_context, generate_lego_stem
        from .generate_semantic import _load_scrag_vae, ensure_slider_lora
        from acestep.handler import AceStepHandler
        import torch
        import soundfile as sf_gen
        import numpy as np
        import librosa as lr
        import time as _time

        logger.info("Using lego_stems mode (replace melodic, keep drums+bass)")

        handler = AceStepHandler()
        handler.initialize_service(project_root=".", config_path=cfg.dit_model)

        if cfg.use_scrag_vae:
            _load_scrag_vae(handler)

        # Load LoRA
        if cfg.lora_path:
            lora_dir = Path(cfg.lora_path)
            if not lora_dir.exists() and "/" not in str(cfg.lora_path):
                lora_dir = ensure_slider_lora(str(cfg.lora_path))
            if lora_dir and lora_dir.exists():
                handler.add_lora(str(lora_dir))
                handler.set_lora_scale(cfg.lora_scale)
                handler.use_lora = True
                logger.info(f"LoRA: {lora_dir} at scale {cfg.lora_scale}")

        # Mix drums + bass as context for Lego
        _t_lego_start = _time.time()
        context_audio = _mix_stems_to_context(stems.drums, stems.bass)
        duration = len(context_audio) / 48000

        # Generate the melodic/lead track
        lego_caption = timeline.caption if timeline.caption else (
            "expressive synth lead with dynamic range"
        )
        logger.info(f"Lego caption: {lego_caption[:100]}")

        new_melodic = generate_lego_stem(
            handler=handler,
            context_audio=context_audio,
            track_name="synth",
            caption=lego_caption,
            duration=duration,
            segments=timeline.segments,
            lyrics=timeline.lyrics,
            global_caption=timeline.original_caption or lego_caption,
            bpm=metadata.get("bpm"),
            keyscale=metadata.get("keyscale"),
            guidance_scale=cfg.guidance_scale,
            inference_steps=cfg.inference_steps,
            shift=cfg.shift,
        )
        _timings["lego_generation"] = _time.time() - _t_lego_start
        logger.info(f"⏱️  Lego generation: {_timings['lego_generation']:.1f}s")

        if new_melodic is not None:
            # Mix: original drums + original bass + new melodic
            drums_audio, _ = lr.load(str(stems.drums), sr=48000, mono=False)
            bass_audio, _ = lr.load(str(stems.bass), sr=48000, mono=False)

            if drums_audio.ndim == 1:
                drums_audio = np.stack([drums_audio, drums_audio])
            if bass_audio.ndim == 1:
                bass_audio = np.stack([bass_audio, bass_audio])

            new_melodic_t = new_melodic.T  # (2, samples)

            # Match lengths
            min_len = min(
                drums_audio.shape[1], bass_audio.shape[1], new_melodic_t.shape[1]
            )
            drums_audio = drums_audio[:, :min_len]
            bass_audio = bass_audio[:, :min_len]
            new_melodic_t = new_melodic_t[:, :min_len]

            # Mix with relative levels (drums loud, bass mid, melodic mid)
            mixed = drums_audio * 0.9 + bass_audio * 0.7 + new_melodic_t * 0.8
            peak = np.max(np.abs(mixed))
            if peak > 0:
                mixed = mixed / peak * 0.891

            gen_dir = Path(f"{output_dir}/generated")
            gen_dir.mkdir(parents=True, exist_ok=True)
            out_path = gen_dir / "lego_remix.flac"
            sf_gen.write(str(out_path), mixed.T, 48000)
            instrumental_path = out_path
            logger.info(f"Lego remix: {out_path}")
        else:
            logger.error("Lego generation failed")
            instrumental_path = None

        # Cleanup
        del handler
        import gc as gc_mod
        gc_mod.collect()
        torch.cuda.empty_cache()

    elif cfg.generation_mode == "complete_vocals":
        # Complete task: feed vocals, let model generate full backing freely.
        # No hints, no cover mode — maximum creative freedom.
        # The model hears vocals and generates complementary accompaniment.
        from .generate_semantic import _load_scrag_vae, ensure_slider_lora
        from acestep.handler import AceStepHandler
        import torch
        import soundfile as sf_gen
        import numpy as np
        import time as _time

        logger.info("Using complete_vocals mode (vocals → full backing, no hints)")

        handler = AceStepHandler()
        handler.initialize_service(project_root=".", config_path=cfg.dit_model)

        if cfg.use_scrag_vae:
            _load_scrag_vae(handler)

        # Load LoRA
        if cfg.lora_path:
            lora_dir = Path(cfg.lora_path)
            if not lora_dir.exists() and "/" not in str(cfg.lora_path):
                lora_dir = ensure_slider_lora(str(cfg.lora_path))
            if lora_dir and lora_dir.exists():
                handler.add_lora(str(lora_dir))
                handler.set_lora_scale(cfg.lora_scale)
                handler.use_lora = True
                logger.info(f"LoRA: {lora_dir} at scale {cfg.lora_scale}")

        # Load vocals as source audio
        vocal_audio, sr_voc = sf_gen.read(str(stems.vocals))
        if vocal_audio.ndim == 1:
            vocal_audio = np.stack([vocal_audio, vocal_audio], axis=-1)
        if sr_voc != 48000:
            import librosa as lr
            vocal_audio = lr.resample(
                vocal_audio.T, orig_sr=sr_voc, target_sr=48000
            ).T
            sr_voc = 48000
        target_wavs = torch.tensor(
            vocal_audio.T, dtype=torch.float32
        ).unsqueeze(0)
        duration = len(vocal_audio) / sr_voc

        metas = [{"audio_duration": duration, "time_signature": "4/4"}]
        if metadata.get("bpm"):
            metas[0]["bpm"] = metadata["bpm"]
        if metadata.get("keyscale"):
            metas[0]["keyscale"] = metadata["keyscale"]

        # Build instruction for complete task
        instruction = "Complete the input track with drums, bass, synth:"

        logger.info(
            f"Generating complete backing ({duration:.0f}s, "
            f"bpm={metadata.get('bpm')}, key={metadata.get('keyscale')})"
        )
        logger.info(f"Caption: {timeline.caption[:100]}")
        logger.info(f"Lyrics: {timeline.lyrics[:200]}")

        _t_gen_start = _time.time()
        result = handler.service_generate(
            captions=timeline.caption,
            lyrics=timeline.lyrics,
            target_wavs=target_wavs,
            metas=metas,
            instructions=[instruction],
            guidance_scale=cfg.guidance_scale,
            infer_steps=cfg.inference_steps,
            shift=cfg.shift,
            task_type="complete",
            infer_method="ode",
        )
        _timings["complete_generation"] = _time.time() - _t_gen_start
        logger.info(f"⏱️  Complete generation: {_timings['complete_generation']:.1f}s")

        # Decode output
        gen_dir = Path(f"{output_dir}/generated")
        gen_dir.mkdir(parents=True, exist_ok=True)

        if "target_latents" in result:
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
            out_path = gen_dir / "complete_backing.flac"
            sf_gen.write(str(out_path), audio_np.T, 48000)
            instrumental_path = out_path
            logger.info(f"Complete backing: {out_path}")
        else:
            logger.error("Complete generation failed — no latents returned")
            instrumental_path = None

        # Cleanup
        del handler
        import gc as gc_mod
        gc_mod.collect()
        torch.cuda.empty_cache()

    elif cfg.generation_mode == "text2music_free":
        # Pure text2music: no cover conditioning, no hints, no source audio.
        # The model generates freely from caption + BPM + key + lyrics.
        # The rearranged caption specifies different instruments — the model
        # actually uses them because there's no timbre-locked hints overriding.
        from acestep.handler import AceStepHandler
        from .generate_semantic import _load_scrag_vae, ensure_slider_lora
        import torch
        import soundfile as sf_gen
        import numpy as np
        import librosa as lr
        import time as _time

        logger.info("Using text2music_free mode (no cover conditioning)")

        handler = AceStepHandler()
        handler.initialize_service(project_root=".", config_path=cfg.dit_model)

        if cfg.use_scrag_vae:
            _load_scrag_vae(handler)

        # Load LoRA
        if cfg.lora_path:
            lora_dir = Path(cfg.lora_path)
            if not lora_dir.exists() and "/" not in str(cfg.lora_path):
                lora_dir = ensure_slider_lora(str(cfg.lora_path))
            if lora_dir and lora_dir.exists():
                handler.add_lora(str(lora_dir))
                handler.set_lora_scale(cfg.lora_scale)
                handler.use_lora = True
                logger.info(f"LoRA: {lora_dir} at scale {cfg.lora_scale}")

        # Get duration from instrumental stem
        inst_audio, sr_inst = sf_gen.read(str(stems.instrumental))
        duration = len(inst_audio) / sr_inst

        # Build metadata — BPM + key anchor the generation harmonically
        metas = [{"audio_duration": duration, "time_signature": "4/4"}]
        if metadata.get("bpm"):
            metas[0]["bpm"] = metadata["bpm"]
        if metadata.get("keyscale"):
            metas[0]["keyscale"] = metadata["keyscale"]

        # Generate — pure text2music, no target_wavs, no cover mode
        logger.info(
            f"Generating text2music (duration={duration:.1f}s, "
            f"bpm={metadata.get('bpm')}, key={metadata.get('keyscale')}, "
            f"guidance={cfg.guidance_scale})..."
        )
        _t_gen_start = _time.time()
        result = handler.service_generate(
            captions=timeline.caption,
            lyrics=timeline.lyrics,
            metas=metas,
            guidance_scale=cfg.guidance_scale,
            infer_steps=cfg.inference_steps,
            shift=cfg.shift,
            task_type="text2music",
            infer_method="ode",
        )
        _timings["text2music_generation"] = _time.time() - _t_gen_start
        logger.info(f"⏱️  Text2music generation: {_timings['text2music_generation']:.1f}s")

        # Decode output
        gen_dir = Path(f"{output_dir}/generated")
        gen_dir.mkdir(parents=True, exist_ok=True)

        if "target_latents" in result:
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
            out_path = gen_dir / "text2music_free.flac"
            sf_gen.write(str(out_path), audio_np.T, 48000)
            instrumental_path = out_path
            logger.info(f"Text2music output: {out_path}")
        else:
            logger.error("Text2music generation failed — no latents returned")
            instrumental_path = None

        # Cleanup
        del handler
        import gc as gc_mod
        gc_mod.collect()
        torch.cuda.empty_cache()

    elif cfg.generation_mode == "cover_genre":
        # Cover-mode genre shift: cover task + genre caption + CFG.
        # Cover mode preserves structure, CFG pushes toward genre caption.
        # No hints — let cover mode handle chord preservation via audio_cover_strength.
        from .cover_genre import (
            refine_caption_for_genre, generate_cover_genre,
            repaint_failing_sections,
        )
        from .generate_semantic import _load_scrag_vae, ensure_slider_lora
        from acestep.handler import AceStepHandler
        import torch
        import soundfile as sf_gen
        import numpy as np
        import time as _time

        logger.info("Using cover_genre mode (cover + genre caption + CFG)")

        # Phase 1: LM enhances genre caption
        refined_caption = timeline.caption
        if cfg.refine_caption_lm:
            _t_lm_start = _time.time()
            refined_caption = refine_caption_for_genre(
                caption=timeline.caption,
                lyrics=timeline.lyrics,
                bpm=metadata.get("bpm", 112),
                keyscale=metadata.get("keyscale", ""),
                duration=sf_gen.info(str(stems.instrumental)).duration,
                lm_model=cfg.lm_model,
                lm_temperature=cfg.lm_temperature,
            )
            _timings["lm_caption"] = _time.time() - _t_lm_start
            logger.info(f"⏱️  LM caption: {_timings['lm_caption']:.1f}s")

        # Phase 2: Load DiT
        _t_render_start = _time.time()
        handler = AceStepHandler()
        handler.initialize_service(project_root=".", config_path=cfg.dit_model)

        if cfg.use_scrag_vae:
            _load_scrag_vae(handler)

        # Load LoRA
        if cfg.lora_path:
            lora_dir = Path(cfg.lora_path)
            if not lora_dir.exists() and "/" not in str(cfg.lora_path):
                lora_dir = ensure_slider_lora(str(cfg.lora_path))
            if lora_dir and lora_dir.exists():
                handler.add_lora(str(lora_dir))
                handler.set_lora_scale(cfg.lora_scale)
                handler.use_lora = True
                logger.info(f"LoRA: {lora_dir} at scale {cfg.lora_scale}")

        # Source audio: full instrumental (structural skeleton)
        source_path = str(stems.instrumental)
        logger.info(f"Source: {source_path} (full instrumental = structural skeleton)")

        # Timbre reference (optional)
        refer_audios = None
        if cfg.use_timbre_reference:
            from .timbre_reference import generate_timbre_reference

            logger.info("Generating timbre reference clip...")
            _t_ref_start = _time.time()
            ref_audio = generate_timbre_reference(
                handler=handler,
                caption=refined_caption,
                duration=sf_gen.info(str(stems.instrumental)).duration,
                lyrics=timeline.lyrics,
                bpm=metadata.get("bpm"),
                keyscale=metadata.get("keyscale"),
                guidance_scale=cfg.guidance_scale,
                inference_steps=cfg.inference_steps,
                shift=cfg.shift,
            )
            if ref_audio is not None:
                refer_audios = [[ref_audio]]
                _timings["timbre_reference"] = _time.time() - _t_ref_start
                logger.info(
                    f"⏱️  Timbre reference: {_timings['timbre_reference']:.1f}s"
                )
            else:
                logger.warning("Timbre reference failed — caption-only timbre")

        # Phase 3: Generate cover (no hints — clean cover mode)
        audio_np = generate_cover_genre(
            handler=handler,
            source_audio_path=source_path,
            caption=refined_caption,
            lyrics=timeline.lyrics,
            hints=None,
            bpm=metadata.get("bpm"),
            keyscale=metadata.get("keyscale"),
            audio_cover_strength=cfg.audio_cover_strength,
            cover_noise_strength=cfg.cover_noise_strength,
            guidance_scale=cfg.guidance_scale,
            inference_steps=cfg.inference_steps,
            shift=cfg.shift,
            refer_audios=refer_audios,
        )
        _timings["cover_render"] = _time.time() - _t_render_start
        logger.info(f"⏱️  Cover render: {_timings['cover_render']:.1f}s")

        if audio_np is not None:
            gen_dir = Path(f"{output_dir}/generated")
            gen_dir.mkdir(parents=True, exist_ok=True)
            out_path = gen_dir / "cover_genre.flac"
            sf_gen.write(str(out_path), audio_np, 48000)
            instrumental_path = out_path
            logger.info(f"Cover genre output: {out_path}")

            # Phase 3: QC + native repaint for failing sections
            if cfg.max_repaint_attempts > 0:
                from .post_gen_qc import (
                    analyze_generated_audio, get_failing_sections,
                )

                logger.info("QC: checking sections...")
                qc_results = analyze_generated_audio(
                    audio_path=str(out_path),
                    bpm=metadata.get("bpm", 100),
                    key=metadata.get("keyscale", ""),
                    segments=timeline.segments,
                    original_audio_path=str(stems.instrumental),
                )
                failing = get_failing_sections(qc_results)

                if failing:
                    logger.info(f"QC: {len(failing)} sections need repaint")
                    fixed_path = repaint_failing_sections(
                        handler=handler,
                        audio_path=str(out_path),
                        failing_sections=failing,
                        caption=refined_caption,
                        lyrics=timeline.lyrics,
                        source_audio_path=source_path,
                        hints=None,
                        bpm=metadata.get("bpm"),
                        keyscale=metadata.get("keyscale"),
                        audio_cover_strength=min(cfg.audio_cover_strength + 0.2, 0.9),
                        cover_noise_strength=cfg.cover_noise_strength,
                        guidance_scale=cfg.guidance_scale,
                        inference_steps=cfg.inference_steps,
                        shift=cfg.shift,
                        max_attempts=cfg.max_repaint_attempts,
                    )
                    if fixed_path and fixed_path != str(out_path):
                        instrumental_path = Path(fixed_path)
                        logger.info(f"Repainted output: {fixed_path}")
                else:
                    logger.info("QC: all sections pass ✅")
        else:
            instrumental_path = None

        # Cleanup
        del handler
        import gc as gc_mod
        gc_mod.collect()
        torch.cuda.empty_cache()

    elif cfg.generation_mode == "complete_remix":
        # Complete-task remix: vocal+bass → complete task generates new instruments.
        # Pro remixer approach: model hears vocal+bass, generates complementary parts
        # in the target genre. Chord accuracy is natural (model harmonizes with what
        # it hears). Creativity comes from the LM-refined caption.
        from .complete_remix import (
            refine_caption_with_lm, mix_vocal_and_bass,
            build_complete_instruction, generate_complete_remix,
        )
        from .generate_semantic import _load_scrag_vae, ensure_slider_lora
        from .rearrangement import parse_instruments_from_caption
        from acestep.handler import AceStepHandler
        import torch
        import soundfile as sf_gen
        import numpy as np
        import time as _time

        logger.info("Using complete_remix mode (vocal+bass → complete task)")

        # Phase 1: LM refines caption for complexity/creativity
        refined_caption = timeline.caption
        if cfg.refine_caption_lm:
            _t_lm_start = _time.time()
            refined_caption = refine_caption_with_lm(
                caption=timeline.caption,
                lyrics=timeline.lyrics,
                bpm=metadata.get("bpm", 112),
                keyscale=metadata.get("keyscale", ""),
                duration=sf_gen.info(str(stems.instrumental)).duration,
                lm_model=cfg.lm_model,
                lm_temperature=cfg.lm_temperature,
            )
            _timings["lm_caption_refine"] = _time.time() - _t_lm_start
            logger.info(f"⏱️  LM caption refinement: {_timings['lm_caption_refine']:.1f}s")

        # Phase 2: Mix vocal + bass as source audio
        _t_mix_start = _time.time()
        if cfg.include_vocal_in_source:
            source_mix = mix_vocal_and_bass(
                vocal_path=str(stems.vocals),
                bass_path=str(stems.bass),
                bass_level_db=cfg.bass_mix_db,
            )
        else:
            # Bass only — model generates full-energy instruments without
            # ducking for the vocal. Vocal gets mixed in later by the DAW step.
            import librosa as lr
            bass_audio, sr_b = sf_gen.read(str(stems.bass))
            if bass_audio.ndim == 1:
                bass_audio = np.stack([bass_audio, bass_audio], axis=-1)
            if sr_b != 48000:
                bass_audio = lr.resample(bass_audio.T, orig_sr=sr_b, target_sr=48000).T
            peak = np.max(np.abs(bass_audio))
            if peak > 0:
                bass_audio = bass_audio / peak * 0.891
            source_mix = bass_audio
            logger.info(f"Bass-only source: {len(source_mix) / 48000:.1f}s")
        _timings["source_mix"] = _time.time() - _t_mix_start

        # Phase 3: Build instruction from genre instruments
        # Parse instruments from the rearrangement caption
        from .rearrangement import sanitize_rearrangement
        raw_instruments = parse_instruments_from_caption(timeline.caption)
        # Use the validated instruments if available
        if raw_instruments.get("melodic"):
            instruments = raw_instruments
        else:
            instruments = {"drums": "drums", "bass": "bass", "melodic": "synth"}
        instruction = build_complete_instruction(instruments)

        # Phase 4: Load DiT and generate
        _t_render_start = _time.time()
        handler = AceStepHandler()
        handler.initialize_service(project_root=".", config_path=cfg.dit_model)

        if cfg.use_scrag_vae:
            _load_scrag_vae(handler)

        # Load LoRA
        if cfg.lora_path:
            lora_dir = Path(cfg.lora_path)
            if not lora_dir.exists() and "/" not in str(cfg.lora_path):
                lora_dir = ensure_slider_lora(str(cfg.lora_path))
            if lora_dir and lora_dir.exists():
                handler.add_lora(str(lora_dir))
                handler.set_lora_scale(cfg.lora_scale)
                handler.use_lora = True
                logger.info(f"LoRA: {lora_dir} at scale {cfg.lora_scale}")

        audio_np = generate_complete_remix(
            handler=handler,
            source_mix=source_mix,
            caption=refined_caption,
            lyrics=timeline.lyrics,
            instruction=instruction,
            bpm=metadata.get("bpm"),
            keyscale=metadata.get("keyscale"),
            guidance_scale=cfg.guidance_scale,
            inference_steps=cfg.inference_steps,
            shift=cfg.shift,
        )
        _timings["complete_render"] = _time.time() - _t_render_start
        logger.info(f"⏱️  Complete render: {_timings['complete_render']:.1f}s")

        if audio_np is not None:
            gen_dir = Path(f"{output_dir}/generated")
            gen_dir.mkdir(parents=True, exist_ok=True)
            out_path = gen_dir / "complete_remix.flac"
            sf_gen.write(str(out_path), audio_np, 48000)
            instrumental_path = out_path
            logger.info(f"Complete remix output: {out_path}")
        else:
            instrumental_path = None

        # Cleanup
        del handler
        import gc as gc_mod
        gc_mod.collect()
        torch.cuda.empty_cache()

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
        import time as _time

        _t_qc_start = _time.time()
        _qc_section_log = []
        try:
            # Reuse the handler from generation (still loaded with hints + LoRA)
            fixed_path, _qc_section_log = run_qc_retry_loop(
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
                max_repaint_attempts=cfg.max_repaint_attempts,
            )
            instrumental_path = fixed_path

        except Exception as e:
            logger.warning(f"QC retry loop failed: {e}. Using spliced generation.")

        _t_qc_end = _time.time()
        _timings["qc_retry_loop"] = _t_qc_end - _t_qc_start
        logger.info(f"⏱️  QC retry loop: {_timings['qc_retry_loop']:.1f}s")

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

    # === PIPELINE SUMMARY ===
    _timings["total"] = _time.time() - _t_pipeline_start
    logger.info("")
    logger.info("=" * 60)
    logger.info("PIPELINE SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Caption: {timeline.caption[:200]}")
    logger.info(f"Lyrics (first 200): {timeline.lyrics[:200]}")
    logger.info("-" * 40)
    logger.info("⏱️  TIMINGS:")
    for step, secs in _timings.items():
        logger.info(f"  {step:25s} {secs:7.1f}s")
    logger.info("-" * 40)

    # QC retry per-section log
    if _qc_section_log:
        logger.info("QC RETRIES:")
        for entry in _qc_section_log:
            logger.info(
                f"  [{entry['label']}] {entry['start']:.1f}-{entry['end']:.1f}s: "
                f"{entry['issue']} → {entry['result']} (⏱️ {entry['time_sec']}s)"
            )
    else:
        logger.info("QC RETRIES: none needed")

    # Final QC snapshot on the output
    try:
        from .post_gen_qc import analyze_generated_audio, get_failing_sections
        final_qc = analyze_generated_audio(
            audio_path=str(instrumental_path),
            bpm=metadata.get("bpm", 100),
            key=metadata.get("keyscale", ""),
            segments=timeline.segments,
            original_audio_path=str(stems.instrumental),
        )
        failing = get_failing_sections(final_qc)
        total_sections = len(final_qc) if final_qc else 0
        passing = total_sections - len(failing)
        logger.info(f"QC FINAL: {passing}/{total_sections} sections pass")
        for sec in final_qc:
            status = "✅" if sec.passes else "❌"
            issues = []
            if sec.out_of_key_bars > 0:
                issues.append(f"out-of-key: {sec.out_of_key_bars} bars")
            if sec.chord_mismatch_bars > 0:
                issues.append(f"wrong chord: {sec.chord_mismatch_bars} bars")
            if sec.bass_dropout_bars > 0:
                issues.append(f"bass dropout: {sec.bass_dropout_bars} bars")
            issue_str = ", ".join(issues) if issues else "clean"
            logger.info(
                f"  {status} [{sec.label}] {sec.start_sec:.1f}-{sec.end_sec:.1f}s: {issue_str}"
            )
    except Exception as e:
        logger.warning(f"Final QC summary failed: {e}")

    logger.info("-" * 40)
    logger.info(f"Config: cns={cfg.cover_noise_strength}, hints={cfg.hints_strength}, "
                f"blend={cfg.use_hint_blending}, guidance={cfg.guidance_scale}, "
                f"max_retries={cfg.max_repaint_attempts}")
    logger.info("=" * 60)

    return final


def _run_hint_blending(handler, hints_stem_path: Path, output_dir: str, cfg) -> "torch.Tensor":
    """Orchestrate hint blending: chord detection → MIDI render → blend.

    Falls back to standard bass-stem hints on any failure.

    Args:
        handler: Initialized AceStepHandler.
        hints_stem_path: Path to bass stem (chord source + fallback).
        output_dir: Pipeline output directory.
        cfg: PipelineConfig with blend_factor.

    Returns:
        Blended or fallback hints tensor [B, T, D].
    """
    import time as _time
    import torch
    from .semantic_cover import extract_semantic_hints
    from .chord_detector import detect_chords
    from .midi_renderer import render_chords_to_wav
    from .hint_blender import extract_and_blend
    from .blend_metadata import write_blend_metadata

    _t_blend_start = _time.time()
    logger.info("=" * 30)
    logger.info("Hint Blending: enabled")

    # Step 1: Detect chords from bass stem
    try:
        detection = detect_chords(hints_stem_path)
    except Exception as e:
        logger.warning(f"Hint blending: chord detection failed ({e}), using bass hints")
        return extract_semantic_hints(handler, str(hints_stem_path))

    # Confidence gate
    if detection.should_skip:
        logger.warning(
            f"Hint blending: skipped (confidence={detection.confidence:.2f}), "
            f"using bass hints"
        )
        return extract_semantic_hints(handler, str(hints_stem_path))

    # Step 2: Render chords to piano WAV
    piano_path = Path(output_dir) / "hint_blend" / "piano_chords.wav"
    try:
        rendered = render_chords_to_wav(
            chords=detection.chords,
            output_path=piano_path,
            duration=detection.duration,
            sample_rate=48000,
            velocity=80,
        )
    except Exception as e:
        logger.warning(f"Hint blending: MIDI rendering failed ({e}), using bass hints")
        return extract_semantic_hints(handler, str(hints_stem_path))

    if rendered is None:
        logger.warning("Hint blending: MIDI rendering produced no output, using bass hints")
        return extract_semantic_hints(handler, str(hints_stem_path))

    # Step 3: Extract and blend hints
    try:
        blended = extract_and_blend(
            handler=handler,
            chord_source_path=hints_stem_path,
            timbre_source_path=rendered,
            blend_factor=cfg.blend_factor,
        )
    except Exception as e:
        logger.warning(f"Hint blending: extraction/blend failed ({e}), using bass hints")
        return extract_semantic_hints(handler, str(hints_stem_path))

    # Write metadata for reproducibility
    write_blend_metadata(
        output_dir=Path(output_dir) / "hint_blend",
        output_stem="cover",
        blend_factor=cfg.blend_factor,
        chord_confidence=detection.confidence,
        midi_settings={
            "sample_rate": 48000,
            "velocity": 80,
            "instrument": "piano",
            "voicing": "closed_C3_C5",
        },
    )

    logger.info(
        f"Hint blending: success (confidence={detection.confidence:.2f}, "
        f"factor={cfg.blend_factor}, chords={len(detection.chords)})"
    )
    _t_blend_end = _time.time()
    logger.info(f"⏱️  Hint blending total: {_t_blend_end - _t_blend_start:.1f}s")
    return blended


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
