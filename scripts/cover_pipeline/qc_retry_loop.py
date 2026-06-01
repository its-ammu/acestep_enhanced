"""QC retry loop: detect bad bars and fix them via repaint or splice.

Strategy:
1. Run QC on generated audio (per-bar bass dropout + out-of-key detection)
2. For failing sections: try pragmatic repaint with new seeds
3. Only apply repaint if the new version PASSES QC for that section
4. Try up to max_repaint_attempts seeds per section; keep original if none pass

This is the agentic post-generation quality control that ensures
consistent output quality regardless of random seed.
"""

from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
import torch
from loguru import logger

from .post_gen_qc import analyze_generated_audio, get_failing_sections, SectionQCResult


def _qc_section_passes(
    audio: np.ndarray,
    sr: int,
    section_start: float,
    section_end: float,
    bpm: int,
    key: str,
    label: str,
    original_audio_path: str,
) -> bool:
    """Run QC on a single section of audio and return whether it passes.

    Args:
        audio: Full audio array (samples, channels).
        sr: Sample rate.
        section_start: Section start in seconds.
        section_end: Section end in seconds.
        bpm: BPM for bar calculation.
        key: Key string (e.g., "Eb Major").
        label: Section label.
        original_audio_path: Path to original instrumental for baseline.

    Returns:
        True if the section passes QC.
    """
    import tempfile

    # Write the audio to a temp file for analysis
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
        sf.write(tmp_path, audio, sr)

    try:
        # Create a single-section segment list for targeted QC
        single_segment = [{"start": section_start, "end": section_end, "label": label}]
        results = analyze_generated_audio(
            audio_path=tmp_path,
            bpm=bpm,
            key=key,
            segments=single_segment,
            original_audio_path=original_audio_path,
        )
        # Check if the section passes
        return all(r.passes for r in results)
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def _spectral_flux_ratio(
    audio: np.ndarray,
    sr: int,
    section_start: float,
    section_end: float,
    original_audio_path: str,
) -> float:
    """Compare spectral flux of generated vs original for a section.

    A solo has high spectral flux (melody moves). Empty chords have low flux
    (static harmony). Returns ratio: generated_flux / original_flux.
    Below 0.5 means the generated section is much less melodically active.

    Args:
        audio: Generated audio (samples, channels).
        sr: Sample rate.
        section_start: Section start in seconds.
        section_end: Section end in seconds.
        original_audio_path: Path to original instrumental.

    Returns:
        Flux ratio (generated / original). <0.5 = likely missing melody.
    """
    import librosa

    start_sample = int(section_start * sr)
    end_sample = int(section_end * sr)

    # Generated section
    gen_section = audio[start_sample:end_sample]
    if gen_section.ndim == 2:
        gen_section = gen_section.mean(axis=1)

    # Original section
    y_orig, _ = librosa.load(original_audio_path, sr=sr, mono=True,
                             offset=section_start, duration=section_end - section_start)

    if len(gen_section) < sr or len(y_orig) < sr:
        return 1.0  # Can't measure, assume OK

    # Spectral flux: mean of frame-to-frame spectral change
    S_gen = np.abs(librosa.stft(gen_section.astype(np.float32), n_fft=2048, hop_length=512))
    S_orig = np.abs(librosa.stft(y_orig, n_fft=2048, hop_length=512))

    flux_gen = np.mean(np.sqrt(np.mean(np.diff(S_gen, axis=1) ** 2, axis=0)))
    flux_orig = np.mean(np.sqrt(np.mean(np.diff(S_orig, axis=1) ** 2, axis=0)))

    if flux_orig < 1e-6:
        return 1.0  # Original is silent, skip

    return flux_gen / flux_orig


def _repaint_solo_sections(
    audio: np.ndarray,
    sr: int,
    segments: list[dict],
    original_instrumental_path: str,
    handler,
    target_wavs: torch.Tensor,
    metas: list[dict],
    caption: str,
    lyrics: str,
    solo_cns: float = 0.35,
    flux_threshold: float = 0.5,
) -> Optional[np.ndarray]:
    """Detect and repaint solo/instrumental sections that lack melodic content.

    Compares spectral flux of the generated instrumental section against the
    original. If the generated version is much less active (flux ratio < 0.5),
    it means the solo melody was lost. Repaints at higher cns to preserve it.

    Args:
        audio: Generated audio (samples, channels).
        sr: Sample rate.
        segments: Section boundaries from SongFormer.
        original_instrumental_path: Path to original instrumental.
        handler: AceStepHandler with hints patched.
        target_wavs: Source audio tensor.
        metas: Metadata.
        caption: Caption.
        lyrics: Lyrics.
        solo_cns: Cover noise strength for solo repaint (higher = more faithful).
        flux_threshold: Below this ratio = solo is missing.

    Returns:
        Modified audio with solo sections repainted, or None if no change needed.
    """
    from .fast_repaint import repaint_section_fast_audio

    # Find instrumental/solo sections
    inst_segments = [
        seg for seg in segments
        if seg.get("label", "").lower() in ("inst", "instrumental", "solo")
    ]

    if not inst_segments:
        return None

    modified = False
    result_audio = audio.copy()

    for seg in inst_segments:
        start = seg["start"]
        end = seg["end"]

        # Measure: is the solo actually missing?
        flux_ratio = _spectral_flux_ratio(
            audio=result_audio, sr=sr,
            section_start=start, section_end=end,
            original_audio_path=original_instrumental_path,
        )

        logger.info(
            f"  Solo check [{seg.get('label')}] {start:.1f}-{end:.1f}s: "
            f"flux ratio={flux_ratio:.2f} (threshold={flux_threshold})"
        )

        if flux_ratio >= flux_threshold:
            logger.info(f"    ✅ Solo has sufficient melodic content — skipping")
            continue

        # Solo is empty/sparse — repaint at higher cns
        logger.info(
            f"    ⚠️  Solo lacks melody (flux {flux_ratio:.2f} < {flux_threshold}) "
            f"— repainting at cns={solo_cns}"
        )

        candidate = repaint_section_fast_audio(
            handler=handler,
            clean_audio=result_audio,
            start_sec=start,
            end_sec=end,
            target_wavs=target_wavs,
            metas=metas,
            caption=caption,
            lyrics=lyrics,
            cover_noise_strength=solo_cns,
            sr=sr,
        )

        # Verify the repaint has more melodic content
        new_flux_ratio = _spectral_flux_ratio(
            audio=candidate, sr=sr,
            section_start=start, section_end=end,
            original_audio_path=original_instrumental_path,
        )

        if new_flux_ratio > flux_ratio:
            result_audio = candidate
            modified = True
            logger.info(
                f"    ✅ Solo repaint improved: flux {flux_ratio:.2f} → {new_flux_ratio:.2f}"
            )
        else:
            logger.info(
                f"    ❌ Solo repaint didn't improve ({new_flux_ratio:.2f}) — keeping original"
            )

    return result_audio if modified else None


def run_qc_retry_loop(
    handler,
    generated_audio_path: str | Path,
    safe_audio_path: Optional[str | Path],
    original_instrumental_path: str | Path,
    bpm: int,
    key: str,
    segments: list[dict],
    target_wavs: torch.Tensor,
    metas: list[dict],
    caption: str,
    lyrics: str,
    hints: torch.Tensor,
    max_repaint_attempts: int = 3,
    bass_dropout_section_threshold: int = 3,
    solo_sections: Optional[list[dict]] = None,
) -> tuple[Path, list[dict]]:
    """Run QC on generated audio and fix failing sections/bars.

    Only applies repaint if the new version passes QC for that section.
    Tries up to max_repaint_attempts seeds per section; keeps original
    if no attempt produces a passing result.

    Returns:
        Tuple of (path to QC-fixed audio, list of per-section QC summaries).
        Each summary dict has: label, start, end, issue, result, time_sec.

    Args:
        handler: Initialized AceStepHandler (with hints patched).
        generated_audio_path: Path to the creative version.
        safe_audio_path: Path to the safe version (for section-level splice).
        original_instrumental_path: Path to original instrumental (QC baseline).
        bpm: Detected BPM.
        key: Detected key (e.g., "Eb Major").
        segments: Section boundaries from SongFormer.
        target_wavs: Source audio tensor for repaint.
        metas: Metadata dict for generation.
        caption: Caption for generation.
        lyrics: Lyrics for generation.
        hints: Semantic hints tensor (already on device).
        max_repaint_attempts: Max repaint retries per section.
        bass_dropout_section_threshold: If more than this many bars have bass
            dropout in a section, use higher cns for repaint.
        solo_sections: List of segment dicts where solos were detected.
            These get repainted at cns=0.25 for melody preservation.

    Returns:
        Path to the QC-fixed audio file.
    """
    generated_path = Path(generated_audio_path)
    output_dir = generated_path.parent

    # Step 1: Initial QC
    logger.info("=" * 40)
    logger.info("QC RETRY LOOP: Analyzing generated audio...")
    qc_results = analyze_generated_audio(
        audio_path=str(generated_path),
        bpm=bpm,
        key=key,
        segments=segments,
        original_audio_path=str(original_instrumental_path),
    )

    failing = get_failing_sections(qc_results)

    # Load the generated audio for modification
    audio, sr = sf.read(str(generated_path))
    if audio.ndim == 1:
        audio = np.stack([audio, audio], axis=-1)

    # Load safe audio if available (for section-level splice)
    safe_audio = None
    if safe_audio_path and Path(safe_audio_path).exists():
        safe_audio, _ = sf.read(str(safe_audio_path))
        if safe_audio.ndim == 1:
            safe_audio = np.stack([safe_audio, safe_audio], axis=-1)

    # Step 1b: Solo section repaint — detect instrumental sections that lack
    # melodic content and repaint at higher cns to preserve the solo melody.
    solo_repainted = _repaint_solo_sections(
        audio=audio,
        sr=sr,
        segments=segments,
        original_instrumental_path=str(original_instrumental_path),
        handler=handler,
        target_wavs=target_wavs,
        metas=metas,
        caption=caption,
        lyrics=lyrics,
    )
    if solo_repainted is not None:
        audio = solo_repainted

    if not failing:
        logger.info("QC: All sections pass — no fixes needed ✅")
        # Still save if solo was repainted
        if solo_repainted is not None:
            fixed_path = output_dir / "semantic_cover_qc_fixed.flac"
            sf.write(str(fixed_path), audio, sr)
            logger.info(f"QC fixed output (solo only): {fixed_path}")
            return fixed_path, []
        return generated_path, []

    logger.info(f"QC: {len(failing)} sections need fixing")

    # Step 2: Try to fix each failing section (only apply if passes)
    sections_fixed = 0
    sections_skipped = 0
    _qc_section_log = []

    for section in failing:
        total_bars = len(section.bars)
        bad_bars_key = [b for b in section.bars if b.out_of_key_notes]
        bad_bars_bass = [b for b in section.bars if not b.has_bass]
        bad_bars_chord = [b for b in section.bars if not b.chord_root_matches]

        # Determine cns for repaint based on issue type
        if len(bad_bars_bass) >= bass_dropout_section_threshold:
            repaint_cns = 0.3  # Higher cns preserves bass
            issue_desc = f"bass dropout in {len(bad_bars_bass)}/{total_bars} bars"
        elif bad_bars_chord:
            # Wrong chords need higher cns to follow original progression
            repaint_cns = 0.25  # More guidance from original
            issue_desc = f"wrong chord in {len(bad_bars_chord)}/{total_bars} bars"
        elif bad_bars_key:
            repaint_cns = 0.15  # Same creativity level, new seed
            issue_desc = f"{len(bad_bars_key)} bars with out-of-key notes"
        else:
            continue

        # Boost cns for intro/outro — these sections lack bidirectional context
        # and need stronger anchoring to source for chord accuracy
        if section.label in ("intro", "outro"):
            repaint_cns = max(repaint_cns, 0.35)
            issue_desc += " [intro/outro boost: cns=0.35]"

        logger.info(
            f"  [{section.label}] {section.start_sec:.1f}-{section.end_sec:.1f}s: "
            f"attempting repaint ({issue_desc})"
        )

        # Try up to max_repaint_attempts seeds
        import time as _time_qc
        from .fast_repaint import repaint_section_fast_audio

        _t_section_start = _time_qc.time()
        fixed = False
        for attempt in range(max_repaint_attempts):
            candidate = repaint_section_fast_audio(
                handler=handler,
                clean_audio=audio,
                start_sec=section.start_sec,
                end_sec=section.end_sec,
                target_wavs=target_wavs,
                metas=metas,
                caption=caption,
                lyrics=lyrics,
                cover_noise_strength=repaint_cns,
                sr=sr,
            )

            # Verify: does the repainted section pass QC?
            passes = _qc_section_passes(
                audio=candidate,
                sr=sr,
                section_start=section.start_sec,
                section_end=section.end_sec,
                bpm=bpm,
                key=key,
                label=section.label,
                original_audio_path=str(original_instrumental_path),
            )

            if passes:
                audio = candidate
                sections_fixed += 1
                _elapsed = _time_qc.time() - _t_section_start
                logger.info(
                    f"    ✅ Attempt {attempt + 1}/{max_repaint_attempts}: "
                    f"section passes QC — applied "
                    f"(⏱️ {_elapsed:.1f}s)"
                )
                _qc_section_log.append({
                    "label": section.label, "start": section.start_sec,
                    "end": section.end_sec, "issue": issue_desc,
                    "result": f"repaint_pass (attempt {attempt + 1})",
                    "time_sec": round(_elapsed, 1),
                })
                fixed = True
                break
            else:
                logger.info(
                    f"    ❌ Attempt {attempt + 1}/{max_repaint_attempts}: "
                    f"section still fails — discarding"
                )

        if not fixed and safe_audio is not None:
            # Fallback: try the safe version for this section
            logger.info(
                f"    Trying safe fallback... "
                f"(⏱️ repaint took {_time_qc.time() - _t_section_start:.1f}s)"
            )
            start_sample = int(section.start_sec * sr)
            end_sample = min(int(section.end_sec * sr), len(audio), len(safe_audio))
            crossfade_samples = int(0.1 * sr)  # 100ms crossfade

            # Build candidate with safe section spliced in
            candidate = audio.copy()
            candidate[start_sample:end_sample] = safe_audio[start_sample:end_sample]

            # Crossfade edges
            fade_in_start = max(0, start_sample - crossfade_samples)
            if fade_in_start < start_sample:
                fade_len = start_sample - fade_in_start
                fade = np.linspace(0, 1, fade_len).reshape(-1, 1)
                candidate[fade_in_start:start_sample] = (
                    audio[fade_in_start:start_sample] * (1 - fade) +
                    safe_audio[fade_in_start:start_sample] * fade
                )
            fade_out_end = min(len(candidate), end_sample + crossfade_samples)
            if fade_out_end > end_sample:
                fade_len = fade_out_end - end_sample
                fade = np.linspace(1, 0, fade_len).reshape(-1, 1)
                candidate[end_sample:fade_out_end] = (
                    safe_audio[end_sample:fade_out_end] * fade +
                    audio[end_sample:fade_out_end] * (1 - fade)
                )

            # Check if safe version passes QC for this section
            passes = _qc_section_passes(
                audio=candidate,
                sr=sr,
                section_start=section.start_sec,
                section_end=section.end_sec,
                bpm=bpm,
                key=key,
                label=section.label,
                original_audio_path=str(original_instrumental_path),
            )

            if passes:
                audio = candidate
                sections_fixed += 1
                _elapsed = _time_qc.time() - _t_section_start
                logger.info(
                    f"    ✅ Safe fallback: section passes QC — spliced safe version "
                    f"(⏱️ {_elapsed:.1f}s total)"
                )
                _qc_section_log.append({
                    "label": section.label, "start": section.start_sec,
                    "end": section.end_sec, "issue": issue_desc,
                    "result": "safe_fallback_pass",
                    "time_sec": round(_elapsed, 1),
                })
                fixed = True
            else:
                logger.info(
                    f"    ❌ Safe fallback also fails — keeping original"
                )

        if not fixed:
            _elapsed = _time_qc.time() - _t_section_start
            sections_skipped += 1
            _qc_section_log.append({
                "label": section.label, "start": section.start_sec,
                "end": section.end_sec, "issue": issue_desc,
                "result": "all_failed_kept_original",
                "time_sec": round(_elapsed, 1),
            })
            logger.warning(
                f"    [{section.label}] {section.start_sec:.1f}-{section.end_sec:.1f}s: "
                f"all attempts + safe fallback failed — keeping original "
                f"(⏱️ {_elapsed:.1f}s wasted)"
            )

    # Save result (only if at least one section was fixed)
    if sections_fixed > 0:
        fixed_path = output_dir / "semantic_cover_qc_fixed.flac"
        sf.write(str(fixed_path), audio, sr)
        logger.info(f"QC fixed output: {fixed_path}")
        logger.info(
            f"QC: {sections_fixed} sections fixed, "
            f"{sections_skipped} kept original (no passing seed found)"
        )

        # Re-verify the full output
        logger.info("QC: Re-verifying fixed output...")
        recheck = analyze_generated_audio(
            audio_path=str(fixed_path),
            bpm=bpm,
            key=key,
            segments=segments,
            original_audio_path=str(original_instrumental_path),
        )
        still_failing = get_failing_sections(recheck)
        if not still_failing:
            logger.info("QC: All sections now pass after fixes ✅")
        else:
            logger.warning(
                f"QC: {len(still_failing)} sections still failing "
                f"(kept original — no worse than before)"
            )
        return fixed_path, _qc_section_log
    else:
        logger.warning(
            f"QC: No sections could be fixed after {max_repaint_attempts} attempts each "
            f"— returning original (no degradation)"
        )
        return generated_path, _qc_section_log


def _repaint_bar(
    handler,
    bar_start: float,
    bar_end: float,
    target_wavs: torch.Tensor,
    metas: list[dict],
    caption: str,
    lyrics: str,
    sr: int = 48000,
    cover_noise_strength: float = 0.15,
) -> Optional[np.ndarray]:
    """Repaint a single bar/section using ACE-Step's repaint mode.

    Args:
        handler: Handler with hints already patched.
        bar_start: Start time in seconds.
        bar_end: End time in seconds.
        target_wavs: Full song source tensor.
        metas: Metadata for generation.
        caption: Caption.
        lyrics: Lyrics.
        sr: Sample rate.
        cover_noise_strength: Noise strength for repaint (0.25 for solos, 0.15 for QC fixes).

    Returns:
        Full-song audio array with the section repainted, or None if failed.
    """
    try:
        result = handler.service_generate(
            captions=caption,
            lyrics=lyrics,
            target_wavs=target_wavs,
            metas=metas,
            audio_cover_strength=1.0,
            guidance_scale=12.0,
            infer_steps=65,
            shift=6.0,
            cover_noise_strength=cover_noise_strength,
            repainting_start=[bar_start],
            repainting_end=[bar_end],
            task_type="cover",
            infer_method="ode",
        )

        if "target_latents" not in result:
            return None

        latents = result["target_latents"]
        if latents.shape[-1] == 64:
            latents = latents.movedim(-1, -2)
        latents = latents.to(dtype=torch.bfloat16)

        with torch.no_grad():
            audio_tensor = handler.tiled_decode(latents)

        # audio_tensor shape: (batch, channels, samples) or (channels, samples)
        audio_np = audio_tensor.float().cpu().numpy().squeeze()

        # Ensure shape is (samples, channels) to match sf.read format
        if audio_np.ndim == 1:
            audio_np = np.stack([audio_np, audio_np], axis=-1)
        elif audio_np.ndim == 2:
            # If shape is (2, N) → transpose to (N, 2)
            if audio_np.shape[0] == 2 and audio_np.shape[1] > 2:
                audio_np = audio_np.T
            # If shape is (N, 2) → already correct
        
        # Normalize
        peak = np.max(np.abs(audio_np))
        if peak > 0:
            audio_np = audio_np / peak * 0.891

        return audio_np

    except Exception as e:
        logger.warning(f"Repaint failed for {bar_start:.1f}-{bar_end:.1f}s: {e}")
        return None
