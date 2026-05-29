"""Main orchestrator for the commercial cover pipeline.

Full workflow:
1. Auto-install dependencies and models
2. Stem separation (Mel-Band RoFormer + Demucs)
3. Audio analysis (All-In-One + librosa + madmom + LLM)
4. Caption & lyrics construction
5. LM pre-processing (4B planner)
6. xl-sft Cover generation
7. Output organization

Usage:
    uv run python -m scripts.cover_pipeline.pipeline \
        --input your_song.mp3 \
        --output output/ \
        --llm_provider ollama \
        --llm_model qwen2.5-omni
"""

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Optional

from loguru import logger

from .audio_analysis import AnalysisResult, run_full_analysis
from .caption_builder import build_caption, build_lyrics, build_metadata
from .deps import ensure_dependencies
from .generate import GenerationConfig, GenerationResult, run_cover_generation, run_lm_preprocess
from .stem_separation import SeparationResult, separate_stems


def run_pipeline(
    input_audio: str | Path,
    output_dir: str | Path,
    llm_provider: str = "local",
    llm_model: str = "Qwen/Qwen2.5-Omni-7B-GPTQ-Int4",
    llm_base_url: str = "http://127.0.0.1:11434",
    llm_api_key: Optional[str] = None,
    dit_model: str = "acestep-v15-xl-sft",
    lm_model: str = "acestep-5Hz-lm-4B",
    audio_cover_strength: float = 0.9,
    guidance_scale: float = 7.0,
    inference_steps: int = 50,
    batch_size: int = 4,
    skip_separation: bool = False,
    skip_analysis: bool = False,
    skip_lm: bool = False,
    checkpoints_dir: Optional[str] = None,
) -> dict:
    """Run the complete cover pipeline.

    Args:
        input_audio: Path to the original song.
        output_dir: Output directory for all pipeline artifacts.
        llm_provider: LLM provider for audio analysis ("ollama", "gemini", "openai").
        llm_model: LLM model name for audio analysis.
        llm_base_url: LLM API base URL.
        llm_api_key: LLM API key (if required).
        dit_model: ACE-Step DiT model name.
        lm_model: ACE-Step LM model name for pre-processing.
        audio_cover_strength: Cover strength (0.0-1.0).
        guidance_scale: CFG guidance scale for xl-sft.
        inference_steps: Number of diffusion steps.
        batch_size: Number of variants to generate.
        skip_separation: Skip stem separation (use existing).
        skip_analysis: Skip audio analysis (use existing).
        skip_lm: Skip LM pre-processing.
        checkpoints_dir: Path to ACE-Step checkpoints.

    Returns:
        Dictionary with all pipeline outputs and paths.
    """
    input_audio = Path(input_audio)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_audio.exists():
        raise FileNotFoundError(f"Input audio not found: {input_audio}")

    pipeline_start = time.time()
    pipeline_result = {
        "input": str(input_audio),
        "output_dir": str(output_dir),
        "steps": {},
    }

    # =========================================================================
    # Step 0: Install dependencies
    # =========================================================================
    logger.info("=" * 60)
    logger.info("STEP 0: Checking dependencies...")
    logger.info("=" * 60)

    dep_status = ensure_dependencies(include_optional=True)
    pipeline_result["steps"]["dependencies"] = dep_status

    # =========================================================================
    # Step 1: Stem Separation
    # =========================================================================
    logger.info("=" * 60)
    logger.info("STEP 1: Stem Separation")
    logger.info("=" * 60)

    separation_dir = output_dir / "stems"
    separation_result_path = separation_dir / "separation_result.json"

    if skip_separation and separation_result_path.exists():
        logger.info("Skipping separation (using existing stems)")
        sep_data = json.loads(separation_result_path.read_text())
        separation = SeparationResult(
            vocals=Path(sep_data["vocals"]),
            instrumental=Path(sep_data["instrumental"]),
            drums=Path(sep_data["drums"]) if sep_data.get("drums") else None,
            bass=Path(sep_data["bass"]) if sep_data.get("bass") else None,
            other=Path(sep_data["other"]) if sep_data.get("other") else None,
        )
    else:
        separation = separate_stems(input_audio, separation_dir)
        # Save result
        sep_data = {
            "vocals": str(separation.vocals),
            "instrumental": str(separation.instrumental),
            "drums": str(separation.drums) if separation.drums else None,
            "bass": str(separation.bass) if separation.bass else None,
            "other": str(separation.other) if separation.other else None,
        }
        separation_result_path.parent.mkdir(parents=True, exist_ok=True)
        separation_result_path.write_text(json.dumps(sep_data, indent=2))

    pipeline_result["steps"]["separation"] = sep_data
    logger.info(f"Vocals: {separation.vocals}")
    logger.info(f"Instrumental: {separation.instrumental}")

    # =========================================================================
    # Step 2: Audio Analysis
    # =========================================================================
    logger.info("=" * 60)
    logger.info("STEP 2: Audio Analysis")
    logger.info("=" * 60)

    analysis_path = output_dir / "analysis.json"

    if skip_analysis and analysis_path.exists():
        logger.info("Skipping analysis (using existing)")
        analysis_data = json.loads(analysis_path.read_text())
        analysis = AnalysisResult(**{
            k: v for k, v in analysis_data.items()
            if k in AnalysisResult.__dataclass_fields__
        })
    else:
        # Pass Demucs stems for multi-pass LLM analysis (improves local model accuracy)
        stem_paths = None
        if separation.drums or separation.bass or separation.other:
            stem_paths = {}
            if separation.drums:
                stem_paths["drums"] = separation.drums
            if separation.bass:
                stem_paths["bass"] = separation.bass
            if separation.other:
                stem_paths["other"] = separation.other

        analysis = run_full_analysis(
            audio_path=input_audio,
            llm_provider=llm_provider,
            llm_model=llm_model,
            llm_base_url=llm_base_url,
            llm_api_key=llm_api_key,
            stem_paths=stem_paths,
        )
        # Save analysis
        analysis_data = asdict(analysis)
        analysis_path.write_text(json.dumps(analysis_data, indent=2, default=str))

    pipeline_result["steps"]["analysis"] = {
        "bpm": analysis.bpm,
        "key": analysis.key,
        "time_signature": analysis.time_signature,
        "sections": len(analysis.segments),
        "chords": len(analysis.chords),
    }

    logger.info(f"BPM: {analysis.bpm}")
    logger.info(f"Key: {analysis.key}")
    logger.info(f"Sections: {len(analysis.segments)}")
    logger.info(f"Chord segments: {len(analysis.chords)}")

    # =========================================================================
    # Step 3: Build Caption & Lyrics
    # =========================================================================
    logger.info("=" * 60)
    logger.info("STEP 3: Building Caption & Lyrics")
    logger.info("=" * 60)

    caption = build_caption(analysis)
    lyrics = build_lyrics(analysis)
    metadata = build_metadata(analysis)

    logger.info(f"Caption: {caption[:150]}...")
    logger.info(f"Lyrics sections: {lyrics.count('[')}")

    # Save caption and lyrics
    (output_dir / "caption.txt").write_text(caption)
    (output_dir / "lyrics.txt").write_text(lyrics)

    pipeline_result["steps"]["caption"] = {
        "caption": caption,
        "lyrics_preview": lyrics[:200],
        "metadata": metadata,
    }

    # =========================================================================
    # Step 4: LM Pre-Processing (4B Planner)
    # =========================================================================
    logger.info("=" * 60)
    logger.info("STEP 4: LM Pre-Processing (4B Planner)")
    logger.info("=" * 60)

    gen_config = GenerationConfig(
        dit_model=dit_model,
        lm_model=lm_model,
        audio_cover_strength=audio_cover_strength,
        guidance_scale=guidance_scale,
        inference_steps=inference_steps,
        batch_size=batch_size,
        bpm=metadata.get("bpm"),
        keyscale=metadata.get("keyscale"),
        timesignature=metadata.get("timesignature", "4/4"),
        duration=metadata.get("duration"),
        checkpoints_dir=checkpoints_dir,
    )

    enhanced_caption = caption
    audio_codes = ""

    if not skip_lm:
        lm_result = run_lm_preprocess(caption, lyrics, gen_config)
        enhanced_caption = lm_result.get("enhanced_caption", caption)
        audio_codes = lm_result.get("audio_codes", "")

        # Save LM outputs
        (output_dir / "enhanced_caption.txt").write_text(enhanced_caption)
        if audio_codes:
            (output_dir / "audio_codes.txt").write_text(audio_codes)

        pipeline_result["steps"]["lm_preprocess"] = {
            "enhanced_caption": enhanced_caption[:200],
            "audio_codes_length": len(audio_codes),
            "codes_count": audio_codes.count("<|audio_code_") if audio_codes else 0,
        }
    else:
        logger.info("Skipping LM pre-processing")
        pipeline_result["steps"]["lm_preprocess"] = {"skipped": True}

    # =========================================================================
    # Step 5: xl-sft Cover Generation
    # =========================================================================
    logger.info("=" * 60)
    logger.info("STEP 5: xl-sft Cover Generation")
    logger.info("=" * 60)

    generation_dir = output_dir / "generated"

    generated_paths = run_cover_generation(
        src_audio=input_audio,
        reference_audio=separation.instrumental,
        caption=enhanced_caption,
        lyrics=lyrics,
        config=gen_config,
        audio_codes=audio_codes,
        output_dir=generation_dir,
    )

    pipeline_result["steps"]["generation"] = {
        "count": len(generated_paths),
        "paths": [str(p) for p in generated_paths],
    }

    # =========================================================================
    # Summary
    # =========================================================================
    elapsed = time.time() - pipeline_start
    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Total time: {elapsed:.1f}s")
    logger.info(f"Generated {len(generated_paths)} instrumental variants")
    logger.info("")
    logger.info("Final assembly (DAW):")
    logger.info(f"  Vocals:       {separation.vocals}")
    logger.info(f"  Instrumental: {generation_dir}/")
    logger.info("")
    logger.info("Mix the original vocals on top of your chosen instrumental in a DAW.")

    pipeline_result["elapsed_seconds"] = elapsed
    pipeline_result["final_outputs"] = {
        "vocals_for_mix": str(separation.vocals),
        "instrumental_candidates": [str(p) for p in generated_paths],
    }

    # Save full pipeline result
    result_path = output_dir / "pipeline_result.json"
    result_path.write_text(json.dumps(pipeline_result, indent=2, default=str))
    logger.info(f"Pipeline result saved: {result_path}")

    return pipeline_result


def main():
    """CLI entry point for the cover pipeline."""
    parser = argparse.ArgumentParser(
        description="Commercial Cover Pipeline: Generate high-quality instrumental covers with original vocals.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with Ollama (local Qwen Omni)
  uv run python -m scripts.cover_pipeline.pipeline --input song.mp3 --output output/

  # With Gemini for analysis
  uv run python -m scripts.cover_pipeline.pipeline \\
      --input song.mp3 --output output/ \\
      --llm_provider gemini --llm_api_key YOUR_KEY

  # Skip steps (resume from existing analysis)
  uv run python -m scripts.cover_pipeline.pipeline \\
      --input song.mp3 --output output/ \\
      --skip_separation --skip_analysis

  # Tune generation parameters
  uv run python -m scripts.cover_pipeline.pipeline \\
      --input song.mp3 --output output/ \\
      --audio_cover_strength 0.85 \\
      --guidance_scale 8.0 \\
      --batch_size 8
        """,
    )

    # Required
    parser.add_argument("--input", "-i", required=True, help="Path to input audio file")
    parser.add_argument("--output", "-o", required=True, help="Output directory")

    # LLM configuration
    parser.add_argument("--llm_provider", default="local",
                        choices=["local", "ollama", "gemini", "openai"],
                        help="LLM provider for audio analysis (default: local)")
    parser.add_argument("--llm_model", default="Qwen/Qwen2.5-Omni-7B-GPTQ-Int4",
                        help="LLM model name (default: Qwen/Qwen2.5-Omni-7B-GPTQ-Int4)")
    parser.add_argument("--llm_base_url", default="http://127.0.0.1:11434",
                        help="LLM API base URL (default: http://127.0.0.1:11434)")
    parser.add_argument("--llm_api_key", default=None,
                        help="LLM API key (required for gemini/openai)")

    # ACE-Step model configuration
    parser.add_argument("--dit_model", default="acestep-v15-xl-sft",
                        help="DiT model (default: acestep-v15-xl-sft)")
    parser.add_argument("--lm_model", default="acestep-5Hz-lm-4B",
                        help="LM model for pre-processing (default: acestep-5Hz-lm-4B)")
    parser.add_argument("--checkpoints_dir", default=None,
                        help="Path to ACE-Step checkpoints directory")

    # Generation parameters
    parser.add_argument("--audio_cover_strength", type=float, default=0.9,
                        help="Cover strength 0.0-1.0 (default: 0.9)")
    parser.add_argument("--guidance_scale", type=float, default=7.0,
                        help="CFG guidance scale (default: 7.0)")
    parser.add_argument("--inference_steps", type=int, default=50,
                        help="Diffusion steps (default: 50)")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Number of variants to generate (default: 4)")

    # Skip flags
    parser.add_argument("--skip_separation", action="store_true",
                        help="Skip stem separation (use existing)")
    parser.add_argument("--skip_analysis", action="store_true",
                        help="Skip audio analysis (use existing)")
    parser.add_argument("--skip_lm", action="store_true",
                        help="Skip LM pre-processing")

    args = parser.parse_args()

    # Run pipeline
    try:
        result = run_pipeline(
            input_audio=args.input,
            output_dir=args.output,
            llm_provider=args.llm_provider,
            llm_model=args.llm_model,
            llm_base_url=args.llm_base_url,
            llm_api_key=args.llm_api_key,
            dit_model=args.dit_model,
            lm_model=args.lm_model,
            audio_cover_strength=args.audio_cover_strength,
            guidance_scale=args.guidance_scale,
            inference_steps=args.inference_steps,
            batch_size=args.batch_size,
            skip_separation=args.skip_separation,
            skip_analysis=args.skip_analysis,
            skip_lm=args.skip_lm,
            checkpoints_dir=args.checkpoints_dir,
        )

        # Print final summary
        print("\n" + "=" * 60)
        print("DONE — Ready for DAW assembly")
        print("=" * 60)
        print(f"\nVocals (original):  {result['final_outputs']['vocals_for_mix']}")
        print(f"Instrumentals ({len(result['final_outputs']['instrumental_candidates'])} variants):")
        for p in result["final_outputs"]["instrumental_candidates"]:
            print(f"  {p}")
        print(f"\nTotal time: {result['elapsed_seconds']:.1f}s")

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
