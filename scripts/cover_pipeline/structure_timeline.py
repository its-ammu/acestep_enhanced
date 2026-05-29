"""Structure timeline generation using SongFormer + Qwen Omni.

Produces coordinated caption + structural lyrics with per-section
energy/style hints. Uses SongFormer for precise section boundaries
and Qwen Omni for audio-grounded annotations.

Flow:
1. SongFormer → section boundaries (intro, verse, chorus, bridge, outro)
2. Qwen pass 1 → caption + vocal profile (listens to audio)
3. Qwen pass 2 → per-section energy hints (anchored to caption)
4. Programmatic assembly → combine tags + hints into lyrics
"""

import gc
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
from loguru import logger


from .audio_analysis import _truncate_to_sentences


@dataclass
class StructureTimelineResult:
    """Result of structure timeline generation."""

    caption: str = ""
    lyrics: str = ""
    segments: list[dict] = field(default_factory=list)
    hints: list[str] = field(default_factory=list)
    vocal_profile: str = ""
    genres: str = ""


# SongFormer label -> ACE-Step tag
_LABEL_TO_TAG = {
    "intro": "Intro",
    "verse": "Verse",
    "pre-chorus": "Pre-Chorus",
    "chorus": "Chorus",
    "bridge": "Bridge",
    "inst": "Instrumental",
    "outro": "Outro",
    "silence": "Silence",
}


def _map_label(label: str) -> str:
    """Convert SongFormer label to ACE-Step section name."""
    return _LABEL_TO_TAG.get(label, label.title())


# CLAP instrument labels for zero-shot classification
_CLAP_INSTRUMENT_LABELS = [
    "synthesizer",
    "electric guitar",
    "acoustic guitar",
    "piano",
    "organ",
    "strings",
    "brass",
    "flute",
    "violin",
    "saxophone",
    "harmonica",
    "mandolin",
    "banjo",
]

# Map CLAP labels to short names for the temporal script
_CLAP_LABEL_TO_SHORT = {
    "synthesizer": "Synth",
    "electric guitar": "Guitar",
    "acoustic guitar": "Acoustic guitar",
    "piano": "Piano",
    "organ": "Organ",
    "strings": "Strings",
    "brass": "Brass",
    "flute": "Flute",
    "violin": "Violin",
    "saxophone": "Saxophone",
    "harmonica": "Harmonica",
    "mandolin": "Mandolin",
    "banjo": "Banjo",
}


def _clap_identify_instrument(
    audio_chunk: "np.ndarray",
    sr: int,
    clap_model=None,
    clap_processor=None,
    threshold: float = 0.2,
) -> str:
    """Identify the dominant instrument in an audio chunk using CLAP.

    Uses zero-shot audio classification with CLAP (Contrastive Language-Audio
    Pretraining) to match the audio against candidate instrument labels.

    Args:
        audio_chunk: Mono audio array.
        sr: Sample rate.
        clap_model: Pre-loaded CLAP model (or None to load).
        clap_processor: Pre-loaded CLAP processor (or None to load).
        threshold: Minimum similarity score to report an instrument.

    Returns:
        Short instrument name (e.g., "Synth", "Guitar") or "".
    """
    import numpy as np

    try:
        import laion_clap

        # Use the laion-clap API
        if clap_model is None:
            clap_model = laion_clap.CLAP_Module(enable_fusion=False)
            clap_model.load_ckpt()  # Downloads default checkpoint

        # Ensure audio is float32, mono, correct sample rate (48kHz for CLAP)
        if audio_chunk.dtype != np.float32:
            audio_chunk = audio_chunk.astype(np.float32)

        # CLAP expects 48kHz
        if sr != 48000:
            import librosa
            audio_chunk = librosa.resample(audio_chunk, orig_sr=sr, target_sr=48000)

        # Get audio embedding
        audio_embed = clap_model.get_audio_embedding_from_data(
            [audio_chunk], use_tensor=False
        )

        # Get text embeddings for instrument labels
        text_embed = clap_model.get_text_embedding(
            _CLAP_INSTRUMENT_LABELS, use_tensor=False
        )

        # Cosine similarity
        similarities = np.dot(audio_embed, text_embed.T).flatten()

        # Get top instrument
        top_idx = int(np.argmax(similarities))
        top_score = similarities[top_idx]

        if top_score < threshold:
            return ""

        top_label = _CLAP_INSTRUMENT_LABELS[top_idx]
        short_name = _CLAP_LABEL_TO_SHORT.get(top_label, top_label.title())

        logger.debug(f"CLAP: {short_name} (score={top_score:.3f})")
        return short_name

    except ImportError:
        logger.warning("laion-clap not installed, falling back to generic 'guitar'")
        return ""
    except Exception as e:
        logger.warning(f"CLAP instrument detection failed: {e}")
        return ""


def _free_gpu():
    """Release GPU memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def analyze_structure(audio_path: str | Path) -> list[dict]:
    """Run SongFormer to detect section boundaries.

    Args:
        audio_path: Path to audio file.

    Returns:
        List of segment dicts with "label", "start", "end" keys.
    """
    audio_path = str(Path(audio_path).resolve())

    try:
        from .songformer_analyzer import analyze_and_unload

        logger.info("Analyzing song structure via SongFormer...")
        segments = analyze_and_unload(audio_path)
        _free_gpu()
        logger.info(f"SongFormer: {len(segments)} sections detected")
        return segments
    except ImportError as e:
        logger.warning(f"SongFormer not available: {e}")
        return []
    except Exception as e:
        logger.error(f"SongFormer analysis failed: {e}")
        return []


def _build_section_list(segments: list[dict]) -> tuple[str, list[str]]:
    """Build numbered section list and tags from segments.

    Returns:
        Tuple of (numbered_list_string, list_of_tag_strings).
    """
    counters: dict[str, int] = {}
    tags = []
    lines = []

    for i, seg in enumerate(segments):
        label = seg.get("label", "")
        if label in ("silence", "end"):
            continue

        tag_name = _map_label(label)

        # Number repeated sections
        if label in ("verse", "chorus"):
            counters[label] = counters.get(label, 0) + 1
            tag = f"[{tag_name} {counters[label]}]"
        else:
            tag = f"[{tag_name}]"

        tags.append(tag)
        start_m, start_s = divmod(int(seg["start"]), 60)
        end_m, end_s = divmod(int(seg["end"]), 60)
        lines.append(f"{i+1}. {tag} ({start_m}:{start_s:02d}-{end_m}:{end_s:02d})")

    return "\n".join(lines), tags


def generate_structure_timeline(
    audio_path: str | Path,
    metadata: Optional[dict] = None,
    qwen_model: str = "Qwen/Qwen2.5-Omni-7B",
    stem_paths: Optional[dict[str, Path]] = None,
) -> StructureTimelineResult:
    """Generate coordinated caption + structural lyrics.

    Args:
        audio_path: Path to audio file (instrumental stem recommended).
        metadata: Optional dict with bpm, keyscale, timesignature.
        qwen_model: HuggingFace model ID for Qwen Omni.
        stem_paths: Optional dict of stem_name -> path (drums, bass, other)
                    for per-instrument caption enrichment.

    Returns:
        StructureTimelineResult with caption, lyrics, segments, hints.
    """
    audio_path = Path(audio_path)
    result = StructureTimelineResult()

    # Step 1: SongFormer structure detection
    segments = analyze_structure(audio_path)
    result.segments = segments

    if not segments:
        logger.warning("No segments detected, using generic structure")
        result.lyrics = (
            "[Intro]\n[Instrumental]\n\n"
            "[Verse 1]\n[Instrumental]\n\n"
            "[Chorus - powerful]\n[Instrumental]\n\n"
            "[Verse 2]\n[Instrumental]\n\n"
            "[Bridge]\n[Instrumental]\n\n"
            "[Chorus - powerful]\n[Instrumental]\n\n"
            "[Outro]\n[Instrumental]"
        )
        return result

    section_list_str, section_tags = _build_section_list(segments)
    # Filter segments to match section_tags (excludes silence/end)
    filtered_segments = [s for s in segments if s.get("label") not in ("silence", "end")]
    logger.info(f"Sections ({len(section_tags)}):\n{section_list_str}")

    # Step 2: Qwen pass 1 — caption generation
    # Step 3: Qwen per-chunk — section annotations
    caption, hints, vocal_profile, genres = _run_qwen_two_pass(
        audio_path, section_list_str, section_tags, metadata, qwen_model,
        filtered_segments, stem_paths,
    )

    result.caption = caption
    result.hints = hints
    result.vocal_profile = vocal_profile
    result.genres = genres

    # Step 4: Assemble lyrics from tags + hints
    result.lyrics = _assemble_lyrics(section_tags, hints)
    logger.info(f"Generated timeline:\n{result.lyrics}")

    return result


def _analyze_section_stems(
    stem_audio: dict[str, "np.ndarray"],
    start_sample: int,
    end_sample: int,
    sr: int,
    silence_threshold_db: float = -35.0,
    solo_ratio: float = 2.0,
) -> str:
    """Analyze which stems are active in a section and build a hint string.

    Uses RMS energy per stem to determine:
    - Which instruments are present (above silence threshold)
    - Whether there's a solo/lead (one stem much louder than others)
    - Overall energy level relative to the song's peak
    - Energy trajectory (building/steady/dropping)

    Args:
        stem_audio: Dict of stem_name -> mono audio array at sr.
        start_sample: Section start in samples.
        end_sample: Section end in samples.
        sr: Sample rate.
        silence_threshold_db: Below this RMS = silent (not active).
        solo_ratio: If one stem is this many times louder than others, it's a solo.

    Returns:
        Hint string like "drums bass guitar full energy" or "guitar solo melodic".
        Empty string if no stems available.
    """
    import numpy as np

    if not stem_audio:
        return ""

    # Calculate RMS energy per stem in this section
    stem_rms = {}
    stem_rms_linear = {}
    for name, audio in stem_audio.items():
        section = audio[start_sample:min(end_sample, len(audio))]
        if len(section) == 0:
            stem_rms[name] = -100.0
            stem_rms_linear[name] = 0.0
            continue
        rms = np.sqrt(np.mean(section**2)) + 1e-10
        stem_rms[name] = 20 * np.log10(rms)
        stem_rms_linear[name] = rms

    # Calculate song-wide peak RMS per stem (for relative comparison)
    # Use the section's own context — compare stems against each other
    max_rms_db = max(stem_rms.values()) if stem_rms else -100.0

    # Determine which stems are active (relative: within 20dB of loudest)
    active_threshold = max(silence_threshold_db, max_rms_db - 20.0)
    active_stems = [name for name, db in stem_rms.items() if db > active_threshold]

    if not active_stems:
        return "silence"

    # Check for solo/lead (one stem much louder than the rest)
    if len(active_stems) >= 2:
        sorted_stems = sorted(active_stems, key=lambda n: stem_rms_linear[n], reverse=True)
        loudest = sorted_stems[0]
        second = sorted_stems[1]
        loudest_lin = stem_rms_linear[loudest]
        second_lin = stem_rms_linear[second]

        if second_lin > 0 and loudest_lin / second_lin > solo_ratio:
            # Solo detected — loudest stem dominates
            solo_name = "guitar" if loudest == "other" else loudest
            # Check if it's melodic (high spectral flux in "other" stem)
            if loudest == "other":
                return f"{solo_name} solo melodic"
            elif loudest == "drums":
                return f"drum fill intense"
            else:
                return f"{solo_name} lead prominent"

    # Overall energy level — relative to this section's peak
    # Use total combined energy
    total_rms = sum(stem_rms_linear[n] for n in active_stems)
    total_db = 20 * np.log10(total_rms + 1e-10)

    # Energy categories based on how many stems are active and their combined level
    if len(active_stems) >= 3 and total_db > -15:
        energy = "peak energy"
    elif len(active_stems) >= 3:
        energy = "full energy"
    elif len(active_stems) == 2 and total_db > -20:
        energy = "moderate energy"
    elif len(active_stems) == 2:
        energy = "moderate"
    else:
        energy = "soft"

    # Energy trajectory (compare first third vs last third)
    section_len = end_sample - start_sample
    third = section_len // 3
    first_start = start_sample
    first_end = start_sample + third
    last_start = end_sample - third
    last_end = end_sample

    first_rms_total = 0.0
    last_rms_total = 0.0
    for name in active_stems:
        audio = stem_audio[name]
        first = audio[first_start:min(first_end, len(audio))]
        last = audio[last_start:min(last_end, len(audio))]
        if len(first) > 0:
            first_rms_total += np.sqrt(np.mean(first**2))
        if len(last) > 0:
            last_rms_total += np.sqrt(np.mean(last**2))

    trajectory = ""
    if first_rms_total > 0:
        ratio = last_rms_total / first_rms_total
        if ratio > 1.4:
            trajectory = "building"
        elif ratio < 0.65:
            trajectory = "dropping"

    # Build hint: active instruments + energy + trajectory
    instrument_names = {
        "drums": "drums",
        "bass": "bass",
        "other": "guitar",
    }
    instruments = " ".join(instrument_names.get(n, n) for n in sorted(active_stems))

    parts = [instruments]
    if trajectory:
        parts.append(trajectory)
    parts.append(energy)

    return " ".join(parts)


def _run_qwen_two_pass(
    audio_path: Path,
    section_list_str: str,
    section_tags: list[str],
    metadata: Optional[dict],
    qwen_model: str,
    segments: list[dict],
    stem_paths: Optional[dict[str, Path]] = None,
) -> tuple[str, list[str], str, str]:
    """Run Qwen Omni: per-stem analysis, caption, then per-chunk hints.

    Loads Qwen ONCE, runs all inferences, then unloads.

    Returns:
        Tuple of (caption, hints_list, vocal_profile, genres).
    """
    import gc
    import tempfile

    import librosa
    import soundfile as sf

    # Build metadata context
    meta_str = ""
    if metadata:
        parts = []
        if metadata.get("bpm"):
            parts.append(f"{metadata['bpm']} BPM")
        if metadata.get("keyscale"):
            parts.append(f"key of {metadata['keyscale']}")
        if metadata.get("timesignature"):
            parts.append(f"{metadata['timesignature']} time")
        if parts:
            meta_str = f"Music metadata: {', '.join(parts)}\n\n"

    # Load Qwen once
    try:
        from qwen_omni_utils import process_mm_info
        from transformers import AutoProcessor, Qwen2_5OmniThinkerForConditionalGeneration
    except ImportError as e:
        logger.error(f"Cannot load Qwen (missing deps): {e}")
        return "", [""] * len(section_tags), "", ""

    logger.info(f"Loading Qwen2.5-Omni ({qwen_model}) for structure annotation...")
    processor = AutoProcessor.from_pretrained(qwen_model, trust_remote_code=True)
    qwen_model_obj = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
        qwen_model,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    qwen_model_obj.eval()
    logger.info("Qwen loaded. Running per-stem + caption + per-section annotation...")

    try:
        # Per-stem analysis (drums, bass, other)
        stem_descriptions = {}
        if stem_paths:
            stem_prompts = {
                "drums": "Describe the drum patterns in this track: kick pattern, snare pattern, cymbal pattern, groove feel. Be specific about which beats they hit. One paragraph, plain text.",
                "bass": "Describe the bass in this track: instrument type, tone, playing technique, note patterns. One paragraph, plain text.",
                "other": "Describe the melodic instruments in this track: what instruments, their timbre, effects, playing style. One paragraph, plain text.",
            }
            for stem_name, stem_path in stem_paths.items():
                if stem_path and Path(stem_path).exists():
                    prompt = stem_prompts.get(stem_name, f"Describe this {stem_name} track. Plain text.")
                    logger.info(f"Analyzing stem: {stem_name}")
                    desc = _qwen_infer(qwen_model_obj, processor, Path(stem_path), prompt)
                    desc = _truncate_to_sentences(desc, max_sentences=2)
                    stem_descriptions[stem_name] = desc
                    logger.info(f"  {stem_name}: {desc[:80]}...")

        # Build enriched caption prompt with stem context
        stem_context = ""
        if stem_descriptions:
            stem_context = "The original track has these instruments:\n"
            for name, desc in stem_descriptions.items():
                stem_context += f"- {name}: {desc}\n"
            stem_context += "\n"

        # Pass 1: Caption — describe instruments for a cover version
        # Skip genre detection (Qwen is unreliable at genre classification)
        # Instead, focus on instrument tones which Qwen hears accurately
        caption_prompt = (
            f"{meta_str}"
            f"{stem_context}"
            "Write a short music production caption for this instrumental track. "
            "Describe the instruments and their tone using comma-separated phrases. "
            "Focus on: what instruments are playing, their tone quality (warm/bright/"
            "gritty/smooth/compressed/open), and the overall production feel.\n"
            "Keep it to ONE line, under 30 words. No genre labels needed.\n"
            "Example: 'warm overdriven electric guitar, tight compressed drums, "
            "smooth round bass, polished pop-rock production'\n"
            "Write plain text only."
        )
        caption = _qwen_infer(qwen_model_obj, processor, audio_path, caption_prompt)
        caption = _truncate_to_sentences(caption, max_sentences=2)
        # Remove Qwen hallucination patterns
        hallucination_markers = [
            "Human provided", "human provided", "However they",
            "please ignore", "from your answer", "instructions ask",
        ]
        for marker in hallucination_markers:
            if marker in caption:
                caption = caption[:caption.index(marker)].rstrip(" .,;")
                break
        logger.info(f"Caption: {caption[:150]}...")

        # Pass 2: Per-section hints
        # Strategy: programmatic for drums/bass presence + CLAP for instrument ID
        # + Qwen for playing style (solo vs chords vs arpeggio)
        y, sr = librosa.load(str(audio_path), sr=None)
        import numpy as np

        # Load all stems for per-section energy analysis
        stem_audio = {}
        other_stem_path = None
        if stem_paths:
            for name, path in stem_paths.items():
                if path and Path(path).exists():
                    stem_y, _ = librosa.load(str(path), sr=sr, mono=True)
                    stem_audio[name] = stem_y
                    if name == "other":
                        other_stem_path = path

        # Load "other" stem at native sr for Qwen chunks
        other_audio = None
        other_sr = sr
        if other_stem_path and Path(other_stem_path).exists():
            other_audio, other_sr = librosa.load(str(other_stem_path), sr=None, mono=True)

        hints = []

        for i, (tag, seg) in enumerate(zip(section_tags, segments)):
            start_sample = int(seg["start"] * sr)
            end_sample = int(seg["end"] * sr)
            chunk = y[start_sample:end_sample]

            if len(chunk) < sr:
                hints.append("")
                continue

            # Programmatic: drums/bass presence + energy
            drums_active = False
            bass_active = False
            if "drums" in stem_audio:
                drums_section = stem_audio["drums"][start_sample:min(end_sample, len(stem_audio["drums"]))]
                if len(drums_section) > 0:
                    drums_rms = 20 * np.log10(np.sqrt(np.mean(drums_section**2)) + 1e-10)
                    drums_active = drums_rms > -40

            if "bass" in stem_audio:
                bass_section = stem_audio["bass"][start_sample:min(end_sample, len(stem_audio["bass"]))]
                if len(bass_section) > 0:
                    bass_rms = 20 * np.log10(np.sqrt(np.mean(bass_section**2)) + 1e-10)
                    bass_active = bass_rms > -40

            # Ensemble solo detection + Qwen melodic character for "other" stem
            other_hint = ""
            is_solo = False
            if other_audio is not None:
                other_start = int(seg["start"] * other_sr)
                other_end = int(seg["end"] * other_sr)
                other_chunk = other_audio[other_start:other_end]

                if len(other_chunk) > other_sr:  # At least 1 second
                    solo_votes = 0

                    # Vote 1: Spectral flux (high melodic movement = solo)
                    S_other = np.abs(librosa.stft(other_chunk, n_fft=2048, hop_length=512))
                    flux = np.sqrt(np.mean(np.diff(S_other, axis=1) ** 2))
                    # Compare against average flux across all sections
                    # Higher threshold — only truly melodic sections
                    if flux > 0.03:  # Raised from 0.015
                        solo_votes += 1

                    # Vote 2: Relative energy (solo sections are louder)
                    other_rms = np.sqrt(np.mean(other_chunk**2))
                    # Compare against full "other" stem average
                    full_other_rms = np.sqrt(np.mean(other_audio**2)) + 1e-10
                    energy_ratio = other_rms / full_other_rms
                    if energy_ratio > 1.5:  # Raised from 1.3 — must be clearly louder
                        solo_votes += 1

                    # Vote 3: Qwen analysis (style only — CLAP handles instrument ID)
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                        sf.write(tmp.name, other_chunk, other_sr)
                        chunk_path = Path(tmp.name)
                    try:
                        # Qwen: identify instrument and playing style
                        chunk_prompt = (
                            f"This is a {seg['end'] - seg['start']:.0f}-second clip of "
                            f"a melodic instrument (isolated from drums and bass). "
                            f"Name the instrument and how it's being played. "
                            f"Reply in EXACTLY 2-3 words: instrument first, then style.\n"
                            f"Examples: 'Synth arpeggios', 'Guitar solo', 'Piano chords', "
                            f"'Keyboard comping', 'Synth pad', 'Guitar power chords', "
                            f"'Synth lead melody', 'Piano sustained'\n"
                            f"Reply with ONLY 2-3 words."
                        )
                        other_hint = _qwen_infer(qwen_model_obj, processor, chunk_path, chunk_prompt)
                        other_hint = other_hint.split("\n")[0].strip().strip("'\"").rstrip(".")
                        # Clean up: keep only alphabetic words, filter junk
                        _STOP_WORDS = {"or", "and", "the", "a", "an", "is", "it", "of", "in", "to", "with"}
                        words = other_hint.split()
                        clean_words = [
                            w.strip("',.-\"") for w in words
                            if w.strip("',.-\"").isalpha()
                            and w.strip("',.-\"").lower() not in _STOP_WORDS
                        ]
                        if len(clean_words) > 3:
                            clean_words = clean_words[:3]
                        other_hint = " ".join(clean_words).title()

                        # Check if Qwen detected solo
                        solo_keywords = ["solo", "melody", "lead", "melodic"]
                        if any(kw in other_hint.lower() for kw in solo_keywords):
                            solo_votes += 1
                    finally:
                        chunk_path.unlink(missing_ok=True)

                    # Ensemble decision: ALL 3 must agree for solo override
                    # Otherwise, trust Qwen's description (it was good in v17)
                    if solo_votes >= 3:
                        is_solo = True
                        other_hint = "lead melody solo"
                        logger.info(f"    Solo confirmed (all 3 votes: flux={flux:.4f}, energy={energy_ratio:.2f})")

            # Build combined hint
            parts = []
            if drums_active:
                parts.append("drums")
            if bass_active:
                parts.append("bass")
            if other_hint:
                parts.append(other_hint)
            elif "other" in stem_audio:
                other_section = stem_audio["other"][start_sample:min(end_sample, len(stem_audio["other"]))]
                if len(other_section) > 0:
                    other_rms = 20 * np.log10(np.sqrt(np.mean(other_section**2)) + 1e-10)
                    if other_rms > -40:
                        parts.append("guitar")

            hint = " ".join(parts) if parts else "soft"
            hints.append(hint)
            logger.info(f"  Section {i+1}/{len(section_tags)} {tag}: {hint}")

    finally:
        del qwen_model_obj
        del processor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        logger.info("Qwen unloaded, GPU memory freed")

    return caption, hints, "", ""


def _qwen_infer(model, processor, audio_path: Path, prompt: str) -> str:
    """Run a single Qwen inference on an audio file.

    Args:
        model: Loaded Qwen model.
        processor: Loaded Qwen processor.
        audio_path: Path to audio file.
        prompt: Text prompt.

    Returns:
        Generated text response.
    """
    from qwen_omni_utils import process_mm_info

    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": str(audio_path)},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    text = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
    audios, images, videos = process_mm_info(conversation, use_audio_in_video=True)

    inputs = processor(
        text=[text],
        audio=audios[0] if audios else None,
        sampling_rate=16000,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.5,
        )

    input_len = inputs["input_ids"].shape[-1]
    generated_ids = output_ids[0][input_len:]
    response = processor.decode(generated_ids, skip_special_tokens=True)

    # Truncate at degenerate patterns
    for stop in ["Human:", "human:", "User:", "user:", "\n\n\n"]:
        if stop in response:
            response = response[:response.index(stop)]

    return response.strip()


def _parse_hints(raw: str, expected_count: int) -> list[str]:
    """Parse hint list from Qwen response. Handles multiple formats."""
    hints = []

    for line in raw.split("\n"):
        line = line.strip()
        if not line:
            continue

        # Format: "1. hint text" or "1) hint text"
        match = re.match(r"^\d+[\.\)]\s*(.+)$", line)
        if match:
            hints.append(match.group(1).strip())
            continue

        # Format: "[Section]: hint text" or "[Section] - hint text"
        match = re.match(r"^\[.*?\][\s:\-]+(.+)$", line)
        if match:
            hints.append(match.group(1).strip())
            continue

        # Format: "- hint text"
        match = re.match(r"^[-•]\s+(.+)$", line)
        if match:
            hints.append(match.group(1).strip())
            continue

    # Pad or truncate to match section count
    while len(hints) < expected_count:
        hints.append("")
    return hints[:expected_count]


def _assemble_lyrics(section_tags: list[str], hints: list[str]) -> str:
    """Combine section tags with hints into ACE-Step lyrics format."""
    lines = []
    for tag, hint in zip(section_tags, hints):
        if hint:
            # Combine tag with hint: [Chorus] + "high energy" -> [Chorus - high energy]
            tag_with_hint = f"{tag[:-1]} - {hint}]"
        else:
            tag_with_hint = tag
        lines.append(tag_with_hint)
        lines.append("[Instrumental]")
        lines.append("")

    return "\n".join(lines).strip()
