"""Audio analysis using specialized MIR models + LLM for subjective description.

Model stack:
- All-In-One (allin1): BPM, beats, downbeats, structure segments
- madmom: Chord recognition (frame-level)
- librosa: Key detection (Krumhansl-Schmuckler), chroma features
- Qwen Omni / Gemini: Instrument timbre, production style, texture description
"""

import json
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from loguru import logger


@dataclass
class ChordSegment:
    """A chord with its time span."""

    start: float
    end: float
    chord: str


@dataclass
class StructureSegment:
    """A song section with its time span and label."""

    start: float
    end: float
    label: str


@dataclass
class AnalysisResult:
    """Complete analysis of a song."""

    # From All-In-One / madmom
    bpm: float = 0.0
    key: str = ""
    time_signature: str = "4/4"
    duration: float = 0.0

    # Structure
    segments: list[StructureSegment] = field(default_factory=list)

    # Chords (frame-level)
    chords: list[ChordSegment] = field(default_factory=list)

    # Chord summary per section
    chord_summary: dict[str, str] = field(default_factory=dict)

    # From LLM (Qwen Omni / Gemini)
    instrument_description: str = ""
    production_description: str = ""
    timbre_keywords: list[str] = field(default_factory=list)

    # Raw LLM response
    llm_raw: str = ""


def analyze_bpm_and_structure(audio_path: str | Path) -> dict:
    """Analyze BPM, beats, and structure using All-In-One.

    Returns dict with keys: bpm, segments, beats, downbeats.
    """
    audio_path = Path(audio_path)
    result = {"bpm": 0.0, "segments": [], "beats": [], "downbeats": []}

    try:
        import allin1

        analysis = allin1.analyze(str(audio_path))
        result["bpm"] = analysis.bpm if hasattr(analysis, "bpm") else 0.0

        if hasattr(analysis, "segments"):
            for seg in analysis.segments:
                result["segments"].append({
                    "start": seg.start,
                    "end": seg.end,
                    "label": seg.label,
                })

        if hasattr(analysis, "beats"):
            result["beats"] = [b for b in analysis.beats]

        if hasattr(analysis, "downbeats"):
            result["downbeats"] = [d for d in analysis.downbeats]

        logger.info(f"All-In-One: BPM={result['bpm']}, segments={len(result['segments'])}")

    except ImportError:
        logger.warning("allin1 not installed, falling back to librosa for BPM")
        try:
            import librosa

            y, sr = librosa.load(str(audio_path))
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            result["bpm"] = float(tempo[0]) if hasattr(tempo, "__len__") else float(tempo)
            logger.info(f"librosa BPM: {result['bpm']}")
        except Exception as e:
            logger.error(f"BPM detection failed: {e}")

    except Exception as e:
        logger.error(f"All-In-One analysis failed: {e}")

    return result


def analyze_key(audio_path: str | Path, bass_stem_path: str | Path | None = None) -> str:
    """Detect musical key using essentia (primary) or librosa (fallback).

    Args:
        audio_path: Path to the full audio file.
        bass_stem_path: Optional path to isolated bass stem (unused with essentia).

    Returns:
        Key string like "C Minor" or "Eb Major".
    """
    try:
        import essentia.standard as es

        audio = es.MonoLoader(filename=str(audio_path))()
        key, scale, strength = es.KeyExtractor()(audio)
        result = f"{key} {scale.title()}"
        logger.info(f"Key detection (essentia): {result} (confidence: {strength:.3f})")
        return result

    except ImportError:
        logger.warning("essentia not installed, falling back to librosa")
    except Exception as e:
        logger.warning(f"essentia key detection failed: {e}, falling back to librosa")

    # Fallback: librosa Krumhansl-Schmuckler
    try:
        import librosa
        import numpy as np

        logger.info("Key detection: using full-mix chroma (librosa fallback)")
        y, sr = librosa.load(str(audio_path), duration=60)
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        chroma_mean = np.mean(chroma, axis=1)

        # Krumhansl-Schmuckler key profiles
        major_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
                                  2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
        minor_profile = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
                                  2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

        note_names = ["C", "C#", "D", "D#", "E", "F",
                      "F#", "G", "G#", "A", "A#", "B"]

        # Collect top candidates (both major and minor for each root)
        candidates = []
        for i in range(12):
            rotated = np.roll(chroma_mean, -i)
            corr_major = float(np.corrcoef(rotated, major_profile)[0, 1])
            corr_minor = float(np.corrcoef(rotated, minor_profile)[0, 1])
            candidates.append((corr_major, f"{note_names[i]} Major", i, "major"))
            candidates.append((corr_minor, f"{note_names[i]} Minor", i, "minor"))

        candidates.sort(key=lambda x: x[0], reverse=True)
        top = candidates[0]

        # Check if top major and its relative minor are close in score
        # (D# Major and C minor are relative — same pitch classes)
        # If they're within 0.05 correlation, use bass root to disambiguate
        top_corr, top_key, top_root, top_mode = top

        # Find the relative key
        if top_mode == "major":
            relative_root = (top_root + 9) % 12  # Relative minor is 3 semitones down
            relative_mode = "minor"
        else:
            relative_root = (top_root + 3) % 12  # Relative major is 3 semitones up
            relative_mode = "major"

        relative_key_name = f"{note_names[relative_root]} {relative_mode.title()}"
        relative_corr = 0.0
        for corr, name, root, mode in candidates:
            if root == relative_root and mode == relative_mode:
                relative_corr = corr
                break

        # If relative key is close in score, disambiguate using bass frequencies
        if abs(top_corr - relative_corr) < 0.05:
            # Check which root note is stronger in the low frequencies
            # Low-pass the audio and check chroma
            y_bass = librosa.effects.preemphasis(y, coef=-0.97)  # Boost lows
            S_bass = np.abs(librosa.stft(y, n_fft=4096, hop_length=512))
            # Only look at frequencies below 250Hz
            freqs = librosa.fft_frequencies(sr=sr, n_fft=4096)
            bass_bins = freqs < 250
            bass_spec = S_bass[bass_bins, :]
            bass_chroma = librosa.feature.chroma_stft(S=S_bass, sr=sr, n_fft=4096)
            bass_chroma_mean = bass_chroma.mean(axis=1)

            top_root_energy = bass_chroma_mean[top_root]
            relative_root_energy = bass_chroma_mean[relative_root]

            logger.info(
                f"Key disambiguation: {top_key} ({top_corr:.3f}) vs "
                f"{relative_key_name} ({relative_corr:.3f}). "
                f"Bass root energy: {note_names[top_root]}={top_root_energy:.3f}, "
                f"{note_names[relative_root]}={relative_root_energy:.3f}"
            )

            if relative_root_energy > top_root_energy * 1.1:
                # Relative key's root is stronger in bass — use relative
                best_key = relative_key_name
                best_corr = relative_corr
                logger.info(f"Key detection: {best_key} (bass-confirmed, corr: {best_corr:.3f})")
                return best_key

        best_key = top_key
        best_corr = top_corr
        logger.info(f"Key detection: {best_key} (correlation: {best_corr:.3f})")
        return best_key

    except ImportError:
        logger.error("librosa not installed for key detection")
        return ""
    except Exception as e:
        logger.error(f"Key detection failed: {e}")
        return ""


def analyze_chords(audio_path: str | Path) -> list[ChordSegment]:
    """Detect chords using madmom DeepChroma chord recognition."""
    try:
        from madmom.audio.chroma import DeepChromaProcessor
        from madmom.features.chords import (
            CRFChordRecognitionProcessor,
            DeepChromaChordRecognitionProcessor,
        )

        # Process audio
        dcp = DeepChromaProcessor()
        chroma = dcp(str(audio_path))

        # Recognize chords
        decode = DeepChromaChordRecognitionProcessor()
        chords_raw = decode(chroma)

        chords = []
        for start, end, label in chords_raw:
            chords.append(ChordSegment(start=float(start), end=float(end), chord=label))

        logger.info(f"Chord detection: {len(chords)} chord segments")
        return chords

    except ImportError:
        logger.warning("madmom not installed for chord detection, trying librosa chroma")
        return _fallback_chord_detection(audio_path)
    except Exception as e:
        logger.error(f"Chord detection failed: {e}")
        return _fallback_chord_detection(audio_path)


def _fallback_chord_detection(audio_path: str | Path) -> list[ChordSegment]:
    """Basic chord detection using librosa chroma as fallback."""
    try:
        import librosa
        import numpy as np

        y, sr = librosa.load(str(audio_path))
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=4096)

        note_names = ["C", "C#", "D", "D#", "E", "F",
                      "F#", "G", "G#", "A", "A#", "B"]

        hop_duration = 4096 / sr
        chords = []

        for i in range(chroma.shape[1]):
            frame = chroma[:, i]
            root = int(np.argmax(frame))
            # Simple major/minor detection based on third
            major_third = frame[(root + 4) % 12]
            minor_third = frame[(root + 3) % 12]
            quality = "" if major_third > minor_third else "m"
            chord_label = f"{note_names[root]}{quality}"

            start = i * hop_duration
            end = (i + 1) * hop_duration
            chords.append(ChordSegment(start=start, end=end, chord=chord_label))

        # Merge consecutive identical chords
        merged = []
        for chord in chords:
            if merged and merged[-1].chord == chord.chord:
                merged[-1].end = chord.end
            else:
                merged.append(chord)

        logger.info(f"Fallback chord detection: {len(merged)} segments")
        return merged

    except Exception as e:
        logger.error(f"Fallback chord detection failed: {e}")
        return []


def summarize_chords_by_section(
    chords: list[ChordSegment],
    segments: list[StructureSegment],
) -> dict[str, str]:
    """Map chord progressions to song sections."""
    summary = {}

    for seg in segments:
        section_chords = []
        for chord in chords:
            # Chord overlaps with this section
            if chord.end > seg.start and chord.start < seg.end:
                if chord.chord != "N" and chord.chord not in section_chords[-1:]:
                    section_chords.append(chord.chord)

        if section_chords:
            summary[f"{seg.label} ({seg.start:.1f}-{seg.end:.1f})"] = " - ".join(section_chords)

    return summary


def _truncate_to_sentences(text: str, max_sentences: int = 2) -> str:
    """Truncate text to the first N complete sentences."""
    if not text:
        return ""
    text = text.strip()

    # Find sentence boundaries (period followed by space or end)
    sentences = []
    start = 0
    for i, char in enumerate(text):
        if char == "." and (i + 1 >= len(text) or text[i + 1] in " \n"):
            sentences.append(text[start:i + 1].strip())
            start = i + 1
            if len(sentences) >= max_sentences:
                break

    if sentences:
        return " ".join(sentences)

    # No period found — truncate at 150 chars
    if len(text) > 150:
        # Cut at last space before 150
        cut = text[:150].rfind(" ")
        if cut > 50:
            return text[:cut] + "."
    return text


def analyze_with_llm(
    audio_path: str | Path,
    provider: str = "ollama",
    model: str = "qwen2.5-omni",
    base_url: str = "http://127.0.0.1:11434",
    api_key: Optional[str] = None,
    additional_stems: Optional[dict[str, Path]] = None,
) -> dict:
    """Analyze audio using an LLM with audio understanding.

    Extracts: instrument timbres, production characteristics, texture keywords.

    When additional_stems are provided (e.g., from Demucs), runs separate
    analysis passes on each stem for more detailed descriptions. This is
    especially useful for local models (Qwen Omni 7B) which perform better
    on isolated sources than full mixes.

    Args:
        audio_path: Path to audio file (full mix or instrumental).
        provider: LLM provider ("ollama", "openai", "gemini").
        model: Model name.
        base_url: API base URL.
        api_key: API key (required for non-local providers).
        additional_stems: Optional dict of stem_name -> path for multi-pass analysis.

    Returns:
        Dict with instrument_description, production_description, timbre_keywords.
    """
    audio_path = Path(audio_path)

    main_prompt = """Listen to this audio and describe the instrumental elements in one paragraph. Include:
- Each instrument present with specific timbre adjectives (warm/bright/crisp/dark/punchy/airy)
- Effects on each instrument (reverb, delay, distortion, chorus, compression)
- Production style (stereo width, frequency balance, dynamic range)
- Overall energy and mood

Be concise and specific. Use audio engineering vocabulary. Focus only on instruments, not vocals. Write plain text, no JSON."""

    # Run main analysis
    main_result = _call_llm(audio_path, main_prompt, provider, model, base_url, api_key)
    # Truncate main description to avoid rambling
    if main_result.get("instrument_description"):
        main_result["instrument_description"] = _truncate_to_sentences(
            main_result["instrument_description"], max_sentences=3
        )

    # Multi-pass on individual stems (improves local model accuracy)
    if additional_stems:
        stem_descriptions = []

        stem_prompts = {
            "drums": "Describe this drum track in one sentence: kick character, snare character, hi-hat character, groove feel. Plain text only.",
            "bass": "Describe this bass track in one sentence: instrument type, tone character, playing style, effects. Plain text only.",
            "other": "Describe these melodic instruments in one sentence: what instruments, their timbre, effects, role in arrangement. Plain text only.",
        }

        for stem_name, stem_path in additional_stems.items():
            if stem_path and Path(stem_path).exists():
                prompt = stem_prompts.get(stem_name, f"Describe this isolated {stem_name} audio track in one sentence. Plain text only.")
                logger.info(f"Analyzing stem: {stem_name}")
                stem_result = _call_llm(Path(stem_path), prompt, provider, model, base_url, api_key)
                raw = stem_result.get("instrument_description", "") or stem_result.get("raw", "")
                if raw:
                    # Truncate to first 1-2 sentences
                    clean = _truncate_to_sentences(raw, max_sentences=2)
                    stem_descriptions.append(f"[{stem_name}] {clean}")

        # Merge stem descriptions into main result
        if stem_descriptions:
            combined_instruments = main_result.get("instrument_description", "")
            stem_detail = " | ".join(stem_descriptions)
            if combined_instruments:
                main_result["instrument_description"] = f"{combined_instruments}. Stem details: {stem_detail}"
            else:
                main_result["instrument_description"] = stem_detail

    return main_result


def _call_llm(
    audio_path: Path,
    prompt: str,
    provider: str,
    model: str,
    base_url: str,
    api_key: Optional[str],
) -> dict:
    """Dispatch LLM call to the appropriate provider."""
    if provider == "gemini":
        return _analyze_with_gemini(audio_path, prompt, model, api_key)
    elif provider == "ollama":
        return _analyze_with_ollama(audio_path, prompt, model, base_url)
    elif provider == "openai":
        return _analyze_with_openai(audio_path, prompt, model, api_key, base_url)
    elif provider == "local":
        return _analyze_with_local_qwen(audio_path, prompt, model)
    else:
        logger.warning(f"Unknown LLM provider: {provider}")
        return {"instrument_description": "", "production_description": "", "timbre_keywords": [], "raw": ""}


def _analyze_with_gemini(
    audio_path: Path,
    prompt: str,
    model: str,
    api_key: Optional[str],
) -> dict:
    """Use Gemini API for audio analysis."""
    import os

    api_key = api_key or os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        logger.error("GEMINI_API_KEY not set")
        return {"instrument_description": "", "production_description": "", "timbre_keywords": [], "raw": ""}

    try:
        # Use the existing gemini_caption module
        sys.path.insert(0, str(Path(__file__).parent.parent / "lora_data_prepare"))
        from gemini_caption import get_gemini_service

        service = get_gemini_service(api_key)
        response = service.analyze_audio(
            str(audio_path),
            prompt=prompt,
            model_name=model or "gemini-2.5-pro",
        )

        return _parse_llm_response(response or "")

    except Exception as e:
        logger.error(f"Gemini analysis failed: {e}")
        return {"instrument_description": "", "production_description": "", "timbre_keywords": [], "raw": ""}


def _analyze_with_ollama(
    audio_path: Path,
    prompt: str,
    model: str,
    base_url: str,
) -> dict:
    """Use Ollama (local) for audio analysis."""
    import base64

    try:
        import requests

        # Read and encode audio
        with open(audio_path, "rb") as f:
            audio_b64 = base64.b64encode(f.read()).decode("utf-8")

        # Ollama chat API with images/audio
        response = requests.post(
            f"{base_url}/api/chat",
            json={
                "model": model,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt,
                        "images": [audio_b64],  # Ollama uses 'images' for multimodal
                    }
                ],
                "stream": False,
                "format": "json",
            },
            timeout=300,
        )

        if response.status_code == 200:
            data = response.json()
            content = data.get("message", {}).get("content", "")
            return _parse_llm_response(content)
        else:
            logger.error(f"Ollama request failed: {response.status_code} {response.text[:200]}")

    except Exception as e:
        logger.error(f"Ollama analysis failed: {e}")

    return {"instrument_description": "", "production_description": "", "timbre_keywords": [], "raw": ""}


def _analyze_with_local_qwen(
    audio_path: Path,
    prompt: str,
    model: str,
) -> dict:
    """Load Qwen2.5-Omni directly via transformers for audio analysis.

    Uses GPTQ-Int4 quantization (~12GB VRAM). Loads model, runs inference,
    then explicitly unloads to free GPU for subsequent pipeline steps.

    Args:
        audio_path: Path to audio file.
        prompt: Analysis prompt.
        model: HuggingFace model ID (default: Qwen/Qwen2.5-Omni-7B-GPTQ-Int4).
    """
    import gc

    import torch

    model_id = model or "Qwen/Qwen2.5-Omni-7B-GPTQ-Int4"
    qwen_model = None
    processor = None

    try:
        from qwen_omni_utils import process_mm_info
        from transformers import AutoProcessor, Qwen2_5OmniThinkerForConditionalGeneration

        logger.info(f"Loading Qwen2.5-Omni from {model_id}...")

        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        qwen_model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        qwen_model.eval()

        logger.info("Qwen2.5-Omni loaded, running audio analysis...")

        # Build conversation with audio file path
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": str(audio_path)},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        # Use process_mm_info to preprocess audio into the format the model expects
        text = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
        audios, images, videos = process_mm_info(conversation, use_audio_in_video=True)

        # Tokenize text and process audio features separately
        inputs = processor(
            text=[text],
            audio=audios[0] if audios else None,
            sampling_rate=16000,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(qwen_model.device)

        # Generate (text only, no speech output needed)
        with torch.no_grad():
            output_ids = qwen_model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.5,
            )

        # Decode — strip input tokens
        input_len = inputs["input_ids"].shape[-1]
        generated_ids = output_ids[0][input_len:]
        response_text = processor.decode(generated_ids, skip_special_tokens=True)

        # Truncate at any degenerate patterns (model simulating conversation)
        for stop_pattern in ["Human:", "human:", "User:", "user:", "\n\n\n"]:
            if stop_pattern in response_text:
                response_text = response_text[:response_text.index(stop_pattern)]

        logger.info(f"Qwen2.5-Omni response: {response_text[:200]}...")
        return _parse_llm_response(response_text)

    except ImportError as e:
        logger.error(f"Cannot load Qwen2.5-Omni (missing deps): {e}")
        logger.error(
            "Install: pip install 'transformers>=4.51' accelerate "
            "'qwen-omni-utils[decord]'"
        )
        return {"instrument_description": "", "production_description": "", "timbre_keywords": [], "raw": ""}
    except Exception as e:
        logger.error(f"Qwen2.5-Omni inference failed: {e}")
        return {"instrument_description": "", "production_description": "", "timbre_keywords": [], "raw": ""}
    finally:
        # Explicitly free GPU memory
        del qwen_model
        del processor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        logger.info("Qwen2.5-Omni unloaded, GPU memory freed")


def _analyze_with_openai(
    audio_path: Path,
    prompt: str,
    model: str,
    api_key: Optional[str],
    base_url: str,
) -> dict:
    """Use OpenAI-compatible API for audio analysis."""
    import base64
    import os

    api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        logger.error("OPENAI_API_KEY not set")
        return {"instrument_description": "", "production_description": "", "timbre_keywords": [], "raw": ""}

    try:
        import requests

        with open(audio_path, "rb") as f:
            audio_b64 = base64.b64encode(f.read()).decode("utf-8")

        # Determine mime type
        suffix = audio_path.suffix.lower()
        mime_map = {".mp3": "audio/mp3", ".wav": "audio/wav", ".flac": "audio/flac", ".ogg": "audio/ogg"}
        mime_type = mime_map.get(suffix, "audio/mp3")

        response = requests.post(
            f"{base_url.rstrip('/')}/chat/completions" if "/chat/completions" not in base_url else base_url,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "model": model or "gpt-4o-audio-preview",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "input_audio",
                                "input_audio": {"data": audio_b64, "format": suffix.lstrip(".")},
                            },
                        ],
                    }
                ],
                "response_format": {"type": "json_object"},
            },
            timeout=300,
        )

        if response.status_code == 200:
            data = response.json()
            content = data["choices"][0]["message"]["content"]
            return _parse_llm_response(content)
        else:
            logger.error(f"OpenAI request failed: {response.status_code}")

    except Exception as e:
        logger.error(f"OpenAI analysis failed: {e}")

    return {"instrument_description": "", "production_description": "", "timbre_keywords": [], "raw": ""}


def _parse_llm_response(raw: str) -> dict:
    """Parse LLM response into structured result. Handles both JSON and plain text."""
    result = {
        "instrument_description": "",
        "production_description": "",
        "timbre_keywords": [],
        "raw": raw,
    }

    if not raw or not raw.strip():
        return result

    text = raw.strip()

    # Try JSON parsing first
    try:
        json_text = text
        if "```json" in json_text:
            json_text = json_text.split("```json")[1].split("```")[0]
        elif "```" in json_text:
            json_text = json_text.split("```")[1].split("```")[0]

        start = json_text.find("{")
        end = json_text.rfind("}")
        if start >= 0 and end > start:
            json_text = json_text[start:end + 1]
            json_text = json_text.replace("\\'", "'")

            import re
            json_text = re.sub(r",\s*([}\]])", r"\1", json_text)
            json_text = re.sub(r"//[^\n]*", "", json_text)

            data = json.loads(json_text)

            instruments = data.get("instruments", data.get("description", ""))
            if isinstance(instruments, dict):
                parts = []
                for inst_name, desc in instruments.items():
                    if isinstance(desc, dict):
                        desc_str = ", ".join(f"{k}: {v}" for k, v in desc.items() if v)
                        parts.append(f"{inst_name} ({desc_str})")
                    elif isinstance(desc, str):
                        parts.append(f"{inst_name}: {desc}")
                result["instrument_description"] = "; ".join(parts)
            elif isinstance(instruments, list):
                result["instrument_description"] = "; ".join(str(i) for i in instruments)
            else:
                result["instrument_description"] = str(instruments)

            production = data.get("production", "")
            if isinstance(production, dict):
                result["production_description"] = ", ".join(f"{k}: {v}" for k, v in production.items() if v)
            else:
                result["production_description"] = str(production)

            result["timbre_keywords"] = data.get("timbre_keywords", [])
            if data.get("energy_arc"):
                result["production_description"] += f" Dynamic arc: {data['energy_arc']}"

            return result

    except (json.JSONDecodeError, KeyError, IndexError, TypeError):
        pass

    # Plain text fallback — truncate at first complete sentence(s)
    clean_text = text[:500].strip()
    # Find the second period (keep ~2 sentences max)
    first_period = clean_text.find(".", 30)
    if first_period > 0:
        second_period = clean_text.find(".", first_period + 1)
        if second_period > 0 and second_period < 300:
            clean_text = clean_text[:second_period + 1]
        elif first_period < 200:
            clean_text = clean_text[:first_period + 1]

    # Remove any trailing fragments after semicolons that go nowhere
    last_semi = clean_text.rfind(";")
    if last_semi > 100 and last_semi > len(clean_text) * 0.7:
        clean_text = clean_text[:last_semi + 1]

    result["instrument_description"] = clean_text.strip()
    return result


def run_full_analysis(
    audio_path: str | Path,
    llm_provider: str = "ollama",
    llm_model: str = "qwen2.5-omni",
    llm_base_url: str = "http://127.0.0.1:11434",
    llm_api_key: Optional[str] = None,
    stem_paths: Optional[dict[str, Path]] = None,
) -> AnalysisResult:
    """Run the complete analysis pipeline.

    Combines:
    1. All-In-One: BPM, structure, beats
    2. librosa: Key detection
    3. madmom: Chord recognition
    4. LLM: Instrument/production description (with optional multi-stem passes)

    Args:
        audio_path: Path to the audio file.
        llm_provider: LLM provider for subjective analysis.
        llm_model: LLM model name.
        llm_base_url: LLM API base URL.
        llm_api_key: LLM API key.
        stem_paths: Optional dict of stem_name -> path (from Demucs) for
                    multi-pass LLM analysis. Improves local model accuracy.

    Returns:
        Complete AnalysisResult.
    """
    audio_path = Path(audio_path)
    result = AnalysisResult()

    # Get duration
    try:
        import librosa

        result.duration = librosa.get_duration(path=str(audio_path))
    except Exception:
        pass

    # 1. BPM + Structure (All-In-One)
    logger.info("Analyzing BPM and structure...")
    structure_data = analyze_bpm_and_structure(audio_path)
    result.bpm = structure_data.get("bpm", 0.0)
    result.segments = [
        StructureSegment(start=s["start"], end=s["end"], label=s["label"])
        for s in structure_data.get("segments", [])
    ]

    # 2. Key Detection (librosa)
    logger.info("Detecting key...")
    result.key = analyze_key(audio_path)

    # 3. Chord Recognition (madmom)
    logger.info("Analyzing chords...")
    result.chords = analyze_chords(audio_path)

    # Summarize chords by section
    if result.chords and result.segments:
        result.chord_summary = summarize_chords_by_section(result.chords, result.segments)

    # 4. LLM Analysis (instrument timbre + production)
    logger.info(f"Running LLM analysis ({llm_provider}/{llm_model})...")
    llm_result = analyze_with_llm(
        audio_path,
        provider=llm_provider,
        model=llm_model,
        base_url=llm_base_url,
        api_key=llm_api_key,
        additional_stems=stem_paths,
    )
    result.instrument_description = llm_result.get("instrument_description", "")
    result.production_description = llm_result.get("production_description", "")
    result.timbre_keywords = llm_result.get("timbre_keywords", [])
    result.llm_raw = llm_result.get("raw", "")

    # Log summary
    logger.info(f"Analysis complete: BPM={result.bpm}, Key={result.key}, "
                f"Sections={len(result.segments)}, Chords={len(result.chords)}")

    return result
