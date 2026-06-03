# V48 Cover Genre Pipeline — Complete Technical Documentation

## Overview

V48 uses the `cover_genre` generation mode to produce **genre-shifted instrumental covers** that preserve the original song's structure while transforming it into a completely different genre. The original vocals are preserved and remixed on top of the new AI-generated instrumental.

**Key characteristics:**
- No bass hints (hints + CFG conflict, degrading chord accuracy)
- No LM caption refinement (LM ignores genre intent, rewrites to random genres)
- Template caption from validated instrument vocabulary
- Two-pass generation: timbre reference → cover
- Cover mode with low `audio_cover_strength` (0.4) for maximum genre freedom
- High `guidance_scale` (9.0) for strong caption adherence via CFG

**Results:**
| Song | Chroma | Rhythm | Genre Achieved |
|------|:------:|:------:|----------------|
| Just The Way It Is | 0.301 | 0.666 | Funk/Neo-Soul |
| Its Been Awhile | 0.487 | — | Trip-Hop |

---

## Pipeline Configuration (PipelineConfig)

```python
PipelineConfig(
    input_song='data/song.mp3',
    output_dir='data/output/song_v48',
    generation_mode='cover_genre',
    dit_model='acestep-v15-xl-sft',
    audio_cover_strength=0.4,
    cover_noise_strength=0.0,
    guidance_scale=9.0,
    inference_steps=28,
    shift=3.0,
    use_timbre_reference=True,
    refine_caption_lm=False,
    lora_path='digital-acoustic',
    lora_scale=0.7,
    use_scrag_vae=True,
    rearrange=True,
    max_repaint_attempts=3,
)
```

---

## Complete Process Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        V48 COVER GENRE PIPELINE                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  STEP 1: Stem Separation                                                    │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  Input Song (.mp3/.wav/.flac)                                       │   │
│  │       │                                                             │   │
│  │       ├─► Mel-Band RoFormer ──► Vocals (preserved for final mix)    │   │
│  │       │                     └─► Instrumental (structural source)     │   │
│  │       │                                                             │   │
│  │       └─► Demucs v4 ──► Drums stem (per-section energy analysis)    │   │
│  │                     ├─► Bass stem (per-section energy analysis)      │   │
│  │                     └─► Other stem (melodic instrument detection)    │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  STEP 2: Metadata Detection                                                 │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  librosa ──► BPM (beat tracking)                                    │   │
│  │  Krumhansl-Schmuckler + bass disambiguation ──► Key (e.g. Eb Major) │   │
│  │  Default ──► Time Signature (4/4)                                   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  STEP 3: Structure Timeline                                                 │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  Phase 1: SongFormer ──► Section boundaries                         │   │
│  │  Phase 2: Qwen per-stem ──► Instrument descriptions                 │   │
│  │  Phase 3: Qwen caption ──► Original instrument description          │   │
│  │  Phase 4: Qwen per-section ──► Energy/style hints                   │   │
│  │  Phase 5: Qwen genre selection ──► Genre number (1-10)              │   │
│  │  Phase 6: Template caption ──► Performance-rich genre caption        │   │
│  │  Phase 7: Translate temporal script ──► Energy-only lyrics           │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  STEP 4: Cover Generation (Two-Pass)                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  Pass 1: Timbre Reference                                           │   │
│  │     caption + lyrics + BPM/key + silence ──► text2music ──► ref_audio│   │
│  │                                                                     │   │
│  │  Pass 2: Actual Cover                                               │   │
│  │     caption + lyrics + BPM/key                                      │   │
│  │     + instrumental stem (structural skeleton, acs=0.4)              │   │
│  │     + ref_audio (timbre target)                                     │   │
│  │     + LoRA (digital-acoustic @ 0.7)                                 │   │
│  │     + ScragVAE decoder                                              │   │
│  │     ──► cover generation ──► new instrumental                       │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  STEP 5: Post-Generation QC                                                 │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  Per-bar analysis: bass dropout + out-of-key + chord root mismatch  │   │
│  │  Failing sections ──► repaint (higher acs=0.6, new seed)            │   │
│  │  Up to 3 attempts per section                                       │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  STEP 6: DAW Mix                                                            │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  Vocal chain: HPF 80Hz + presence + de-ess + compression + reverb   │   │
│  │  Instrumental chain: HPF 30Hz + warmth + vocal notch + compression  │   │
│  │  Dynamic balance: match loudness to original                        │   │
│  │  Master bus: light compression (1.5:1) + limiter (-1dB)             │   │
│  │  ──► final_cover_daw.wav                                            │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Step 1: Stem Separation

**File:** `stem_separation.py`
**GPU:** ~4GB total, freed after

### 1.1 Mel-Band RoFormer

The highest-quality vocal isolation available (SDR ~11.2). Uses the `audio-separator` Python package.

**Input:** Original song (any format)
**Output:**
- `vocals.wav` — Clean isolated vocals, preserved untouched for the final mix
- `instrumental.wav` — Everything except vocals, used as structural source for cover generation

### 1.2 Demucs v4 (htdemucs_ft)

Multi-stem separation for analysis purposes. Runs as subprocess.

**Input:** Original song
**Output:**
- `drums.mp3` — Isolated drums (used for per-section energy detection)
- `bass.mp3` — Isolated bass (used for per-section energy detection)
- `other.mp3` — Everything else: guitars, keys, synths (used for melodic instrument identification)

### 1.3 GPU Cleanup

```python
gc.collect()
torch.cuda.empty_cache()
torch.cuda.synchronize()
```

Both separation models are freed from GPU memory before proceeding.

---

## Step 2: Metadata Detection

**File:** `audio_analysis.py`
**GPU:** None (CPU only)

### 2.1 BPM Detection

```python
bpm_result = analyze_bpm_and_structure(input_song)
```

Uses librosa beat tracking. Output: integer BPM (e.g., 112).

### 2.2 Key Detection

```python
key = analyze_key(input_song, bass_stem_path=stems.bass)
```

Two-stage process:
1. **Krumhansl-Schmuckler profiles** — Standard chroma-based key detection using pitch class correlation against major/minor key profiles
2. **Bass-root disambiguation** — Standard detection often confuses relative major/minor (e.g., Eb Major vs C Minor share the same pitch classes). The bass stem's dominant root note disambiguates: if the bass emphasizes C more than Eb, it's C Minor.

**Why this matters:** The DiT uses the key label to decide which notes are valid. Wrong key (e.g., "Eb Major" when the song is in "C Minor") causes the model to generate notes like C# that clash with the actual harmony.

---

## Step 3: Structure Timeline

**File:** `structure_timeline.py`
**GPU:** ~4GB (SongFormer) → ~14GB (Qwen), sequential

This is the most complex step. It produces the **caption** (what instruments to use) and **lyrics** (temporal energy script) that drive the generation.

### 3.1 SongFormer Structure Detection

**Files:** `songformer_analyzer.py`, `songformer_setup.py`, `songformer_inference.py`, `songformer_embeddings.py`
**GPU:** ~4GB

#### Model Stack (3 models)

| Model | Size | Purpose |
|-------|------|---------|
| MuQ (MuQ-large-msd-iter) | ~1.5GB | Music understanding embeddings |
| MusicFM (MusicFM25Hz) | ~1.5GB | Music foundation model embeddings |
| SongFormer (transformer) | ~1GB | Section classification |

#### Process

1. **Load audio at 24kHz** (SongFormer's native sample rate)
2. **Sliding window** over the audio (420-second windows, 420-second hop):
   - Feed each window through MuQ → extract hidden state layer 10 (2048-dim at ~8.3 frames/sec)
   - Feed same window through MusicFM → extract hidden state layer 10 (2048-dim)
   - Also process 30-second sub-chunks for finer detail
   - Concatenate all 4 embeddings → `[1, frames, 4096]`
3. **SongFormer inference** — The transformer classifies each frame into one of 128 structure labels
4. **Postprocessing** — Converts frame-level predictions to timestamped segments, removes short artifacts

#### Output

```python
segments = [
    {"label": "intro", "start": 0.0, "end": 17.2},
    {"label": "verse", "start": 17.2, "end": 45.8},
    {"label": "chorus", "start": 45.8, "end": 72.1},
    {"label": "verse", "start": 72.1, "end": 100.5},
    {"label": "chorus", "start": 100.5, "end": 128.0},
    {"label": "bridge", "start": 128.0, "end": 145.3},
    {"label": "chorus", "start": 145.3, "end": 172.8},
    {"label": "outro", "start": 172.8, "end": 195.0},
]
```

Models unloaded, GPU freed.

### 3.2 Build Section Tags

```python
section_list_str, section_tags = _build_section_list(segments)
```

Converts raw segments to numbered tags:
```
section_tags = ["[Intro]", "[Verse 1]", "[Chorus 1]", "[Verse 2]", 
                "[Chorus 2]", "[Bridge]", "[Chorus 3]", "[Outro]"]
```

Verses and choruses are numbered. Other sections are not.

### 3.3 Load Qwen 2.5-Omni 7B

**GPU:** ~14GB

```python
qwen_model_obj = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-Omni-7B",
    device_map="auto",
    torch_dtype=torch.bfloat16,
)
```

Loaded **once** and reused for all subsequent inference passes (5+ calls).

### 3.4 Per-Stem Analysis (3 Qwen calls)

Each Demucs stem is fed to Qwen individually for targeted description:

**Drums prompt:**
> "Describe the drum patterns in this track: kick pattern, snare pattern, cymbal pattern, groove feel. Be specific about which beats they hit. One paragraph, plain text."

**Bass prompt:**
> "Describe the bass in this track: instrument type, tone, playing technique, note patterns. One paragraph, plain text."

**Other (melodic) prompt:**
> "Describe the melodic instruments in this track: what instruments, their timbre, effects, playing style. One paragraph, plain text."

Each response is truncated to 2 sentences maximum.

**Example outputs:**
```
drums: "Tight kick on beats 1 and 3 with snare on 2 and 4, open hi-hat eighth notes."
bass: "Round electric bass with fingerpicked quarter notes, clean tone, slight warmth."
other: "Overdriven electric guitar with palm-muted verses and open power chords in choruses."
```

### 3.5 Caption Generation (1 Qwen call)

Qwen listens to the **full original mix** and generates a production caption. The prompt includes:
- Metadata (BPM, key, time signature)
- Per-stem descriptions from 3.4 as context
- Instructions to describe each instrument's tone, effects, and playing style (50-80 words)

```python
caption_prompt = (
    "Music metadata: 112 BPM, key of Eb Major, 4/4 time\n\n"
    "The original track has these instruments:\n"
    "- drums: Tight kick on beats 1 and 3...\n"
    "- bass: Round electric bass...\n"
    "- other: Overdriven electric guitar...\n\n"
    "Write a music production caption describing this instrumental track. "
    "Describe EACH instrument in a separate sentence..."
)
```

**Output example:**
```
"Overdriven electric guitar with warm crunch tone playing power chords.
Tight compressed drums with punchy kick and bright snare."
```

Hallucination markers are stripped (truncate if Qwen outputs "Human provided..." etc.)

### 3.6 Per-Section Hints (N Qwen calls + programmatic analysis)

For each section, a **hybrid** approach combining DSP and neural analysis:

#### Programmatic: Drums/Bass Presence (CPU, instant)

```python
drums_rms = 20 * np.log10(np.sqrt(np.mean(drums_section**2)) + 1e-10)
drums_active = drums_rms > -40  # Above -40dB threshold
```

Simple RMS energy check per section. No LLM needed for "is bass/drums playing?"

#### Solo Detection: 3-Vote Ensemble

For the "other" stem in each section, three independent methods vote:

| Vote | Method | Threshold | Detects |
|------|--------|-----------|---------|
| 1 | Spectral flux | > 0.03 | High melodic movement (solo has rapid pitch changes) |
| 2 | Relative energy | > 1.5× average | Solo sections louder than accompaniment |
| 3 | Qwen listening | "solo"/"lead"/"melody" in response | Semantic understanding |

**Qwen chunk prompt:**
> "This is a 15-second clip of a melodic instrument (isolated from drums and bass). Name the instrument and how it's being played. Reply in EXACTLY 2-3 words."

**Ensemble rule:** ALL 3 must agree for "solo" override. Otherwise, Qwen's 2-3 word description is used directly (e.g., "Guitar Power Chords", "Synth Arpeggios").

#### Combined Hint Assembly

```python
parts = []
if drums_active: parts.append("drums")
if bass_active: parts.append("bass")
if other_hint: parts.append(other_hint)  # e.g., "Guitar Power Chords"
hint = " ".join(parts)  # → "drums bass Guitar Power Chords"
```

### 3.7 Genre Selection (1 Qwen call)

Qwen is shown the original caption + BPM/key, then asked to pick ONE number from 10 genre options:

```
This song has: Overdriven electric guitar with warm crunch tone...
It's 112 BPM in Eb Major.

I want to remix this into a different genre. Which of these genres would 
work best at this tempo and match the song's energy? Pick ONE number:

1. Synthwave (electronic drums, analog synth bass, synth lead)
2. Funk (tight funk drums, slap bass, clavinet)
3. Industrial (aggressive electronic drums, distorted bass, distorted synth)
4. Neo-Soul (live drums with ghost notes, warm electric bass, rhodes piano)
5. Disco (disco drums, disco bass, string section)
6. Trip-Hop (breakbeat drums, sub bass, atmospheric synth pad)
7. Latin Rock (latin percussion, fingerpicked bass, nylon guitar)
8. Electro-Funk (808 drum machine, moog bass, vocoder synth)
9. Post-Punk (aggressive drums, driving bass, angular synth)
10. Cinematic (orchestral percussion, cello, brass section)

Reply with ONLY the number. Nothing else.
```

This is a **classification task** — far more reliable than asking Qwen to freely name instruments.

**`parse_genre_choice()`** maps the response to pre-validated instruments:
```python
_GENRE_INSTRUMENT_MAP = {
    "6": {"drums": "Breakbeat Drums", "bass": "Sub Bass", "melodic": "Analog Synth Pad"},
    ...
}
```

Falls back to keyword matching if not a clean number, then random selection from energetic genres as last resort.

### 3.8 Unload Qwen (~14GB freed)

### 3.9 Build Template Caption

**File:** `rearrangement.py:build_caption_from_instruments()`

From the validated instrument dict, a **rich performance description** is generated via template matching. The templates include genre-specific playing style instructions:

```python
# Input: {"drums": "Breakbeat Drums", "bass": "Sub Bass", "melodic": "Analog Synth Pad"}
# "Analog Synth Pad" matches the "pad" template branch:

caption = (
    "lush analog synth pad with slow evolving textures, "
    "wide stereo chorusing, subtle filter movement, "
    "harmonic swells building through choruses. "
    "Breakbeat Drums with tight groove and dynamic fills, "
    "deep sub bass locking with kick drum pattern. "
    "Professional production, punchy mix, wide stereo field."
)
```

Template branches exist for: synth leads, pads, Rhodes/piano, guitar, brass, strings, distorted instruments, and a generic fallback.

### 3.10 Translate Temporal Script

**File:** `rearrangement.py:translate_temporal_script()`

The original per-section hints (which contain original instrument names like "Guitar Power Chords") are translated into **energy-only tags**. The caption already declares the new instruments — the lyrics just need to say when to change energy.

**Translation logic:**
- Solo detected → "explosive solo"
- "peak energy" / "full energy" in hint → "high energy"
- "building" → "building energy"
- "dropping" → "fading"
- No explicit energy → infer from section type:
  - Intro → "building, atmospheric"
  - Chorus (position 0) → "high energy, powerful"
  - Chorus (position 1+) → "high energy, driving"
  - Chorus (position 3+) → "peak energy, anthemic"
  - Verse → "moderate energy"
  - Bridge → "building energy"
  - Outro (last) → "fade out"

**Output:**
```
[Intro - building, atmospheric]
[Instrumental]

[Verse 1 - moderate energy]
[Instrumental]

[Chorus 1 - high energy, powerful]
[Instrumental]

[Verse 2 - moderate energy]
[Instrumental]

[Chorus 2 - high energy, driving]
[Instrumental]

[Bridge - building energy]
[Instrumental]

[Chorus 3 - peak energy, anthemic]
[Instrumental]

[Outro - fade out]
[Instrumental]
```

### 3.11 Final Output

```python
StructureTimelineResult(
    caption="lush analog synth pad with slow evolving textures...",
    original_caption="Overdriven electric guitar with warm crunch tone...",
    lyrics="[Intro - building, atmospheric]\n[Instrumental]\n\n...",
    segments=[{"label": "intro", "start": 0.0, "end": 17.2}, ...],
    hints=["soft", "drums bass Guitar Arpeggios", ...],
)
```

---

## Step 4: Cover Generation (Two-Pass)

**File:** `pipeline_runner.py` (cover_genre branch), `cover_genre.py`, `timbre_reference.py`
**GPU:** ~18GB

### 4.1 Load DiT Model

```python
handler = AceStepHandler()
handler.initialize_service(project_root=".", config_path="acestep-v15-xl-sft")
```

The xl-sft model (4B parameters) is loaded. This model supports CFG (guidance_scale > 1.0), which is essential for v48's approach.

### 4.2 Load ScragVAE

Replaces the default VAE decoder with a community fine-tuned version:
- +38% high-frequency energy
- +29dB dynamic range
- Crisper transients, less muffled output

The encoder is unchanged — all existing models remain compatible.

### 4.3 Load LoRA (digital-acoustic)

A concept slider LoRA at scale 0.7 that pushes output toward more polished/produced sound. Auto-downloaded from HuggingFace on first use.

### 4.4 Pass 1: Generate Timbre Reference

**File:** `timbre_reference.py`

```python
ref_audio = generate_timbre_reference(
    handler=handler,
    caption=refined_caption,      # Template caption
    duration=full_song_duration,  # Match full song length
    lyrics=timeline.lyrics,       # Temporal script
    bpm=metadata["bpm"],
    keyscale=metadata["keyscale"],
    guidance_scale=9.0,
    inference_steps=28,
    shift=3.0,
)
```

**What happens:**
1. `target_wavs = torch.zeros(1, 2, samples)` — Silence (no source audio)
2. `task_type = "text2music"` — Generate freely from caption
3. The DiT generates what the target instruments sound like at this BPM/key, following the temporal energy arc from the lyrics
4. Output: raw audio tensor `[2, samples]` at 48kHz — kept in GPU memory

**Purpose:** Gives the DiT a concrete audio example of the target sound palette, rather than relying solely on text description.

### 4.5 Pass 2: Generate Cover

**File:** `cover_genre.py:generate_cover_genre()`

```python
audio_np = generate_cover_genre(
    handler=handler,
    source_audio_path=stems.instrumental,  # Full instrumental
    caption=refined_caption,               # Template caption
    lyrics=timeline.lyrics,                # Temporal script
    hints=None,                            # NO bass hints
    bpm=metadata["bpm"],
    keyscale=metadata["keyscale"],
    audio_cover_strength=0.4,              # Low = major genre shift allowed
    cover_noise_strength=0.0,              # No additional noise
    guidance_scale=9.0,                    # High = strong caption adherence
    inference_steps=28,
    shift=3.0,
    refer_audios=[[ref_audio]],            # Pass 1 output
)
```

**What happens internally:**

1. **Source audio encoding:** The instrumental stem is loaded, resampled to 48kHz, and passed as `target_wavs`
2. **VAE encode source:** The handler encodes `target_wavs` through the VAE encoder → source latents `[1, T, 64]`
3. **VAE encode reference:** The timbre reference audio is encoded through the same VAE → reference latents
4. **Timbre encoder:** A transformer within the DiT processes the reference latents:
   - Projects from 64-dim → hidden_size
   - Prepends a CLS token for aggregation
   - Self-attention extracts a global timbre signature
   - Output: condensed timbre embeddings representing "what this should sound like"
5. **Conditioning assembly:** Three sources are packed into one sequence:
   - Lyric embeddings (temporal structure)
   - Timbre embeddings (from reference audio)
   - Text embeddings (from caption via text encoder)
6. **Noisy starting point:** `audio_cover_strength=0.4` means the denoising starts from `0.4 × source_latent + 0.6 × noise`. This preserves rough temporal structure but allows major deviation.
7. **ODE diffusion:** 28 steps of denoising, guided by:
   - CFG at scale 9.0 pushing toward the caption's instruments
   - Timbre encoder grounding the target sound concretely
   - Source latent providing structural skeleton
8. **Decode:** Latents → ScragVAE decoder → audio
9. **Normalize:** Peak normalization to 0.891 (-1dB headroom)

### 4.6 Why No Hints in V48

The v48 commit message explains:

> "Remove 25Hz hints from cover_genre (hints + CFG fight, chroma dropped to 0.35)"

When combined:
- **Hints** guide latent-level chord voicings (pull toward original harmony)
- **High guidance_scale** pushes toward genre caption (pull toward new instruments)

These conflict at the latent level. The model can't satisfy both — result is worse than either alone.

V48's approach: let `audio_cover_strength=0.4` handle whatever chord preservation it can, and let guidance_scale handle the genre shift. Accept the trade-off (chroma 0.30-0.49 instead of 0.60+).

---

## Step 5: Post-Generation QC

**File:** `pipeline_runner.py`, `post_gen_qc.py`, `cover_genre.py:repaint_failing_sections()`
**GPU:** Reuses loaded handler

### 5.1 Per-Bar Analysis

```python
qc_results = analyze_generated_audio(
    audio_path=str(out_path),
    bpm=metadata["bpm"],
    key=metadata["keyscale"],
    segments=timeline.segments,
    original_audio_path=str(stems.instrumental),
)
```

Divides audio into bars (using BPM) and checks each bar for:

| Check | Method | Threshold |
|-------|--------|-----------|
| Bass dropout | Low-frequency RMS energy | < -45dB where original has bass |
| Out-of-key notes | Chroma vs key scale | Energy of non-scale pitch classes > 2× in-scale |
| Wrong chord root | Chroma argmax comparison | Root doesn't match original bar's root |

### 5.2 Section-Level Pass/Fail

Bars are grouped by section. A section fails if it has too many failing bars (bass dropout in 3+ bars, or any chord mismatch bars).

### 5.3 Repaint Failing Sections

For each failing section:

```python
repaint_failing_sections(
    handler=handler,
    audio_path=current_audio,
    failing_sections=failing,
    caption=refined_caption,
    source_audio_path=stems.instrumental,
    hints=None,                              # Still no hints
    audio_cover_strength=0.6,                # HIGHER than initial (0.4 + 0.2)
    cover_noise_strength=0.0,
    guidance_scale=9.0,
    inference_steps=28,
    shift=3.0,
    max_attempts=3,
)
```

**Key:** Repaint uses `audio_cover_strength=0.6` — higher than the initial generation's 0.4. This locks chords more tightly on retry while maintaining some genre character. Different random seed gives a different result.

The repaint uses ACE-Step's **native repaint task** (`task_type="repaint"`):
- Only the failing time range is regenerated
- Rest of the song is preserved unchanged
- Uses `repainting_start` and `repainting_end` parameters

Up to 3 attempts per section. If all fail, the original section is kept unchanged.

---

## Step 6: DAW Mix

**File:** `mix_daw.py`
**GPU:** None (CPU, Pedalboard library)

### 6.1 Vocal Chain

| Effect | Parameters | Purpose |
|--------|-----------|---------|
| High-pass filter | 80 Hz | Remove rumble/plosives |
| Presence boost | +3dB at 3kHz, Q=1.5 | Vocal clarity |
| De-ess | -3dB at 7kHz, Q=2.0 | Tame sibilance |
| Compressor | 3:1, threshold -18dB | Even dynamics |
| Plate reverb | Short decay | Space/depth |

### 6.2 Instrumental Chain

| Effect | Parameters | Purpose |
|--------|-----------|---------|
| High-pass filter | 30 Hz | Sub cleanup |
| Low-shelf warmth | +2dB at 200Hz | Body |
| Vocal notch | -2.5dB at 3.5kHz, Q=1.5 | Carve space for vocals |
| Compressor | 2.5:1, threshold -15dB | Consistency |
| Gain | +5dB | AI instrumentals are typically quieter |

### 6.3 Dynamic Balance

Measures LUFS loudness of the original mix's instrumental vs the generated instrumental, applies gain compensation to match.

### 6.4 Master Bus

| Effect | Parameters | Purpose |
|--------|-----------|---------|
| Bus compressor | 1.5:1, threshold -8dB | Glue (light — vocals already mastered) |
| Limiter | -1dB ceiling | Prevent clipping |

### 6.5 Output

`final_cover_daw.wav` — The complete cover: original vocals + new genre-shifted instrumental, professionally mixed and loudness-matched.

---

## Key Technical Decisions in V48

### Why No LM Caption Refinement

> "LM ignores genre intent, rewrites to random genres"

The 5Hz LM's `format_sample_from_input` was designed for general-purpose caption formatting, not genre-locked rewriting. When given "Trip-Hop with breakbeat drums...", it would often output something like "indie rock with acoustic guitar..." — completely ignoring the genre constraint.

**Solution:** Use a deterministic template (`build_caption_from_instruments`) that guarantees genre-specific performance descriptions.

### Why Template Captions Over Qwen Captions

> "Remove Qwen caption generation (output was inconsistent/rambling)"

Qwen's free-form instrument descriptions were unreliable:
- Sometimes rambling multi-paragraph responses
- Often included genre labels (DiT doesn't respond well to genre words)
- Quality varied wildly between runs

**Solution:** A fixed template that embeds performance instructions (how to play, not just what instrument). The template was validated across multiple songs.

### Why Full Song Duration for Timbre Reference

The timbre reference uses `duration=full_song_duration` and receives the full temporal script. This means the reference isn't just a static 30-second clip — it has the same energy arc as the target song (quiet verses, building bridges, loud choruses).

The timbre encoder sees this dynamic range and extracts not just "what instruments" but "how those instruments change intensity across the song structure."

### Why `audio_cover_strength=0.4`

| Value | Effect |
|-------|--------|
| 0.1-0.3 | Near-total freedom — model barely follows original structure |
| **0.4** | **V48: Strong genre shift, loose structure preservation** |
| 0.5-0.7 | Moderate shift — recognizable structure, different sound |
| 0.8-1.0 | Faithful cover — same structure, subtle tonal changes |

At 0.4, the starting latent is 40% source + 60% noise. The model preserves gross timing (where verse/chorus boundaries are, overall song length) but is free to rewrite harmony, rhythm patterns, and instrumentation. This is the sweet spot for "recognizably the same song but in a different genre."

---

## Output Structure

```
data/output/song_v48/
├── stems/
│   ├── melband/
│   │   ├── *_(Vocals)_*.wav          # Preserved original vocals
│   │   └── *_(Instrumental)_*.wav    # Structural source for cover
│   └── demucs/
│       └── htdemucs_ft/song/
│           ├── drums.mp3              # Per-section energy analysis
│           ├── bass.mp3               # Per-section energy analysis
│           └── other.mp3              # Melodic instrument identification
├── generated/
│   ├── cover_genre.flac               # Initial cover generation
│   └── repainted.flac                 # After QC fixes (if any)
└── final_cover_daw.wav                # Final mix (vocals + new instrumental)
```

---

## Hardware Requirements

Tested on **ml.g5.16xlarge** (1x A10G 24GB, 64GB RAM):

| Step | Peak VRAM | Duration |
|------|:---------:|:--------:|
| Mel-Band RoFormer | ~2 GB | 30s |
| Demucs v4 | ~2 GB | 60s |
| SongFormer (3 models) | ~4 GB | 5s |
| Qwen 2.5-Omni 7B | ~14 GB | 3-5 min |
| ACE-Step xl-sft + LoRA + ScragVAE | ~18 GB | |
| — Timbre reference (text2music) | (shared) | ~60s |
| — Cover generation | (shared) | ~60s |
| — QC repaint (per section) | (shared) | ~30s each |
| Pedalboard DAW mix | 0 GB (CPU) | 5s |

**Total pipeline time:** ~10-15 min per song (depending on QC retries).

Models load/unload sequentially. Peak is ~18GB during DiT generation.

---

## Comparison with Other Modes

| | V48 (cover_genre) | Semantic (default) | Remix Blend |
|---|---|---|---|
| Genre shift | Strong | None | Moderate |
| Chord accuracy | Low (0.30-0.49) | High (0.58-0.81) | Medium (0.54) |
| Mechanism | acs=0.4 + CFG | 25Hz bass hints | LM codes + bass blend |
| Hints | None | Bass stem 25Hz | Blended (alpha) |
| Source | Full instrumental | Full instrumental | Full instrumental |
| Caption source | Template | Qwen + original instruments | Template |
| Best for | Radical genre changes | Faithful covers | Creative remixes |
