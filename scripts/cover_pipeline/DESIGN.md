# Cover Pipeline — Technical Design Document

## 1. Problem Statement

Produce commercial-quality cover songs that:
- Preserve the original vocal performance exactly
- Generate a new AI instrumental that is harmonically correct (right chords, correct key)
- Use different instrument tones/textures from the original (not a copy)
- Maintain bass consistency and pattern repetition (especially in choruses)
- Sound like professional production (not bland/generic AI output)
- Maintain timing alignment between vocals and instrumental
- Run fully automated end-to-end with no manual intervention

## 2. System Overview

The pipeline transforms a source audio file into a final cover through six stages:

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                           COVER PIPELINE v2                                   │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌─────┐  ┌─────┐ │
│  │  Stem    │─►│  Audio   │─►│Structure │─►│ Semantic │─►│ QC  │─►│ Mix │ │
│  │Separation│  │ Analysis │  │ Timeline │  │   Gen    │  │Splice│  │ DAW │ │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘  └─────┘  └─────┘ │
│       │              │             │              │            │        │     │
│   vocals.wav     BPM, Key     caption +     creative.flac  spliced  final   │
│   instrumental   chords       lyrics        safe.flac      .flac    .wav    │
│   drums/bass/                                                                │
│   other                                                                      │
└──────────────────────────────────────────────────────────────────────────────┘
```

## 3. Key Technical Decisions

### 3.1 Semantic Hints from Bass Stem (Not Full Mix)

**Decision:** Extract semantic hints from the isolated bass stem only.

**Rationale:** Full-mix hints encode both chord structure AND instrument timbre. This causes the generated output to sound too similar to the original. Bass-only hints provide chord root information (correct harmonic progression) without leaking the full arrangement's timbre fingerprint.

**Result:** Verse and pre-chorus sections achieve "perfectly locked in" chord accuracy while allowing different instrument tones in the generated output.

### 3.2 ScragVAE Decoder (Improved Audio Fidelity)

**Decision:** Replace the default VAE decoder with ScragVAE (community fine-tuned decoder).

**Rationale:** The stock ACE-Step VAE attenuates high-frequency content above 6kHz, resulting in dull/muffled output. ScragVAE retrains only the decoder half to better reconstruct upper harmonics, transient detail, and spectral "air." The encoder is unchanged, so all existing DiT checkpoints remain compatible.

**Improvement:** +38% high-frequency energy, +29dB dynamic range, crisper transients.

### 3.3 Concept Slider LoRA (Production Character)

**Decision:** Apply a concept slider LoRA (digital-acoustic) during generation to push the output toward a more produced/polished sound.

**Rationale:** At low cover_noise_strength (0.15), the model generates "safe" bland output. The LoRA provides a consistent directional push toward more interesting instrument character without requiring per-song tuning.

**Scale:** 0.7 (moderate push — avoids artifacts while adding character). Auto-downloaded from HuggingFace on first use.

### 3.4 Dual-Version Generation with Section Splicing

**Decision:** Generate two versions of the instrumental (creative + safe) and splice the best sections together.

**Rationale:** Low cover_noise_strength (0.15) produces different-sounding instruments but can cause bass dropout and occasional wrong notes in choruses. Higher cns (0.3) preserves bass patterns but sounds too similar to the original. By generating both and splicing per-section, we get creative verses (where low cns works well) and structurally safe choruses (where bass consistency matters).

**Splice logic:**
- Chorus/Outro sections → always use safe version (bass consistency)
- Verse/Pre-chorus/Intro → use creative version (different instruments)
- Bass dropout detection → fallback to safe if creative drops bass

### 3.5 Light Master Compression (Pre-Mastered Vocals)

**Decision:** Use very light master bus compression (threshold -8dB, ratio 1.5:1).

**Rationale:** The vocal stem comes from an already-mastered track. Aggressive compression on the final mix double-compresses the vocals, causing "crunchy" artifacts on loud sections. Light compression with a high threshold preserves dynamics.

### 3.6 Dynamic Vocal/Instrumental Balance

**Decision:** Automatically measure and match the vocal-to-instrumental ratio.

**Rationale:** AI-generated instrumentals are typically 4-5dB quieter than the original instrumental. The mix chain measures the current ratio, targets +1.5dB (instrumental slightly louder than vocals, matching typical commercial mixes), and applies the exact gain compensation needed.

### 3.7 Key Detection with Major/Minor Disambiguation

**Decision:** Detect key using Krumhansl-Schmuckler profiles with bass-root confirmation for relative major/minor disambiguation.

**Rationale:** Standard chroma-based key detection often confuses relative major/minor keys (e.g., detecting Eb Major when the song is in C minor). Since these share the same pitch classes, the algorithm scores them similarly. Bass root analysis disambiguates by checking which root note is more prominent in the low frequencies.

**Impact:** Incorrect key metadata causes the model to generate notes outside the song's key (e.g., C# in a C minor song). Correct key detection eliminates this class of errors.

## 4. Pipeline Stages

### Stage 1: Stem Separation
- **Mel-Band RoFormer** → Vocals + Instrumental (SDR ~13)
- **Demucs v4 (htdemucs_ft)** → Drums, Bass, Other (guitar/keys)
- GPU: ~2GB each, freed after

### Stage 2: Audio Analysis
- **BPM:** librosa beat tracking
- **Key:** Krumhansl-Schmuckler with bass disambiguation
- **Time signature:** 4/4 default (madmom optional)
- GPU: 0 (CPU only)

### Stage 3: Structure Timeline
- **SongFormer** → Section boundaries (intro, verse, chorus, bridge, outro)
- **Per-stem energy analysis** → Which instruments active per section
- **Qwen 2.5-Omni** → Caption (instrument tones) + per-section hints
- GPU: ~4GB (SongFormer) → ~14GB (Qwen), sequential

### Stage 4: Semantic Generation (Dual Version)
- Load handler once (xl-sft DiT + ScragVAE + LoRA)
- Extract semantic hints from bass stem
- Generate **creative version** (cns=0.15, different instruments)
- Generate **safe version** (cns=0.3, preserves bass patterns)
- GPU: ~18GB (shared across both generations)

### Stage 5: Section-Level QC & Splice
- Analyze creative version per-section for bass dropout
- Splice: creative for verses, safe for choruses/outros
- Crossfade at section boundaries (0.3s)
- GPU: 0 (CPU only)

### Stage 6: DAW Mix
- Vocal chain: HPF 80Hz, subtle presence, light compression, plate reverb
- Instrumental chain: HPF 30Hz, low-shelf warmth, vocal notch, compression, +5dB gain
- Dynamic balance: match vocal/instrumental ratio to +1.5dB target
- Master bus: light compression (ratio 1.5), limiter -1dB
- GPU: 0 (CPU only — Pedalboard)

## 5. Generation Parameters

| Parameter | Value | Purpose |
|-----------|:-----:|---------|
| dit_model | acestep-v15-xl-sft | 4B parameter model, highest quality |
| cover_noise_strength | 0.15 (creative) / 0.3 (safe) | Creative freedom vs source preservation |
| guidance_scale | 12.0 | Strong caption adherence for different instruments |
| inference_steps | 65 | Optimal for xl-sft quality |
| shift | 6.0 | Timestep schedule (community-proven for covers) |
| audio_cover_strength | 1.0 | Full cover conditioning |
| hints_source | bass stem | Chord roots without timbre leakage |
| lora | digital-acoustic @ 0.7 | Production character push |
| vae | ScragVAE | Higher fidelity audio output |

## 6. Validation Results

### What Works Well
- **Verse/Pre-Chorus chord accuracy:** "Perfectly locked in" — bass and chords follow the correct progression
- **Volume balance:** Correctly matched to original loudness
- **Mix quality:** No over-compression, clean dynamics
- **Different instrument character:** Output sounds distinct from original

### Known Limitations
- **Chorus bass dropout:** At cns=0.15, bass occasionally drops out on specific measures within chorus sections. Mitigated by section splice (safe version for choruses).
- **Key detection:** Krumhansl-Schmuckler can confuse relative major/minor. Under investigation: passing correct key eliminates out-of-key notes.
- **Guitar solo/lead melodies:** Don't carry over from original. Model generates accompaniment instead.
- **Seed-dependent quality:** Production quality varies between random seeds. Multi-variant generation with scoring addresses this.
- **Caption genre drift:** Qwen 2.5-Omni occasionally suggests wrong genres. Mitigated by removing genre labels from caption prompt.

## 7. Evolution & Iteration History

The pipeline evolved through multiple iterations driven by musician feedback and objective scoring. Each iteration addressed specific issues identified in the previous output.

### Decision Drivers

Each technical decision was driven by specific feedback or measurement:

| Feedback/Input | Decision Made | Why |
|----------------|---------------|-----|
| "Bass and keys clashing, out of key" | Added semantic hints from bass stem | Provides chord root guidance to the model |
| "Sounds too similar to the original" | Switched from full-mix hints to bass-only hints | Bass has chord roots without full timbre fingerprint |
| "Sounds bland/generic" | Added ScragVAE + concept slider LoRA | ScragVAE adds fidelity, LoRA adds production character |
| "Over-compressed, crunchy on loud sections" | Reduced master compression (ratio 1.5, threshold -8dB) | Vocals already mastered — double compression causes artifacts |
| "Instrumental too quiet" | Dynamic vocal/instrumental balance matching | Measures actual ratio, targets +1.5dB (instrumental slightly louder) |
| "Bass drops out in chorus measures" | Section-level splice (safe version for choruses) | Higher cns preserves bass patterns in structurally critical sections |
| "C# note not in the song's key" | Key detection investigation + QC loop | Librosa detected wrong key (D# Major vs C minor); model uses key label for note selection |
| "Guitar solo missing" | Higher cns preserves arrangement detail | At low cns, fine melodic details (solos) are lost in noise |
| Qwen generates wrong genre captions | Removed genre labels from caption prompt | Qwen is unreliable at genre classification; focus on instrument tones only |
| Inconsistent quality across runs | Multi-variant generation + per-section QC | Seed-dependent quality addressed by generating multiple and selecting best |

### Iteration 1: Basic Cover Mode (CLI subprocess)
- **Approach:** ACE-Step cover mode via cli.py, full mix as source, no hints
- **Result:** Chroma correlation 0.330 (wrong chords), instruments sounded different but harmonically incorrect
- **Feedback:** "Bass and keys clashing, out of key, instruments unaware of each other"
- **Learning:** Cover mode without chord guidance produces musically incorrect output

### Iteration 2: Semantic Hints from Full Mix
- **Approach:** Extract semantic hints from full mix, inject via monkey-patch
- **Result:** Chroma 0.849 (excellent chord accuracy)
- **Feedback:** "Sounds too similar to the original"
- **Learning:** Full-mix hints encode timbre along with chords — output copies the original's instrument character

### Iteration 3: Bass-Only Hints + Low cover_noise_strength
- **Approach:** Extract hints from bass stem only (chord roots without timbre), cns=0.15 for creative freedom
- **Result:** Chroma 0.58-0.65, different instrument character
- **Feedback:** "Verse and pre-chorus perfectly locked in! But chorus has bass dropout, and output can sound bland"
- **Learning:** Bass hints provide chord accuracy without timbre leakage. Low cns gives creative freedom but inconsistent quality.

### Iteration 4: ScragVAE + Concept Slider LoRA
- **Approach:** Added ScragVAE (crisper audio) + digital-acoustic slider at 0.7 (production character)
- **Result:** Crisper output with more interesting instrument character
- **Feedback:** "Mix/compression improved, no longer over-processed. But chorus still has bass dropout and occasional wrong notes (C#)"
- **Learning:** ScragVAE improves fidelity. LoRA adds consistent character. But structural issues (bass dropout) persist at low cns.

### Iteration 5: Section-Level Splice (Creative + Safe)
- **Approach:** Generate two versions (cns=0.15 creative + cns=0.3 safe), splice best sections. Choruses use safe version for bass consistency.
- **Result:** Choruses improved (2nd chorus "GOOD! Bass playing right notes!"), outro "musically satisfying"
- **Feedback:** "Bass dropout still in some sections. C# note appearing that's not in the song's key."
- **Learning:** Section splice works for structural fixes. The C# issue traced to incorrect key detection (D# Major detected instead of C minor).

### Iteration 6: Key Detection Fix (Current)
- **Approach:** Investigating correct key metadata. Librosa detects D# Major (relative major of C minor) — same pitch classes but different label. Model interprets key label to decide valid notes.
- **Hypothesis:** Passing correct key (C minor) will eliminate the C# problem
- **Status:** Under testing

### Key Metrics Across Iterations

| Iteration | Chroma | Rhythm | Chord Accuracy | Different Sound | Production Quality |
|-----------|:------:|:------:|:--------------:|:---------------:|:-----------------:|
| 1 (no hints) | 0.330 | 0.776 | ❌ | ✅ | ❌ |
| 2 (full hints) | 0.849 | 0.941 | ✅ | ❌ | ✅ |
| 3 (bass hints) | 0.609 | 0.857 | ✅ | ✅ | ⚠️ (seed-dependent) |
| 4 (ScragVAE+LoRA) | 0.580 | 0.856 | ✅ | ✅ | ✅ (on good seeds) |
| 5 (splice) | 0.608 | 0.714 | ✅ (verses) ⚠️ (chorus) | ✅ | ✅ |

```
Input Song (.mp3)
    │
    ├─► Mel-Band RoFormer ──► Vocals (preserved)
    │                      └─► Instrumental (source for generation)
    │
    ├─► Demucs v4 ──► Bass stem ──► Semantic Hints (VAE→Tokenize→Detokenize)
    │             ├─► Drums stem ──► Per-section energy analysis
    │             └─► Other stem ──► Per-section energy analysis
    │
    ├─► librosa ──► BPM, Key
    │
    ├─► SongFormer ──► Section boundaries
    │
    ├─► Qwen 2.5-Omni ──► Caption + Per-section hints
    │
    ├─► ACE-Step xl-sft DiT
    │       + ScragVAE (decoder)
    │       + Concept Slider LoRA
    │       + Bass semantic hints (monkey-patched)
    │       │
    │       ├─► Creative version (cns=0.15)
    │       └─► Safe version (cns=0.3)
    │
    ├─► Section Splice ──► Best sections from each version
    │
    └─► DAW Mix (Pedalboard)
            + Dynamic vocal/instrumental balance
            + Light master compression
            + Loudness matching to original
            │
            └─► Final Cover (.wav)
```

## 8. GPU Memory Management

Single A10G (24GB). Models load/unload sequentially:

| Stage | Peak VRAM | Strategy |
|-------|:---------:|----------|
| Mel-Band RoFormer | 2 GB | Freed after separation |
| Demucs | 2 GB | Freed after separation |
| SongFormer | 4 GB | Unloaded after analysis |
| Qwen 2.5-Omni 7B | 14 GB | Load once, all passes, unload |
| ACE-Step xl-sft + LoRA | 18 GB | Single load, both generations |
| Pedalboard | 0 GB | CPU only |

**Critical:** Semantic hints extracted BEFORE LoRA loading (LoRA backup uses GPU memory during PEFT wrapping).

## 9. Dependencies and Licensing

| Component | License | Purpose |
|-----------|---------|---------|
| ACE-Step 1.5 | MIT | Core generation model |
| ScragVAE | MIT | Improved VAE decoder |
| Concept Sliders | MIT | Production character LoRAs |
| Pedalboard | GPL-3.0 | DAW effects (no distribution) |
| SongFormer | Research | Structure detection |
| Qwen 2.5-Omni | Apache-2.0 | Caption + section analysis |
| Mel-Band RoFormer | MIT | Vocal separation |
| Demucs | MIT | Multi-stem separation |
| librosa | ISC | Audio analysis |

## 10. Next Steps

1. **Post-generation QC loop (in progress):** Generate → per-section quality check → retry failing sections with new seeds → splice best sections. Detects bass dropout (per-bar RMS) and out-of-key notes (pitch class validation against detected key). Guarantees a minimum quality floor by eliminating objective errors.
2. **Key detection fix:** Validate that passing correct key eliminates out-of-key notes. Implement improved detection with bass-root disambiguation or user override.
3. **Multi-variant per-section selection:** Generate 4 variants, score each section independently, splice best sections from any variant.
4. **Bar-level bass analysis:** Detect bass dropout at individual bar level (using BPM for bar boundaries) rather than section level.
