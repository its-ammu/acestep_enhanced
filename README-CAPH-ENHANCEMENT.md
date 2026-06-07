# CAPH-Steered Remix Pipeline

## Cross-Attention Pitch-to-Harmony (CAPH) Alignment for ACE-Step

---

## What Problem Does This Solve?

When you remix a song (e.g. shift "pop ballad" → "lo-fi hip hop"), ACE-Step generates a new
instrumental that fits the target genre but often **drifts harmonically** — it may be in a
different key or use dissonant chord progressions that clash with your original vocal when
mixed together.

**CAPH Alignment** solves this by continuously steering the AI-generated instrumental *toward*
notes that are musically consonant with your original vocal's pitch during the diffusion
process itself.

---

## Workflow

```
[Song.mp3]
    │
    ▼
[Melbound Reformer / any stem separator]
    ├── vocal.wav   ──────────────────────────────────────────────────┐
    │                                                                  │
    └── instrumental.wav ──► Source Audio in ACE-Step Remix mode      │
                                                                       ▼
                                                         Upload to "Vocal Stem" field
                                                         → extract pitch (f0)
                                                         → vocal chroma [1, T, 12]
                                                                       │
                         ACE-Step Flow-Edit Sampling Loop              │
                         ┌─────────────────────────────────────┐       │
                         │  Each step:                          │◄──────┘
                         │  1. zt += dt·(V_target - V_source)  │  genre shift
                         │  2. zt -= λ·∇CDL(zt, vocal_chroma)  │  harmonic correction
                         └─────────────────────────────────────┘
                                          │
                                          ▼
                              New Instrumental (genre-shifted
                               AND harmonically aligned)
                                          │
                                          ▼
                         Mix with original vocal.wav in your DAW
                              → Final Remix ✅
```

---

## The Full System — Layer by Layer

### Layer 1 — Music Theory Engine (`acestep/models/common/caph_aligner.py`)

The `CAPHAligner` is a small PyTorch module that encodes Western music theory. It has two jobs:

**Job 1 — Measure harmonic dissonance via the Music Theory Penalty Matrix (M)**

A hardcoded 12×12 table of interval dissonance values based on Western tonal harmony:

| Interval | Semitones | Penalty | Musical Reason |
|----------|-----------|---------|----------------|
| Unison | 0 | 0.0 | Perfect consonance — same note |
| Minor 3rd | 3 | 0.0 | Core chord tone — blend beautifully |
| Major 3rd | 4 | 0.0 | Core chord tone — blend beautifully |
| Perfect 5th | 7 | 0.0 | Most stable interval in Western music |
| Major 6th | 9 | 0.0 | Sweet consonance |
| Perfect 4th | 5 | 0.4 | Mildly consonant, context-dependent |
| Minor 6th | 8 | 0.4 | Softer consonance |
| Major 2nd | 2 | 0.7 | Mild tension, needs resolution |
| Minor 7th | 10 | 0.7 | Jazz-flavored tension |
| Minor 2nd | 1 | 1.0 | Harshest dissonance — semitone clash |
| Tritone | 6 | 1.0 | The "devil's interval" — maximum tension |
| Major 7th | 11 | 1.0 | Leading tone clash |

The matrix is symmetric (M[i,j] = M[j,i]) and registered as a non-learned buffer —
it never changes during training or inference.

**Job 2 — Compute the Chord Distance Loss (CDL)**

```
CDL = mean over T frames of: Σᵢⱼ ( inst_chroma[i] × M[i,j] × vocal_chroma[j] )
```

- `inst_chroma` = softmax of a linear projection of the DiT latent → 12-bin pitch distribution
- `vocal_chroma` = extracted from the uploaded vocal stem
- CDL is a single differentiable scalar: **low = consonant, high = dissonant**

### Layer 2 — Vocal Pitch Extraction (`acestep/models/common/vocal_f0_extraction.py`)

Converts the uploaded vocal stem audio into a chroma tensor:

```
vocal.wav
  → mono waveform
  → torchaudio YIN pitch detection (10ms hop, 50–2000 Hz range)
  → f0 (Hz) sequence
  → MIDI note via 12·log₂(f0/440) + 69
  → pitch class (0–11) modulo 12
  → Gaussian soft assignment (σ=0.5) to 12 chroma bins
  → [1, T, 12] chroma tensor
```

- Unvoiced frames (f0 ≤ 10 Hz: silence, breaths, consonants) produce **zero chroma** — harmless
- Gaussian soft assignment means nearby semitones also activate (more musically natural than hard one-hot)

### Layer 3 — Gradient Steering (`acestep/models/common/caph_steering.py`)

At each diffusion step, *after* the normal V_delta genre shift is applied, one gradient
correction step is taken:

```python
# Normal Flow-Edit step (genre shift):
zt_edit += dt × (V_target_avg - V_source_avg)

# CAPH gradient steering (NEW):
with torch.enable_grad():
    zt_steer = zt_edit.detach().requires_grad_(True)
    _, cdl_loss = caph_aligner(zt_steer, vocal_chroma)
    cdl_grad = autograd.grad(cdl_loss, zt_steer)[0]
zt_edit = zt_edit - λ × cdl_grad       # λ = cdl_guidance_scale slider value
```

**Safety properties:**
- `torch.enable_grad()` is scoped to exactly this block — rest of loop stays `@no_grad()`
- `detach().requires_grad_(True)` prevents graph accumulation across steps
- `λ = 0.0` (default) = completely disabled, zero computation overhead
- If vocal chroma has mismatched temporal length vs. latent, it is interpolated automatically

### Layer 4 — Diffusion Loop (`acestep/models/common/flow_edit.py` and `flow_edit_pipeline.py`)

- Steering only fires inside the edit window `[n_min, n_max]` — same window as V_delta
- `CAPHAligner` is instantiated lazily in `flow_edit_pipeline.py` only when both
  `vocal_chroma` is provided and `cdl_guidance_scale > 0`
- `latent_dim` is inferred from `src_latents.shape[-1]` — no model config lookup needed

### Layer 5 — Service Layer Threading

The `vocal_chroma` tensor and `cdl_guidance_scale` float are threaded through 5 files:

```
generation_progress.py        ← extracts chroma from uploaded vocal stem
    ↓ GenerationParams.flow_edit_vocal_chroma
inference.py                  ← passes to handler via dit_generate_kwargs
    ↓ flow_edit_cdl_guidance_scale
generate_music.py             ← orchestration
    ↓
generate_music_execute.py     ← service call wrapper
    ↓
service_generate.py           ← builds flow_edit_ctx dict
                                 [batch expand: [1,T,12] → [B,T,12] for batch > 1]
    ↓
service_generate_flow_edit.py ← passes to flowedit_generate_audio
    ↓
flow_edit_pipeline.py         ← instantiates CAPHAligner, passes to sampling loop
    ↓
flow_edit.py                  ← applies gradient steering each step
```

### Layer 6 — Gradio UI

New controls added inside the **Edit** accordion panel (Remix mode):

```
> Retake & Edit
    [ ] Edit ✓
      ┌──────────────────────────────────────────────────────────┐
      │ [Copy current → source]                                   │
      │                                                           │
      │ source caption: "original pop ballad piano"               │
      │ source lyrics: ...                                        │
      │                                                           │
      │ n_min [ 0.1 ]   n_max [ 0.9 ]   n_avg [ 1 ]             │
      │                                                           │
      │ 🎤 Vocal Stem (for harmonic alignment)         ← NEW     │
      │    Upload the separated vocal stem here                   │
      │    Its pitch steers the remix to stay harmonic            │
      │                                                           │
      │ Vocal Harmonic Alignment (CDL): [0.0 ━━━━━━ 2.0] ← NEW  │
      │    0=off; 0.1–0.5 subtle; 1.0+ strong                    │
      └──────────────────────────────────────────────────────────┘
```

**Browser feedback toasts on Generate:**

| Situation | Toast |
|-----------|-------|
| Vocal stem uploaded, voices detected | `✅ Vocal Harmonic Alignment active — 847/1200 voiced frames detected` |
| CDL slider > 0 but no vocal stem uploaded | `⚠️ CDL enabled but no Vocal Stem uploaded. Steering disabled.` |
| All frames unvoiced (wrong file?) | `⚠️ Zero voiced frames detected — check you uploaded the vocal stem, not the instrumental` |
| torchaudio extraction error | `⚠️ Vocal stem pitch extraction failed — harmonic alignment disabled. Error: ...` |

---

## Complete File List

| File | Phase | What It Does |
|------|-------|-------------|
| `acestep/models/common/caph_aligner.py` | 1 | Music Theory Penalty Matrix + CAPHAligner + f0_to_chroma |
| `acestep/models/common/caph_aligner_test.py` | 1 | 6 unit tests (shapes, gradients, penalty matrix symmetry) |
| `acestep/models/common/caph_steering.py` | 1 | Lazy aligner creation + gradient steering step |
| `acestep/models/common/vocal_f0_extraction.py` | 1 | Audio file → f0 → chroma tensor |
| `acestep/models/common/caph_remix_workflow_test.py` | 2 | 15 integration tests for the full remix workflow |
| `acestep/models/common/flow_edit.py` | 2 | CDL steering hooked into diffusion loop |
| `acestep/models/common/flow_edit_pipeline.py` | 2 | Lazy CAPHAligner instantiation, passes to loop |
| `acestep/core/generation/handler/service_generate_flow_edit.py` | 2 | vocal_chroma + cdl_guidance_scale into flowedit call |
| `acestep/core/generation/handler/service_generate.py` | 2 | Batch expand + flow_edit_ctx dict |
| `acestep/core/generation/handler/generate_music.py` | 2 | flow_edit_cdl_guidance_scale param |
| `acestep/core/generation/handler/generate_music_execute.py` | 2 | Thread to service_generate |
| `acestep/inference.py` | 2 | GenerationParams.flow_edit_vocal_chroma + dit_generate_kwargs |
| `acestep/ui/gradio/interfaces/generation_tab_variation_morph_controls.py` | 2 | Vocal Stem upload + CDL slider |
| `acestep/ui/gradio/events/wiring/generation_run_wiring.py` | 2 | Wire new inputs to generate button |
| `acestep/ui/gradio/events/results/batch_management_wrapper.py` | 2 | Pass vocal_stem_audio through |
| `acestep/ui/gradio/events/results/generation_progress.py` | 2 | _extract_vocal_chroma_safe + UI toasts |

---

## Usage Guide

### Step 1 — Separate your song
Use any stem separator (Melbound Reformer, Demucs, etc.) to produce:
- `vocal.wav` — the vocal track
- `instrumental.wav` — the instrumental backing

### Step 2 — Configure ACE-Step (Remix mode)
1. Set mode to **Remix**
2. Upload `instrumental.wav` → **Source Audio** field
3. Enable the **Edit** checkbox
4. Fill in:
   - **Source caption**: describe the original style (e.g. `"slow pop ballad, piano, emotional"`)
   - **Target caption**: describe the desired remix style (e.g. `"lo-fi hip hop beat, vinyl texture, chill, 85 BPM"`)
5. Set **n_min=0.1**, **n_max=0.9**, **n_avg=1**
6. Upload `vocal.wav` → **Vocal Stem** field
7. Set **Vocal Harmonic Alignment (CDL)** slider to **0.3**

### Step 3 — Generate
Click **Generate**. You should see:
```
✅ Vocal Harmonic Alignment active — N/M voiced frames detected in vocal stem.
```

### Step 4 — Mix
Mix the generated instrumental output with your original `vocal.wav` in your DAW.

---

## CDL Guidance Scale (λ) Reference

| λ value | Effect | When to use |
|---------|--------|-------------|
| `0.0` | Off — pure genre shift | Default; backward compatible |
| `0.1` | Very subtle | Barely perceptible correction |
| `0.3` | **Recommended start** | Noticeable alignment without constraining genre creativity |
| `0.5` | Moderate | Strong alignment, slight creative constraint |
| `1.0` | Strong | Heavy harmonic correction |
| `1.5+` | Very strong | May over-constrain the genre shift |

---

## GPU Deployment Notes

### VRAM
The `torch.enable_grad()` steering block allocates a small computation graph per step.
- Typical overhead: ~12 MB extra VRAM for a standard latent
- On 12 GB GPUs: test with a short clip (30s) first when using `cdl_guidance_scale > 0`

### Half precision (float16 / bfloat16)
All tests run in float32 on CPU. The aligner is moved to the model's dtype automatically.
If you see no effect at `λ=0.3`, increase to `0.5–1.0` — float16 gradient underflow can
occasionally reduce the effective steering strength.

### Vocal quality
The f0 detector works best on clean separated vocals. Heavily auto-tuned or whispered vocals
may produce sparse voiced frames. The UI toast shows `N/M voiced frames` — if N < 20% of M,
consider increasing `λ` or using a cleaner separation.

---

## Test Summary

```
$ uv run python -m unittest \
    acestep.models.common.caph_aligner_test \
    acestep.models.common.caph_remix_workflow_test \
    acestep.models.common.flow_edit_test \
    acestep.models.common.flow_edit_pipeline_test \
    acestep.core.generation.handler.service_generate_flow_edit_test \
    acestep.ui.gradio.events.wiring.decomposition_contract_generation_test \
    acestep.ui.gradio.events.wiring.context_test

Ran 49 tests in 0.092s — OK ✅
```

| Test Suite | Count | What It Verifies |
|------------|-------|-----------------|
| `caph_aligner_test` | 6 | Penalty matrix symmetry, chroma shapes, gradient flow |
| `caph_remix_workflow_test` | 15 | Full workflow: lazy creation, steering reduces CDL, UI toasts |
| `flow_edit_test` | 7 | Diffusion loop backward compat, APG, EMA |
| `flow_edit_pipeline_test` | 1 | LM hints forwarding |
| `service_generate_flow_edit_test` | 9 | Cover/text2music dispatch, source tokenization |
| `decomposition_contract_generation_test` | 7 | Wiring AST contracts |
| `context_test` | 4 | Component list ordering |
