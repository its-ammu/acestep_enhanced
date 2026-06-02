# Cover Pipeline

Automated cover song generation: preserve original vocals, regenerate the instrumental with AI, with per-section quality control.

## Quick Start

```bash
# 1. Base environment
uv sync

# 2. Fix triton (uv sync installs 3.6 which produces garbled audio with torch 2.10)
uv pip install --python .venv/bin/python "triton==3.3.1" --force-reinstall

# 3. Run the pipeline (deps.py handles all other dependencies automatically)
uv run --no-sync python -c "
from scripts.cover_pipeline.pipeline_runner import run_pipeline, PipelineConfig

cfg = PipelineConfig(
    input_song='data/your_song.mp3',
    output_dir='data/output/your_song',
)
run_pipeline(cfg)
"
```

Output: `data/output/your_song/final_cover_daw.wav`

> **Note:** All additional dependencies (librosa, demucs, pedalboard, pytorch_wavelets, etc.) are installed automatically by `deps.py` on the first run. You do NOT need to install them manually.

## Architecture

```
Original Song
    │
    ├─► [Mel-Band RoFormer] ──► vocals (preserved) + instrumental (source)
    ├─► [Demucs v4] ──► drums/bass/other (per-stem analysis)
    │
    ├─► [Essentia] ──► Key detection (e.g., "Eb Major")
    ├─► [librosa] ──► BPM detection
    ├─► [SongFormer] ──► Section boundaries (full mix input)
    ├─► [Qwen 2.5-Omni] ──► Per-section instrument + style hints
    │
    │    ┌─────────────────────────────────────────────┐
    │    │ Semantic Generation (per-section QC)        │
    │    │                                             │
    │    │  1. Extract semantic hints from bass stem   │
    │    │  2. Generate "creative" (cns=0.15)          │
    │    │  3. Generate "safe" (cns=0.3)               │
    │    │  4. Splice: creative for verses,            │
    │    │     safe for choruses/outros                │
    │    │  5. Solo repaint (cns=0.35 for inst)        │
    │    │  6. QC: per-bar chord root + key check      │
    │    │  7. Repaint failing sections (3 attempts)   │
    │    │  8. Safe fallback if repaint fails           │
    │    └─────────────────────────────────────────────┘
    │
    ├─► [ScragVAE] ──► Higher fidelity audio decode
    ├─► [LoRA slider] ──► digital-acoustic at 0.7 (optional)
    │
    └─► [Pedalboard DAW Mix] ──► Final cover (vocals + instrumental, loudness-matched)
```

## Pipeline Configuration

```python
PipelineConfig(
    input_song='data/song.mp3',       # Any audio format
    output_dir='data/output/song',    # Output directory
    generation_mode='semantic',       # 'semantic', 'audio_codes', or 'cli'
    cover_noise_strength=0.15,        # Creativity level (0.15=creative, 0.3=safe)
    lora_path='digital-acoustic',     # LoRA slider (None to disable)
    lora_scale=0.7,                   # LoRA strength (0.3-0.7 recommended)
    use_scrag_vae=True,               # ScragVAE for better decode quality
    quality_gate=False,               # Legacy stem-level gate (disabled)
)
```

### Generation Modes

| Mode | Description | Timbre Lock? | Chord Accuracy | Genre Shift? |
|------|-------------|:---:|:---:|:---:|
| `semantic` | 25Hz hints from bass stem (default) | Yes (entangled) | High | No |
| `remix_blend` | LM codes + bass hints blended | Partial | Medium (0.54) | Yes |
| `cover_genre` | Cover task + genre caption + CFG | No | Low-Medium (0.30) | Yes |
| `complete_remix` | Complete task with vocal+bass source | No | Low (0.22-0.53) | Yes |
| `audio_codes` | 5Hz codes as loose chord guide | No | Medium | Partial |
| `text2music_free` | Pure text, no source conditioning | No | N/A | Yes |
| `cli` | Subprocess generation (legacy) | Depends on params | Varies | No |

**`remix_blend` mode** (best balance of genre shift + chord accuracy):
- 4B LM generates creative 5Hz codes via CoT planning
- Bass stem provides chord-accurate 25Hz hints
- Blends at configurable alpha (0.3=more chords, 0.5=more creativity)
- xl-turbo renders from blended hints (8 steps, ~15s)
- Best result: chroma=0.54, rhythm=0.86 at alpha=0.4

```python
PipelineConfig(
    generation_mode='remix_blend',
    dit_model='acestep-v15-xl-turbo',
    blend_alpha=0.4,            # 0.0=all bass, 1.0=all LM
    lm_temperature=0.7,         # LM creativity
    inference_steps=8,
    shift=3.0,
    cover_noise_strength=0.15,
)
```

**`cover_genre` mode** (strongest genre shift, uses cover task as intended):
- Cover mode with `audio_cover_strength` controlling structure preservation
- Requires xl-sft model (supports CFG/guidance for caption adherence)
- Source = full instrumental (structural skeleton)
- Genre comes from caption + high guidance_scale
- Optional timbre reference clip for target sound
- Best for: radical genre changes where some chord drift is acceptable

```python
PipelineConfig(
    generation_mode='cover_genre',
    dit_model='acestep-v15-xl-sft',
    audio_cover_strength=0.4,   # 0.3-0.5 for genre shift, 0.7+ for chord accuracy
    cover_noise_strength=0.0,   # Not needed with audio_cover_strength
    guidance_scale=9.0,         # High = follow genre caption strongly
    inference_steps=28,
    shift=3.0,
    use_timbre_reference=True,  # Generate target-genre audio reference
    refine_caption_lm=True,     # LM enhances caption
)
```

**`complete_remix` mode** (experimental — complete task):
- Feeds vocal+bass mix as source for "complete" task
- Model generates new instruments conditioned on what it hears
- Does NOT enforce chord following (model composes independently)
- Use only with xl-base model

```python
PipelineConfig(
    generation_mode='complete_remix',
    dit_model='acestep-v15-xl-base',
    bass_mix_db=-6.0,
    include_vocal_in_source=True,
    guidance_scale=7.0,
    inference_steps=50,
)
```

## QC System

The pipeline includes automated quality control that:

1. **Detects out-of-key notes** — compares generated chroma per bar against the song's key
2. **Detects wrong chord roots** — compares chord root per bar against the original instrumental
3. **Detects empty solos** — measures spectral flux vs original; repaints at higher cns if melody is missing
4. **Repaints failing sections** — generates new versions with different seeds, only applies if the replacement passes QC
5. **Falls back to safe version** — if repaint fails, tries the safe (cns=0.3) version for that section
6. **Never degrades** — if nothing passes, keeps the original section unchanged

## Key Parameters

| Parameter | Default | Effect |
|-----------|---------|--------|
| `cover_noise_strength` | 0.15 | Lower = more creative/different, higher = more faithful |
| `lora_path` | digital-acoustic | Concept slider (None to disable) |
| `lora_scale` | 0.7 | Slider strength (higher = more effect) |
| `use_scrag_vae` | True | Improved VAE decoder for crisper audio |
| `guidance_scale` | 12.0 | Caption adherence |
| `inference_steps` | 65 | Diffusion steps |
| `shift` | 6.0 | Timestep shift (compositional quality) |

## Output Structure

```
data/output/your_song/
├── stems/
│   ├── melband/          # Mel-Band RoFormer (vocals + instrumental)
│   └── demucs/           # Demucs v4 (drums/bass/other)
├── generated/
│   ├── creative.flac     # Low cns version
│   ├── safe.flac         # High cns version
│   ├── semantic_cover.flac        # Spliced version
│   └── semantic_cover_qc_fixed.flac  # After QC fixes
└── final_cover_daw.wav   # Final mix (vocals + QC-fixed instrumental)
```

## Hardware Requirements

Tested on **ml.g5.16xlarge** (1x A10G 24GB, 64GB RAM):

| Step | Peak VRAM | Duration |
|------|:---------:|:--------:|
| Stem separation | ~4 GB | 90s |
| SongFormer | ~4 GB | 5s |
| Qwen 2.5-Omni 7B | ~14 GB | 3-5 min |
| ACE-Step generation (x2) | ~18 GB | 3 min |
| QC repaint (per section) | ~18 GB | 60s each |

Total pipeline time: ~15-25 min per song (depending on QC retries).

## Environment Notes

- **Python**: 3.11-3.12
- **PyTorch**: 2.10.0+cu128 (Linux x86_64)
- **Do NOT install** `onnxruntime-gpu` — causes garbled diffusion output
- **Set env vars** before running:
  ```bash
  export HF_HOME=/home/ec2-user/SageMaker/.cache/huggingface
  export TRANSFORMERS_CACHE=/home/ec2-user/SageMaker/.cache/huggingface
  export UV_CACHE_DIR=/home/ec2-user/SageMaker/.cache/uv
  export TORCHAUDIO_BACKEND=soundfile
  ```

## Troubleshooting

| Issue | Fix |
|-------|-----|
| Garbled audio output | Check triton version: `triton>=3.4` causes garbled output with torch 2.10. Fix: `uv pip install --python .venv/bin/python "triton==3.3.1" --force-reinstall` |
| Garbled after `uv sync` | `uv sync` upgrades triton to 3.6+. The pipeline auto-fixes this via `deps.py`, but if running manually: pin triton first |
| GPU state corruption (all outputs garbled) | Full stop → start from AWS console (not sudo reboot) |
| `onnxruntime-gpu` installed | `uv pip install onnxruntime --force-reinstall` (CPU only) |
| Root disk full | Set HF_HOME to SageMaker volume |
| "No module named librosa" | Run deps install or `ensure_dependencies()` |
| All sections fail QC | Bad seed — re-run (random seed each time) |
| Solo section empty | Pipeline auto-detects and repaints at cns=0.35 |

### Critical: Never Run `uv sync` Without Pinning Triton

`uv sync` resolves the full dependency tree and will upgrade `triton` to 3.6+, which is incompatible with torch 2.10.0+cu128 and produces garbled audio. The pipeline's `ensure_dependencies()` auto-detects and fixes this, but if you run generation manually after `uv sync`:

```bash
uv pip install --python .venv/bin/python "triton==3.3.1" --force-reinstall
```

## Scoring

After pipeline completes, scores are printed:
- **chroma**: Chord correlation vs original (0-1, higher = more accurate chords)
- **rhythm**: Rhythm pattern correlation (0-1, higher = better timing)

Good results: chroma > 0.75, rhythm > 0.85 (for faithful covers)
Genre-shifted remixes: chroma 0.50-0.70 is acceptable (chord substitutions are valid)

### Experimental Results (v44-v48)

| Version | Mode | Key Settings | Chroma | Rhythm | Notes |
|---------|------|--------------|:------:|:------:|-------|
| v44 | remix_blend | alpha=1.0 (pure LM) | 0.175 | 0.024 | LM codes alone = wrong chords |
| v45 | remix_blend | alpha=0.4, temp=0.7 | 0.539 | 0.859 | Best balance, creative |
| v46 | complete_remix | sft, vocal+bass, bass=-6dB | 0.529 | 0.122 | Music ducks during vocals |
| v47 | complete_remix | xl-base, vocal+bass | 0.224 | 0.166 | Model ignores chords entirely |
| v48 | cover_genre | sft, acs=0.4, guidance=9 | 0.301 | 0.666 | Genre came through, chords wrong |

### Key Findings

1. **25Hz bass hints** (remix_blend) give the best chord accuracy for genre-shifted output
2. **Cover mode** at low `audio_cover_strength` produces strong genre shift but poor chords
3. **Complete task** does NOT use source audio as harmonic constraint — it composes independently
4. **xl-turbo** does not support CFG (guidance_scale has no effect) — use xl-sft for caption control
5. **LM-generated captions** are more natural/detailed than templates, DiT responds better to them
6. **Multi-seed selection** is the most effective way to improve chord accuracy (seeds vary 0.45-0.70)
7. **Temporal scripts** (energy tags) effectively transfer the original song's dynamic arc

### Running with Scoring

Full command that runs the pipeline and prints chroma/rhythm scores:

```bash
uv run --no-sync python -c "
from scripts.cover_pipeline.pipeline_runner import run_pipeline, PipelineConfig
cfg = PipelineConfig(
    input_song='data/your_song.mp3',
    output_dir='data/output/your_song_v23',
    cover_noise_strength=0.15,
    lora_path='digital-acoustic',
    lora_scale=0.7,
    use_scrag_vae=True,
    quality_gate=False,
)
result = run_pipeline(cfg)
if result:
    import librosa, numpy as np, glob
    from pathlib import Path
    from scripts.cover_pipeline.stem_quality_gate import _chroma_correlation, _rhythm_correlation
    gen_dir = Path('data/output/your_song_v23/generated')
    for c in ['semantic_cover_qc_fixed.flac', 'semantic_cover.flac', 'creative.flac']:
        p = gen_dir / c
        if p.exists(): gen_path = p; break
    else: gen_path = result
    orig_candidates = glob.glob('data/output/your_song_v23/stems/melband/*Instrumental*')
    if orig_candidates:
        orig = librosa.load(orig_candidates[0], sr=22050, mono=True)[0]
        gen = librosa.load(str(gen_path), sr=22050, mono=True)[0]
        min_len = min(len(orig), len(gen))
        chroma = _chroma_correlation(orig[:min_len], gen[:min_len])
        rhythm = _rhythm_correlation(orig[:min_len], gen[:min_len])
        print(f'=== Score ===')
        print(f'chroma={chroma:.3f}, rhythm={rhythm:.3f}')
"
```

### v23 Baseline Scores

| Song | chroma | rhythm |
|------|:------:|:------:|
| Just The Way It Is | 0.811 | 0.923 |

## Licenses

| Component | License | Commercial Use |
|-----------|---------|:-:|
| ACE-Step 1.5 | MIT | ✅ |
| ScragVAE | Check repo | ⚠️ Verify |
| Pedalboard | GPL-3.0 | ✅ (internal tool use) |
| SongFormer | Research | ⚠️ Verify |
| Qwen 2.5-Omni | Apache-2.0 | ✅ |
| Mel-Band RoFormer | MIT | ✅ |
| Demucs | MIT | ✅ |
