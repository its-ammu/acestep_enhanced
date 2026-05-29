# Cover Pipeline

Automated cover song generation: preserve original vocals, regenerate the instrumental with AI, with per-section quality control.

## Quick Start

```bash
# 1. Base environment
uv sync

# 2. Install additional dependencies
uv pip install --python .venv/bin/python librosa essentia pedalboard audio-separator onnxruntime pytorch_wavelets PyWavelets "setuptools<70"

# 3. Run the pipeline
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
    cover_noise_strength=0.15,        # Creativity level (0.15=creative, 0.3=safe)
    lora_path='digital-acoustic',     # LoRA slider (None to disable)
    lora_scale=0.7,                   # LoRA strength (0.3-0.7 recommended)
    use_scrag_vae=True,               # ScragVAE for better decode quality
    quality_gate=False,               # Legacy stem-level gate (disabled)
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
| Garbled audio output | Restart instance (GPU state corruption from OOM) |
| `onnxruntime-gpu` installed | `uv pip install onnxruntime --force-reinstall` (CPU only) |
| Root disk full | Set HF_HOME to SageMaker volume |
| "No module named librosa" | Run deps install or `ensure_dependencies()` |
| All sections fail QC | Bad seed — re-run (random seed each time) |
| Solo section empty | Pipeline auto-detects and repaints at cns=0.35 |

## Scoring

After pipeline completes, scores are printed:
- **chroma**: Chord correlation vs original (0-1, higher = more accurate chords)
- **rhythm**: Rhythm pattern correlation (0-1, higher = better timing)

Good results: chroma > 0.75, rhythm > 0.85

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
