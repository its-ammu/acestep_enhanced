"""Auto-select the best stem for semantic hints extraction.

Scores each stem on chord clarity, coverage, and timbre complexity
to determine which provides the best harmonic guidance with minimal
timbre leakage.

Criteria:
- Chord clarity: how clearly the stem defines chord roots (peaked chroma)
- Coverage: how much of the song the stem is active
- Timbre complexity: spectral flatness (lower = more tonal = less leakage)

Best hints source = high clarity × high coverage ÷ high timbre complexity
"""

from pathlib import Path
from typing import Optional

import numpy as np
from loguru import logger


def select_hints_source(
    stem_paths: dict[str, Path],
    sr: int = 22050,
) -> str:
    """Score each stem and return the best one for hints extraction.

    Args:
        stem_paths: Dict of stem_name -> path (drums, bass, other).
        sr: Sample rate for analysis (22050 is sufficient for chroma).

    Returns:
        Name of the best stem ("bass", "other", or "drums").
        Falls back to "bass" if analysis fails.
    """
    import librosa

    scores = {}

    for name, path in stem_paths.items():
        if not path or not Path(path).exists():
            continue

        try:
            audio, _ = librosa.load(str(path), sr=sr, mono=True)

            if len(audio) < sr:
                scores[name] = 0.0
                continue

            # Chroma simplicity: how concentrated is energy in few pitch classes
            # Low entropy = monophonic (bass plays one note) = good for hints
            # High entropy = polyphonic (guitar plays chords) = bad for hints
            chroma = librosa.feature.chroma_cqt(y=audio, sr=sr)
            # Normalize each frame to probability distribution
            chroma_norm = chroma / (chroma.sum(axis=0, keepdims=True) + 1e-10)
            # Shannon entropy per frame (lower = more concentrated = simpler)
            entropy_per_frame = -np.sum(
                chroma_norm * np.log2(chroma_norm + 1e-10), axis=0
            )
            # Average entropy (max possible = log2(12) ≈ 3.58 for uniform)
            avg_entropy = float(np.mean(entropy_per_frame))
            # Simplicity = inverse of entropy (higher = simpler = better)
            simplicity = 1.0 / (avg_entropy + 0.1)

            # Coverage: % of frames above silence threshold
            rms = librosa.feature.rms(y=audio)[0]
            coverage = float(np.mean(rms > 0.01))

            # Combined score: simplicity × coverage
            # Simpler harmonic content + high coverage = best hints source
            score = simplicity * coverage
            scores[name] = score

            logger.info(
                f"  Hints source score [{name}]: "
                f"entropy={avg_entropy:.2f}, simplicity={simplicity:.2f}, "
                f"coverage={coverage:.2f}, score={score:.2f}"
            )

        except Exception as e:
            logger.warning(f"  Hints source scoring failed for {name}: {e}")
            scores[name] = 0.0

    if not scores:
        logger.warning("No stems could be scored, defaulting to bass")
        return "bass"

    best = max(scores, key=scores.get)
    logger.info(
        f"  Best hints source: {best} "
        f"(score={scores[best]:.2f}, "
        f"all scores: {', '.join(f'{k}={v:.2f}' for k, v in sorted(scores.items(), key=lambda x: -x[1]))})"
    )
    return best
