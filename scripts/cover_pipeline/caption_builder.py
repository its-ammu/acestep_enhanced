"""Caption and lyrics builder from analysis results.

Constructs the optimal caption and structural lyrics for ACE-Step
Cover mode based on the full analysis pipeline output.
"""

from .audio_analysis import AnalysisResult


def build_caption(analysis: AnalysisResult) -> str:
    """Build an optimized caption for ACE-Step DiT from analysis results.

    Combines instrument descriptions, production characteristics,
    chord hints, and texture keywords into a single caption string
    optimized for xl-sft Cover mode.

    Args:
        analysis: Complete AnalysisResult from the analysis pipeline.

    Returns:
        Caption string ready for ACE-Step.
    """
    parts = []

    # Instrument description (from LLM)
    if analysis.instrument_description:
        parts.append(analysis.instrument_description)

    # Production description (from LLM)
    if analysis.production_description:
        parts.append(analysis.production_description)

    # Chord movement hint
    if analysis.chord_summary:
        # Get the most common progression pattern
        chord_hint = _build_chord_hint(analysis)
        if chord_hint:
            parts.append(chord_hint)

    # Key hint
    if analysis.key:
        parts.append(f"in {analysis.key}")

    # Texture keywords as reinforcement
    if analysis.timbre_keywords:
        # Add keywords not already present in the description
        existing_text = " ".join(parts).lower()
        new_keywords = [kw for kw in analysis.timbre_keywords if kw.lower() not in existing_text]
        if new_keywords:
            parts.append(", ".join(new_keywords[:5]))

    caption = ", ".join(parts)

    # Clean up: remove double commas, extra spaces
    caption = caption.replace(",,", ",").replace("  ", " ").strip().rstrip(",")

    return caption


def build_lyrics(analysis: AnalysisResult) -> str:
    """Build structural lyrics (instrumental) from analysis results.

    Creates section tags with energy/style hints based on the
    detected song structure. All sections marked [Instrumental]
    since we're keeping original vocals.

    Args:
        analysis: Complete AnalysisResult from the analysis pipeline.

    Returns:
        Lyrics string with structure tags for ACE-Step.
    """
    if not analysis.segments:
        # Fallback: generic structure
        return (
            "[Intro]\n[Instrumental]\n\n"
            "[Verse 1]\n[Instrumental]\n\n"
            "[Chorus]\n[Instrumental]\n\n"
            "[Verse 2]\n[Instrumental]\n\n"
            "[Bridge]\n[Instrumental]\n\n"
            "[Chorus]\n[Instrumental]\n\n"
            "[Outro]\n[Instrumental]"
        )

    lines = []
    for i, seg in enumerate(analysis.segments):
        # Map segment labels to standard tags with energy hints
        tag = _format_section_tag(seg.label, i, len(analysis.segments))
        lines.append(tag)
        lines.append("[Instrumental]")
        lines.append("")  # Blank line between sections

    return "\n".join(lines).strip()


def _build_chord_hint(analysis: AnalysisResult) -> str:
    """Build a chord progression hint for the caption."""
    if not analysis.chord_summary:
        return ""

    # Find chorus or verse progression (most representative)
    for section_key, progression in analysis.chord_summary.items():
        section_lower = section_key.lower()
        if "chorus" in section_lower:
            return f"chord progression: {progression} in chorus"

    # Fallback to first available section
    first_key = next(iter(analysis.chord_summary))
    first_prog = analysis.chord_summary[first_key]
    return f"chord movement: {first_prog}"


def _format_section_tag(label: str, index: int, total: int) -> str:
    """Format a section label into an ACE-Step structure tag with energy hint."""
    label_lower = label.lower().strip()

    # Map common labels to tags with appropriate energy hints
    tag_map = {
        "intro": "[Intro - ambient]",
        "verse": "[Verse]",
        "verse 1": "[Verse 1]",
        "verse 2": "[Verse 2]",
        "verse 3": "[Verse 3]",
        "pre-chorus": "[Pre-Chorus - building]",
        "prechorus": "[Pre-Chorus - building]",
        "chorus": "[Chorus - powerful]",
        "bridge": "[Bridge]",
        "outro": "[Outro - fade]",
        "instrumental": "[Instrumental]",
        "solo": "[Guitar Solo]",
        "break": "[Breakdown]",
        "breakdown": "[Breakdown]",
        "drop": "[Drop - high energy]",
        "build": "[Build]",
        "interlude": "[Interlude]",
    }

    # Direct match
    if label_lower in tag_map:
        return tag_map[label_lower]

    # Partial match
    for key, tag in tag_map.items():
        if key in label_lower:
            return tag

    # Default: use the label as-is
    return f"[{label.title()}]"


def build_metadata(analysis: AnalysisResult) -> dict:
    """Extract metadata parameters for ACE-Step from analysis.

    Returns:
        Dict with bpm, keyscale, timesignature, duration.
    """
    return {
        "bpm": int(round(analysis.bpm)) if analysis.bpm > 0 else None,
        "keyscale": analysis.key or None,
        "timesignature": analysis.time_signature or "4/4",
        "duration": int(round(analysis.duration)) if analysis.duration > 0 else None,
    }
