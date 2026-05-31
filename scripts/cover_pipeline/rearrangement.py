"""Re-arrangement caption generation for cover differentiation.

Takes the original instrument description and proposes a completely different
arrangement that complements the vocal. This is the creative differentiation
lever — instead of rendering the same instruments at low fidelity,
we render DIFFERENT instruments at higher fidelity.

Flow:
1. Qwen proposes a re-arrangement (different instrument family)
2. Validate and sanitize instrument names against known vocabulary
3. Parse instrument names from the re-arrangement caption
4. Translate per-section hints: swap original instrument names for new ones
"""

import random
import re
from typing import Optional

from loguru import logger


# Known instruments that ACE-Step renders well (curated from training data)
_VALID_INSTRUMENTS = {
    # Drums/Percussion
    "drums": [
        "Brushed Jazz Drums", "Electronic Drum Machine", "Live Rock Drums",
        "Acoustic Drums", "Tight Funk Drums", "Lo-Fi Drum Machine",
        "Orchestral Percussion", "Bongo Congas", "Cajon Percussion",
    ],
    # Bass
    "bass": [
        "Upright Bass", "Synth Bass", "Electric Bass", "Fretless Bass",
        "Sub Bass", "Slap Bass", "Fingerpicked Bass", "Analog Bass",
        "Moog Bass", "Acoustic Bass",
    ],
    # Melodic
    "melodic": [
        "Rhodes Piano", "Grand Piano", "Nylon Guitar", "Analog Synth Pad",
        "Wurlitzer Piano", "Vibraphone", "Marimba", "Accordion",
        "Pedal Steel Guitar", "Lap Steel Guitar", "Harmonica",
        "Trumpet", "Saxophone", "Clarinet", "Violin", "Cello",
        "Flute", "Organ", "Harpsichord", "Mandolin", "Banjo",
        "Sitar", "Kalimba", "Music Box", "Harp", "Dulcimer",
    ],
}

# Flat set of all valid instrument words for validation
_ALL_VALID_WORDS = set()
for category in _VALID_INSTRUMENTS.values():
    for name in category:
        for word in name.lower().split():
            _ALL_VALID_WORDS.add(word)

# Words that identify the original instrument (to be stripped from hints)
_ORIGINAL_INSTRUMENT_WORDS = {
    "guitar", "electric", "acoustic", "synth", "synthesizer",
    "piano", "keyboard", "organ", "strings", "brass",
    "trumpet", "saxophone", "violin", "cello", "flute",
    "mandolin", "banjo", "harmonica", "rhodes", "wurlitzer",
    "nylon", "steel", "distorted", "overdriven", "clean",
    "fender", "vibraphone", "marimba", "section",
}


def build_rearrangement_prompt(
    original_caption: str,
    vocal_profile: str,
    metadata: Optional[dict] = None,
) -> str:
    """Build the Qwen prompt for re-arrangement caption generation.

    Args:
        original_caption: Description of original instruments.
        vocal_profile: Vocal character description.
        metadata: Dict with bpm, keyscale.

    Returns:
        Complete prompt string for Qwen.
    """
    bpm = metadata.get("bpm", "unknown") if metadata else "unknown"
    keyscale = metadata.get("keyscale", "unknown") if metadata else "unknown"

    prompt = (
        "You are a music producer creating a cover version. "
        "Choose COMPLETELY DIFFERENT instruments from the original.\n\n"
        f"ORIGINAL: {original_caption}\n"
        f"SONG: {bpm} BPM, key of {keyscale}\n\n"
        "Write EXACTLY 3 lines. Each line: category, colon, instrument name "
        "(2-3 words only). No descriptions, no parentheses, no extra text.\n\n"
        "DRUMS: [2-3 word drum instrument]\n"
        "BASS: [2-3 word bass instrument]\n"
        "MELODIC: [2-3 word melodic instrument]\n\n"
        "Examples:\n"
        "DRUMS: Brushed Jazz Drums\n"
        "BASS: Upright Bass\n"
        "MELODIC: Rhodes Piano\n\n"
        "DRUMS: Electronic Drum Machine\n"
        "BASS: Synth Bass\n"
        "MELODIC: Nylon Guitar\n\n"
        "DRUMS: Orchestral Percussion\n"
        "BASS: Fretless Bass\n"
        "MELODIC: Vibraphone\n\n"
        "Now write 3 lines (different from the original):"
    )

    return prompt


def sanitize_rearrangement(raw_caption: str) -> Optional[dict[str, str]]:
    """Validate and sanitize Qwen's rearrangement output.

    Parses the structured output, validates each instrument name against
    the known vocabulary, and falls back to random selection if invalid.

    Args:
        raw_caption: Raw text from Qwen.

    Returns:
        Dict with "drums", "bass", "melodic" keys, or None if completely unusable.
    """
    parsed = parse_instruments_from_caption(raw_caption)

    # Validate each instrument against known vocabulary
    validated = {}
    for category, name in parsed.items():
        if _is_valid_instrument(name, category):
            validated[category] = name
        else:
            logger.warning(
                f"Rearrangement: invalid {category} instrument '{name}', "
                f"selecting random fallback"
            )
            validated[category] = random.choice(_VALID_INSTRUMENTS[category])

    # Ensure all three categories are present
    for category in ("drums", "bass", "melodic"):
        if not validated.get(category):
            validated[category] = random.choice(_VALID_INSTRUMENTS[category])
            logger.warning(
                f"Rearrangement: missing {category}, "
                f"using fallback '{validated[category]}'"
            )

    logger.info(
        f"Rearrangement (validated): drums='{validated['drums']}', "
        f"bass='{validated['bass']}', melodic='{validated['melodic']}'"
    )
    return validated


def _is_valid_instrument(name: str, category: str) -> bool:
    """Check if an instrument name is valid for its category.

    Validates by checking if at least one word matches known instrument vocabulary
    and the name doesn't contain obvious garbage (numbers, special chars, long words).

    Args:
        name: Instrument name to validate.
        category: One of "drums", "bass", "melodic".

    Returns:
        True if the name appears to be a real instrument.
    """
    if not name or len(name) < 3:
        return False

    # Reject if too long (>5 words = probably garbage)
    words = name.split()
    if len(words) > 5:
        return False

    # Reject if contains non-alpha characters (except hyphens and spaces)
    if re.search(r"[^a-zA-Z\s\-]", name):
        return False

    # Reject if any word is too long (>15 chars = hallucination)
    if any(len(w) > 15 for w in words):
        return False

    # Check if at least one word matches known vocabulary
    name_lower = name.lower()
    name_words = set(name_lower.split())
    if name_words & _ALL_VALID_WORDS:
        return True

    # Check against the category's known instruments (fuzzy)
    for valid_name in _VALID_INSTRUMENTS.get(category, []):
        if valid_name.lower() in name_lower or name_lower in valid_name.lower():
            return True

    return False


def build_caption_from_instruments(instruments: dict[str, str]) -> str:
    """Build a clean DiT caption from validated instrument names.

    Args:
        instruments: Dict with "drums", "bass", "melodic" keys.

    Returns:
        Caption string suitable for the DiT text encoder.
    """
    parts = []
    if instruments.get("drums"):
        parts.append(instruments["drums"])
    if instruments.get("bass"):
        parts.append(instruments["bass"])
    if instruments.get("melodic"):
        parts.append(instruments["melodic"])
    return ". ".join(parts) + "."


def _extract_short_name(raw_description: str, max_words: int = 4) -> str:
    """Extract a short instrument name from a description.

    Takes "Brushed Jazz Drums (soft ride cymbal, ghost notes)" and returns
    "Brushed Jazz Drums". Strips parenthetical details and truncates.

    Args:
        raw_description: Full instrument description from Qwen.
        max_words: Maximum words to keep.

    Returns:
        Short instrument name (2-4 words).
    """
    # Remove parenthetical content
    name = re.sub(r"\(.*?\)", "", raw_description).strip()
    # Remove trailing punctuation
    name = name.rstrip(".,;:")
    # Truncate to max_words
    words = name.split()
    if len(words) > max_words:
        words = words[:max_words]
    return " ".join(words).strip()


def parse_instruments_from_caption(caption: str) -> dict[str, str]:
    """Parse instrument names from a re-arrangement caption.

    Handles two formats:
    1. Structured: "DRUMS: ...\nBASS: ...\nMELODIC: ..."
    2. Freeform: natural language (fallback regex matching)

    Args:
        caption: Re-arrangement caption text.

    Returns:
        Dict with keys "melodic", "drums", "bass" mapping to instrument names.
    """
    result = {"melodic": "", "drums": "", "bass": ""}

    # Try structured format first (DRUMS:/BASS:/MELODIC: lines)
    for line in caption.split("\n"):
        line_stripped = line.strip()
        line_lower = line_stripped.lower()
        if line_lower.startswith("drums:"):
            raw = line_stripped[6:].strip()
            result["drums"] = _extract_short_name(raw)
        elif line_lower.startswith("bass:"):
            raw = line_stripped[5:].strip()
            result["bass"] = _extract_short_name(raw)
        elif line_lower.startswith("melodic:"):
            raw = line_stripped[8:].strip()
            result["melodic"] = _extract_short_name(raw)

    # If structured parsing found all three, we're done
    if result["drums"] and result["bass"] and result["melodic"]:
        logger.info(
            f"Parsed instruments (structured): melodic='{result['melodic'][:40]}', "
            f"drums='{result['drums'][:40]}', bass='{result['bass'][:40]}'"
        )
        return result

    # Fallback: freeform regex matching
    caption_lower = caption.lower()

    # Detect drums instrument
    if not result["drums"]:
        drums_patterns = [
            r"(brushed\s+(?:jazz\s+)?drums)",
            r"(electronic\s+drum\s+machine)",
            r"(live\s+(?:\w+\s+)?drums)",
            r"(acoustic\s+drums)",
            r"(drum\s+machine)",
            r"(programmed\s+drums)",
            r"(jazz\s+drums)",
            r"(rock\s+drums)",
            r"((?:tight|punchy|compressed)\s+drums)",
        ]
        for pattern in drums_patterns:
            match = re.search(pattern, caption_lower)
            if match:
                result["drums"] = match.group(1).title()
                break
        if not result["drums"]:
            if "drums" in caption_lower or "drum" in caption_lower:
                result["drums"] = "Drums"

    # Detect bass instrument
    if not result["bass"]:
        bass_patterns = [
            r"(upright\s+bass)",
            r"(synth\s+bass)",
            r"(electric\s+bass)",
            r"(analog\s+bass)",
            r"(sub\s+bass)",
            r"(fingerpicked\s+bass)",
            r"(bass\s+guitar)",
            r"(walking\s+bass)",
            r"(fretless\s+bass)",
        ]
        for pattern in bass_patterns:
            match = re.search(pattern, caption_lower)
            if match:
                result["bass"] = match.group(1).title()
                break
        if not result["bass"]:
            if "bass" in caption_lower:
                result["bass"] = "Bass"

    # Detect melodic instrument
    if not result["melodic"]:
        melodic_patterns = [
            r"((?:smooth\s+)?fender\s+rhodes(?:\s+keyboard)?)",
            r"((?:warm\s+)?rhodes\s+(?:electric\s+)?(?:piano|keyboard))",
            r"(rhodes(?:\s+keyboard)?)",
            r"(nylon[- ]string\s+(?:acoustic\s+)?guitar)",
            r"((?:clean|jangly)\s+electric\s+guitar)",
            r"(acoustic\s+guitar)",
            r"(analog\s+synth\s+pad)",
            r"(lush\s+(?:analog\s+)?synth\s+pad)",
            r"(grand\s+piano)",
            r"(wurlitzer)",
            r"(organ)",
            r"(piano)",
            r"(keyboard)",
            r"(synth\s+(?:pad|lead))",
            r"(trumpet)",
            r"(saxophone)",
            r"(brass\s+section)",
            r"(strings?)",
            r"(violin)",
            r"(cello)",
            r"(vibraphone)",
            r"(marimba)",
            r"(guitar)",
            r"(synth)",
        ]
        for pattern in melodic_patterns:
            match = re.search(pattern, caption_lower)
            if match:
                candidate = match.group(1).title()
                if "bass" not in candidate.lower() and "drum" not in candidate.lower():
                    result["melodic"] = candidate
                    break

    logger.info(
        f"Parsed instruments (fallback): melodic='{result['melodic'][:40]}', "
        f"drums='{result['drums'][:40]}', bass='{result['bass'][:40]}'"
    )
    return result


def _extract_playing_style(hint_fragment: str) -> str:
    """Extract playing style words from a hint, stripping instrument names.

    Takes something like "Electric Guitar Power Chords" and returns
    "Power Chords" by removing known instrument words.

    Args:
        hint_fragment: The melodic portion of a hint string.

    Returns:
        Playing style words only (e.g., "Strumming", "Power Chords", "Solo").
    """
    words = hint_fragment.split()
    style_words = [
        w for w in words
        if w.lower() not in _ORIGINAL_INSTRUMENT_WORDS
    ]
    return " ".join(style_words).strip()


def translate_temporal_script(
    section_tags: list[str],
    original_hints: list[str],
    new_instruments: dict[str, str],
) -> str:
    """Translate temporal script hints to use re-arranged instrument names.

    Simple approach: swap instrument names, keep playing style words as-is.
    The model interprets "Rhodes Piano Strumming" as "rhythmic chordal Rhodes"
    which is directionally correct.

    Args:
        section_tags: Section tags from SongFormer (e.g., "[Verse 1]").
        original_hints: Per-section hints with original instruments.
        new_instruments: Dict from parse_instruments_from_caption().

    Returns:
        Assembled lyrics string with new instrument names.
    """
    melodic_name = new_instruments.get("melodic", "")
    drums_name = new_instruments.get("drums", "Drums")
    bass_name = new_instruments.get("bass", "Bass")

    lines = []
    for tag, hint in zip(section_tags, original_hints):
        if not hint or hint == "soft":
            lines.append(tag)
            lines.append("[Instrumental]")
            lines.append("")
            continue

        hint_lower = hint.lower()
        new_parts = []

        # Check drums presence in original hint
        has_drums = "drums" in hint_lower or "drum" in hint_lower
        if has_drums:
            new_parts.append(drums_name)

        # Check bass presence in original hint
        has_bass = "bass" in hint_lower
        if has_bass:
            new_parts.append(bass_name)

        # Extract melodic part: remove drums/bass words, get the rest
        melodic_fragment = hint
        for remove_word in ["drums", "drum", "bass"]:
            melodic_fragment = re.sub(
                rf"\b{remove_word}\b", "", melodic_fragment, flags=re.IGNORECASE
            )
        melodic_fragment = melodic_fragment.strip()

        # If there's a melodic component, swap instrument name + keep style
        if melodic_fragment and melodic_name:
            style = _extract_playing_style(melodic_fragment)
            if style:
                new_parts.append(f"{melodic_name} {style}")
            else:
                new_parts.append(melodic_name)
        elif melodic_fragment and not melodic_name:
            # No melodic instrument parsed — keep original fragment
            style = _extract_playing_style(melodic_fragment)
            if style:
                new_parts.append(style)

        # Build the new hint
        new_hint = " ".join(new_parts) if new_parts else ""

        if new_hint:
            tag_with_hint = f"{tag[:-1]} - {new_hint}]"
        else:
            tag_with_hint = tag

        lines.append(tag_with_hint)
        lines.append("[Instrumental]")
        lines.append("")

    return "\n".join(lines).strip()
