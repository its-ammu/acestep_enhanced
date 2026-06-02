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
        "Punchy Electronic Drums", "Industrial Drums", "Breakbeat Drums",
        "808 Drum Machine", "Aggressive Rock Drums", "Disco Drums",
    ],
    # Bass
    "bass": [
        "Upright Bass", "Synth Bass", "Electric Bass", "Fretless Bass",
        "Sub Bass", "Slap Bass", "Fingerpicked Bass", "Analog Bass",
        "Moog Bass", "Acoustic Bass", "Distorted Synth Bass",
        "Funk Bass", "Aggressive Bass", "Wobble Bass", "Acid Bass",
    ],
    # Melodic
    "melodic": [
        "Rhodes Piano", "Grand Piano", "Nylon Guitar", "Analog Synth Pad",
        "Wurlitzer Piano", "Vibraphone", "Marimba", "Accordion",
        "Pedal Steel Guitar", "Lap Steel Guitar", "Harmonica",
        "Trumpet", "Saxophone", "Clarinet", "Violin", "Cello",
        "Flute", "Organ", "Harpsichord", "Mandolin", "Banjo",
        "Sitar", "Kalimba", "Music Box", "Harp", "Dulcimer",
        "Analog Synth Lead", "Distorted Synth", "Electric Guitar",
        "Overdriven Guitar", "Synth Arpeggio", "Brass Section",
        "Screaming Synth Lead", "Wah Guitar", "Theremin",
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
    "drums", "drum", "bass", "percussion", "kit",
}


def build_rearrangement_prompt(
    original_caption: str,
    vocal_profile: str,
    metadata: Optional[dict] = None,
) -> str:
    """Build the creative Qwen prompt — pick a genre number from a list.

    Instead of asking Qwen to name instruments (unreliable), we give it
    numbered genre options and ask it to pick ONE number. This is a
    classification task that Qwen handles reliably.

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
        f"This song has: {original_caption}\n"
        f"It's {bpm} BPM in {keyscale}.\n\n"
        "I want to remix this into a different genre. "
        "Which of these genres would work best at this tempo "
        "and match the song's energy? Pick ONE number:\n\n"
        "1. Synthwave (electronic drums, analog synth bass, synth lead)\n"
        "2. Funk (tight funk drums, slap bass, clavinet)\n"
        "3. Industrial (aggressive electronic drums, distorted bass, distorted synth)\n"
        "4. Neo-Soul (live drums with ghost notes, warm electric bass, rhodes piano)\n"
        "5. Disco (disco drums, disco bass, string section)\n"
        "6. Trip-Hop (breakbeat drums, sub bass, atmospheric synth pad)\n"
        "7. Latin Rock (latin percussion, fingerpicked bass, nylon guitar)\n"
        "8. Electro-Funk (808 drum machine, moog bass, vocoder synth)\n"
        "9. Post-Punk (aggressive drums, driving bass, angular synth)\n"
        "10. Cinematic (orchestral percussion, cello, brass section)\n\n"
        "Reply with ONLY the number. Nothing else."
    )

    return prompt


# Pre-validated genre → instrument mappings (ACE-Step renders these well)
_GENRE_INSTRUMENT_MAP = {
    "1": {"drums": "Electronic Drum Machine", "bass": "Analog Bass", "melodic": "Analog Synth Lead"},
    "2": {"drums": "Tight Funk Drums", "bass": "Slap Bass", "melodic": "Rhodes Piano"},
    "3": {"drums": "Aggressive Rock Drums", "bass": "Distorted Synth Bass", "melodic": "Distorted Synth"},
    "4": {"drums": "Live Rock Drums", "bass": "Electric Bass", "melodic": "Rhodes Piano"},
    "5": {"drums": "Disco Drums", "bass": "Electric Bass", "melodic": "Analog Synth Pad"},
    "6": {"drums": "Breakbeat Drums", "bass": "Sub Bass", "melodic": "Analog Synth Pad"},
    "7": {"drums": "Bongo Congas", "bass": "Fingerpicked Bass", "melodic": "Nylon Guitar"},
    "8": {"drums": "808 Drum Machine", "bass": "Moog Bass", "melodic": "Analog Synth Lead"},
    "9": {"drums": "Aggressive Rock Drums", "bass": "Electric Bass", "melodic": "Analog Synth Lead"},
    "10": {"drums": "Orchestral Percussion", "bass": "Cello", "melodic": "Trumpet"},
}

# Genre-specific layer templates for multi-layer Lego generation
# Each genre has 3 layers: pad (harmonic fill), lead (melody), rhythm (movement)
_GENRE_LAYERS = {
    "1": [  # Synthwave
        {"track": "synth", "caption": "warm analog synth pad with lush sustained chords, slow filter sweeps, wide stereo, atmospheric"},
        {"track": "synth", "caption": "expressive analog synth lead with dynamic melody, pitch bends, vibrato, soaring in choruses"},
        {"track": "synth", "caption": "pulsing synth arpeggio, 16th note pattern, rhythmic and driving, sidechain pumping"},
    ],
    "2": [  # Funk
        {"track": "keyboard", "caption": "rhodes piano with warm jazzy chords, rhythmic comping, ghost notes, groove-locked"},
        {"track": "keyboard", "caption": "clavinet lead with funky stabs, wah-wah effect, syncopated rhythm, percussive attack"},
        {"track": "brass", "caption": "tight brass stabs, syncopated hits, punchy and short, filling gaps between vocals"},
    ],
    "3": [  # Industrial
        {"track": "synth", "caption": "dark distorted synth pad, heavy and aggressive, low-frequency rumble, menacing atmosphere"},
        {"track": "synth", "caption": "screaming distorted synth lead, aggressive melody, harsh filter sweeps, intense"},
        {"track": "synth", "caption": "glitchy rhythmic synth stabs, industrial percussion hits, mechanical and relentless"},
    ],
    "4": [  # Neo-Soul
        {"track": "keyboard", "caption": "warm rhodes piano with soft jazzy chords, gentle comping, intimate and smooth"},
        {"track": "guitar", "caption": "clean electric guitar with gentle melody, jazz voicings, subtle bends, warm tone"},
        {"track": "strings", "caption": "soft string pad, sustained harmonies, lush and warm, background texture"},
    ],
    "5": [  # Disco
        {"track": "strings", "caption": "lush disco string section, sustained chords, sweeping arrangements, orchestral"},
        {"track": "keyboard", "caption": "funky electric piano, rhythmic chords, bright and percussive, disco groove"},
        {"track": "guitar", "caption": "disco rhythm guitar, muted 16th note strumming, tight and funky, wah pedal"},
    ],
    "6": [  # Trip-Hop
        {"track": "synth", "caption": "dark atmospheric synth pad, reverb-drenched, slow evolving texture, haunting"},
        {"track": "keyboard", "caption": "sparse piano melody, lo-fi processed, reverb, melancholic and minimal"},
        {"track": "synth", "caption": "subtle vinyl crackle texture with ambient noise, atmospheric background layer"},
    ],
    "7": [  # Latin Rock
        {"track": "guitar", "caption": "nylon string guitar with fingerpicked arpeggios, warm and expressive, latin feel"},
        {"track": "keyboard", "caption": "organ with sustained chords, warm tone, filling harmonic space"},
        {"track": "percussion", "caption": "shaker and tambourine rhythm, steady 8th notes, adding movement and groove"},
    ],
    "8": [  # Electro-Funk
        {"track": "synth", "caption": "thick moog synth pad, warm analog chords, funky filter modulation, groovy"},
        {"track": "synth", "caption": "vocoder synth lead, robotic melody, funky and expressive, talk-box style"},
        {"track": "synth", "caption": "synth arpeggio with funky rhythm, 16th note pattern, bouncy and energetic"},
    ],
    "9": [  # Post-Punk
        {"track": "synth", "caption": "angular synth pad, cold and sharp, minimal chords, post-punk atmosphere"},
        {"track": "synth", "caption": "aggressive synth lead, angular melody, sharp attack, driving and urgent"},
        {"track": "guitar", "caption": "jangly guitar arpeggios, chorus effect, rhythmic and hypnotic, post-punk style"},
    ],
    "10": [  # Cinematic
        {"track": "strings", "caption": "epic orchestral strings, sweeping sustained chords, building and dramatic"},
        {"track": "brass", "caption": "powerful brass melody, heroic and soaring, dynamic crescendos"},
        {"track": "percussion", "caption": "orchestral percussion hits, timpani rolls, dramatic accents, cinematic impact"},
    ],
}


def get_genre_layers(genre_number: str) -> list[dict]:
    """Get the multi-layer Lego templates for a genre.

    Args:
        genre_number: Genre number string ("1"-"10").

    Returns:
        List of layer dicts with "track" and "caption" keys.
        Falls back to synthwave if genre not found.
    """
    return _GENRE_LAYERS.get(genre_number, _GENRE_LAYERS["1"])


# Stores the last selected genre number for downstream use
_last_genre_number: str = "1"


def get_last_genre_number() -> str:
    """Return the genre number from the most recent parse_genre_choice call."""
    return _last_genre_number


def parse_genre_choice(response: str) -> Optional[dict[str, str]]:
    """Parse Qwen's genre response into instrument dict.

    Handles both number responses ("8") and name responses ("Electro-funk").
    Falls back to random selection if completely unparseable.
    Also stores the genre number for downstream layer lookup.

    Args:
        response: Qwen's response (ideally a number or genre name).

    Returns:
        Instrument dict or None if unparseable.
    """
    global _last_genre_number
    response_lower = response.lower().strip()

    # Try number first
    numbers = re.findall(r"\b(\d{1,2})\b", response)
    if numbers:
        choice = numbers[0]
        if choice in _GENRE_INSTRUMENT_MAP:
            _last_genre_number = choice
            instruments = _GENRE_INSTRUMENT_MAP[choice]
            logger.info(f"Genre choice (number): {choice} → {instruments}")
            return instruments

    # Try matching genre name keywords (longer/more specific patterns first)
    _GENRE_NAME_MAP = [
        ("electro-funk", "8"), ("electro funk", "8"), ("electrofunk", "8"),
        ("synth-pop", "1"), ("synth pop", "1"), ("synthpop", "1"),
        ("synthwave", "1"), ("synth-wave", "1"), ("synth wave", "1"),
        ("neo-soul", "4"), ("neo soul", "4"), ("neosoul", "4"),
        ("trip-hop", "6"), ("trip hop", "6"), ("triphop", "6"),
        ("post-punk", "9"), ("post punk", "9"), ("postpunk", "9"),
        ("latin rock", "7"),
        ("industrial", "3"),
        ("cinematic", "10"), ("orchestral", "10"),
        ("disco", "5"),
        ("latin", "7"),
        ("funk", "2"), ("funky", "2"),
    ]

    for name, num in _GENRE_NAME_MAP:
        if name in response_lower:
            _last_genre_number = num
            instruments = _GENRE_INSTRUMENT_MAP[num]
            logger.info(f"Genre choice (name match '{name}'): {num} → {instruments}")
            return instruments

    # Last resort: pick a random energetic option
    import random
    energetic_choices = ["1", "2", "3", "8", "9"]  # Skip mellow options
    choice = random.choice(energetic_choices)
    _last_genre_number = choice
    instruments = _GENRE_INSTRUMENT_MAP[choice]
    logger.warning(
        f"Could not parse genre from: '{response[:50]}' — "
        f"random fallback: {choice} → {instruments}"
    )
    return instruments


def build_formatting_prompt(creative_response: str) -> str:
    """Build the formatting prompt — extract structured instrument names.

    This is Call 2 of 2: pure formatting, no creativity needed.

    Args:
        creative_response: Free-form text from the creative call.

    Returns:
        Prompt asking for structured extraction.
    """
    prompt = (
        f"Extract the three instruments from this text:\n"
        f'"{creative_response}"\n\n'
        f"Write exactly 3 lines:\n"
        f"DRUMS: [the drum/percussion instrument name, 2-3 words]\n"
        f"BASS: [the bass instrument name, 2-3 words]\n"
        f"MELODIC: [the melodic instrument name, 2-3 words]\n"
        f"Nothing else."
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

    # Reject if contains non-alpha characters (except hyphens, spaces, and digits)
    if re.search(r"[^a-zA-Z\s\-0-9]", name):
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
    """Build a rich DiT caption from validated instrument names.

    The caption should include genre context, mood, production style,
    AND performance instructions. ACE-Step responds to playing style
    descriptors that tell it HOW to perform, not just WHAT instrument.

    Args:
        instruments: Dict with "drums", "bass", "melodic" keys.

    Returns:
        Caption string suitable for the DiT text encoder.
    """
    drums = instruments.get("drums", "drums")
    bass = instruments.get("bass", "bass")
    melodic = instruments.get("melodic", "synth")

    # Genre-specific performance descriptions for richer output
    melodic_lower = melodic.lower()

    # Determine performance style based on instrument type
    if any(k in melodic_lower for k in ("synth lead", "analog synth lead")):
        melodic_desc = (
            f"expressive {melodic.lower()} with dynamic performance, "
            f"rhythmic arpeggios building to soaring sustained leads, "
            f"aggressive pitch bends and filter sweeps in solo sections"
        )
    elif any(k in melodic_lower for k in ("synth pad", "pad")):
        melodic_desc = (
            f"lush {melodic.lower()} with slow evolving textures, "
            f"wide stereo chorusing, subtle filter movement, "
            f"harmonic swells building through choruses"
        )
    elif any(k in melodic_lower for k in ("rhodes", "wurlitzer", "piano")):
        melodic_desc = (
            f"warm {melodic.lower()} with jazzy chord voicings, "
            f"rhythmic comping with ghost notes in verses, "
            f"sustained emotional chords swelling in choruses"
        )
    elif any(k in melodic_lower for k in ("guitar", "nylon")):
        melodic_desc = (
            f"expressive {melodic.lower()} with fingerpicked arpeggios "
            f"in verses, strummed chords building in choruses, "
            f"melodic lead lines with vibrato in instrumental sections"
        )
    elif any(k in melodic_lower for k in ("trumpet", "brass", "sax")):
        melodic_desc = (
            f"powerful {melodic.lower()} with dynamic phrasing, "
            f"sustained melodic lines building intensity, "
            f"punchy staccato accents and soaring legato passages"
        )
    elif any(k in melodic_lower for k in ("violin", "cello", "string")):
        melodic_desc = (
            f"sweeping {melodic.lower()} with emotional sustained lines, "
            f"dynamic crescendos building through sections, "
            f"pizzicato rhythmic accents alternating with legato passages"
        )
    elif "distorted" in melodic_lower:
        melodic_desc = (
            f"aggressive {melodic.lower()} with heavy distortion, "
            f"grinding rhythmic patterns in verses, "
            f"screaming lead lines with feedback in choruses"
        )
    else:
        melodic_desc = (
            f"expressive {melodic.lower()} with dynamic performance, "
            f"rhythmic patterns in verses building to powerful leads "
            f"in choruses, intense solo with pitch variation"
        )

    caption = (
        f"{melodic_desc}. "
        f"{drums} with tight groove and dynamic fills, "
        f"deep {bass.lower()} locking with kick drum pattern. "
        f"Professional production, punchy mix, wide stereo field."
    )
    return caption


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
    """Translate temporal script to use energy/style hints without instrument names.

    The caption already tells the model what instruments to use. The lyrics
    should provide structural guidance (section boundaries) and energy/style
    descriptors using official ACE-Step energy tags.

    Args:
        section_tags: Section tags from SongFormer (e.g., "[Verse 1]").
        original_hints: Per-section hints with original instruments.
        new_instruments: Dict from parse_instruments_from_caption().

    Returns:
        Assembled lyrics string with clean section tags and energy hints.
    """
    lines = []
    num_sections = len(section_tags)

    for idx, (tag, hint) in enumerate(zip(section_tags, original_hints)):
        if not hint or hint == "soft":
            lines.append(f"{tag[:-1]} - soft]")
            lines.append("[Instrumental]")
            lines.append("")
            continue

        hint_lower = hint.lower()

        # Determine energy level from hint content and section position
        energy_keywords = []

        # Solo/lead detection
        if "solo" in hint_lower or "lead" in hint_lower:
            energy_keywords.append("explosive solo")
        # Energy from stem analysis keywords
        elif "peak energy" in hint_lower or "full energy" in hint_lower:
            energy_keywords.append("high energy")
        elif "building" in hint_lower:
            energy_keywords.append("building energy")
        elif "dropping" in hint_lower:
            energy_keywords.append("fading")
        elif "moderate energy" in hint_lower or "moderate" in hint_lower:
            energy_keywords.append("moderate energy")
        elif "soft" in hint_lower:
            energy_keywords.append("soft")
        else:
            # Infer energy from section type and position
            tag_lower = tag.lower()
            if "intro" in tag_lower:
                energy_keywords.append("building, atmospheric")
            elif "chorus" in tag_lower:
                # Later choruses are more intense
                chorus_position = sum(
                    1 for t in section_tags[:idx] if "chorus" in t.lower()
                )
                if chorus_position >= 3:
                    energy_keywords.append("peak energy, anthemic")
                elif chorus_position >= 1:
                    energy_keywords.append("high energy, driving")
                else:
                    energy_keywords.append("high energy, powerful")
            elif "verse" in tag_lower:
                energy_keywords.append("moderate energy")
            elif "inst" in tag_lower:
                energy_keywords.append("intense, dynamic")
            elif "outro" in tag_lower:
                if idx >= num_sections - 1:
                    energy_keywords.append("fade out")
                else:
                    energy_keywords.append("powerful, fading")
            elif "bridge" in tag_lower:
                energy_keywords.append("building energy")

        # Extract playing style (mutes, riff, strumming, etc.)
        style = _extract_playing_style(hint)
        if style and style.lower() not in ("strumming", ""):
            energy_keywords.append(style.lower())

        # Build tag with energy hint
        if energy_keywords:
            hint_str = ", ".join(energy_keywords)
            tag_with_hint = f"{tag[:-1]} - {hint_str}]"
        else:
            tag_with_hint = tag

        lines.append(tag_with_hint)
        lines.append("[Instrumental]")
        lines.append("")

    return "\n".join(lines).strip()
