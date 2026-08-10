"""Which language is this text, and which voice should speak it?

Script detection, not statistical language ID. For the languages that matter
here the script *is* the answer -- Devanagari means Hindi/Marathi, Tamil script
means Tamil -- and a script check cannot be wrong about it, needs no model, and
costs nothing.

The interesting case is code-mixing. "मैंने अभी email भेज दिया" is Hindi with an
English noun in it, and it must be routed to the Hindi voice: an Indic model
pronouncing an English word sounds like an Indian person saying it, which is
correct, whereas an English model reading Devanagari produces nothing usable.
So any meaningful amount of Indic script wins.
"""

from __future__ import annotations

from dataclasses import dataclass

#: Unicode blocks for the scripts we route on.
SCRIPT_RANGES: dict[str, tuple[str, str]] = {
    "devanagari": ("ऀ", "ॿ"),  # Hindi, Marathi, Nepali, Sanskrit
    "bengali": ("ঀ", "৿"),
    "gurmukhi": ("਀", "੿"),  # Punjabi
    "gujarati": ("઀", "૿"),
    "oriya": ("଀", "୿"),
    "tamil": ("஀", "௿"),
    "telugu": ("ఀ", "౿"),
    "kannada": ("ಀ", "೿"),
    "malayalam": ("ഀ", "ൿ"),
}

#: Script -> default language tag. Devanagari is ambiguous (Hindi/Marathi/…);
#: Hindi is the right default and callers can override.
SCRIPT_TO_LANG = {
    "devanagari": "hi",
    "bengali": "bn",
    "gurmukhi": "pa",
    "gujarati": "gu",
    "oriya": "or",
    "tamil": "ta",
    "telugu": "te",
    "kannada": "kn",
    "malayalam": "ml",
}

#: Below this share of Indic characters, treat the text as English with a
#: borrowed word rather than as code-mixed Indic. One stray Devanagari
#: character in an English sentence should not switch the whole voice.
INDIC_SHARE_THRESHOLD = 0.15


@dataclass(frozen=True)
class Detection:
    language: str
    script: str
    indic_share: float
    is_code_mixed: bool

    @property
    def is_indic(self) -> bool:
        return self.language != "en"


def detect(text: str) -> Detection:
    """Identify the language to synthesize this text in."""
    counts: dict[str, int] = {}
    latin = 0
    letters = 0

    for ch in text:
        if not ch.isalpha():
            continue
        letters += 1
        if ch.isascii():
            latin += 1
            continue
        for script, (low, high) in SCRIPT_RANGES.items():
            if low <= ch <= high:
                counts[script] = counts.get(script, 0) + 1
                break

    if not letters or not counts:
        return Detection("en", "latin", 0.0, False)

    script = max(counts, key=counts.get)
    indic = counts[script]
    share = indic / letters

    if share < INDIC_SHARE_THRESHOLD:
        return Detection("en", "latin", share, latin > 0 and indic > 0)

    return Detection(
        language=SCRIPT_TO_LANG[script],
        script=script,
        indic_share=share,
        is_code_mixed=latin > 0,
    )
