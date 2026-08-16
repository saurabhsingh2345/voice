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


#: Marathi's retroflex lateral. It is not a letter of standard Hindi, so its
#: presence identifies the text as Marathi (or Konkani) with essentially no false
#: positives. That precision is the whole reason this check is one character
#: rather than a word list.
#:
#: It is here because we measured that we cannot say it. `eval/devanagari.py`
#: re-generated "सकाळी मी शाळेत लवकर पोहोचलो." under four seeds and `ळ` came back
#: 0 times out of 4 --- and on every seed the model reached for Hindi instead,
#: rendering शाळेत (*shaaLet*, "in school") as शायद (*shaayad*, "maybe"). The
#: grapheme is in the tokenizer; it is not in the model's output.
MARATHI_MARKER = "ळ"

#: Function words that occur in Nepali and not in Hindi. Unlike `ळ` this is a
#: recall-limited heuristic, not a definitive test --- Nepali and Hindi share far
#: too much Devanagari for a clean signal, and a Nepali sentence built only from
#: shared vocabulary will not be caught. It is deliberately biased to precision:
#: a miss reads as Hindi and synthesizes as it does today, which is the current
#: behaviour and no worse.
NEPALI_MARKERS = frozenset({"तपाईं", "एउटा", "मैले", "हिजो", "भोलि", "छन्", "गर्न", "हुन्छ"})


def devanagari_language_note(text: str) -> str | None:
    """Warn when Devanagari text is a language we render badly.

    `detect` maps every Devanagari string to `hi`, which is the right default and
    is why Marathi and Nepali reach the Hindi engine today *without tripping any
    guard*. They are not refused because they are not detected. This names that
    case so a caller can decide what to do about it.

    Returns None for ordinary Hindi. Precision over recall throughout: a missed
    Marathi sentence behaves exactly as it does now, while a false positive would
    warn about correct Hindi, which is worse.

    See `eval_out/devanagari/FINDINGS.md`. The short version: Chatterbox reads
    both languages intelligibly (0.77 and 0.80 mean round-trip overlap, nothing
    below the 0.50 alarm) and speaks neither natively.
    """
    detection = detect(text)
    if detection.script != "devanagari":
        return None

    if MARATHI_MARKER in text:
        return (
            "This looks like Marathi. It will synthesize and be understandable, but "
            f"the model does not produce {MARATHI_MARKER!r} — measured 0 of 4 seeds — "
            "and substitutes Hindi words for Marathi ones where the two collide. "
            "Chatterbox Multilingual carries one Indic language token and it is Hindi."
        )

    if any(marker in text for marker in NEPALI_MARKERS):
        return (
            "This looks like Nepali. It will synthesize and be understandable, but the "
            "model is reading it with Hindi phonology; conjuncts such as 'र्' survive "
            "only some generations. Chatterbox Multilingual carries one Indic language "
            "token and it is Hindi."
        )

    return None
