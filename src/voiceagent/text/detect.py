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
#: It is used as a *detector* here, and the reason to refuse Marathi is the
#: measurement in `eval/devanagari.py` --- but read that measurement carefully,
#: because half of it is confounded and the honest half is the other one:
#:
#:   * **Load-bearing:** on every one of four seeds the model rendered शाळेत
#:     (*shaaLet*, "in school") as शायद (*shaayad*, "maybe"). A different, real
#:     Hindi word. A scorer that merely disliked a grapheme would write शालेत; it
#:     would not invent a Hindi adverb. That is a Hindi lexicon reading Marathi,
#:     it does not depend on spelling, and it is why this refuses.
#:   * **Not load-bearing:** "`ळ` survived 0 of 4 seeds". True, and not the
#:     model's fault alone --- `ळ` has never appeared in any Whisper transcript
#:     this project has produced, including from Indic Parler-TTS, which does
#:     speak Marathi. Whisper spends two BPE tokens on `ळ` against one for `ल`
#:     and `र` and is biased against emitting it, so the round trip cannot tell
#:     "not said" from "not written".
#:
#: None of that weakens `ळ` as a *detector*: it is absent from standard Hindi, so
#: seeing it in input text identifies Marathi regardless of what any ASR does
#: with the output.
MARATHI_MARKER = "ळ"

#: Function words that occur in Nepali and not in Hindi. Unlike `ळ` this is a
#: recall-limited heuristic, not a definitive test --- Nepali and Hindi share far
#: too much Devanagari for a clean signal, and a Nepali sentence built only from
#: shared vocabulary will not be caught. It is deliberately biased to precision:
#: a miss reads as Hindi and synthesizes as it does today, which is the current
#: behaviour and no worse.
#:
#: MATCHED AS WHOLE WORDS, NOT SUBSTRINGS, and that is not a detail. Substring
#: matching flagged the ordinary Hindi sentence "छन्द में लिखी कविता सुंदर थी।"
#: because `छन्` sits inside छन्द (*chhand*, verse), and "यह गर्नल साहब का आदेश
#: है।" because `गर्न` sits inside गर्नल (colonel). As an advisory header those
#: were noise; as a refusal they would block correct Hindi, which is why this was
#: fixed at the same time the Marathi case was hardened.
#: Matched as a **word prefix**, because Nepali agglutinates: तपाईं takes a
#: postposition and becomes तपाईंलाई, and whole-word matching misses every
#: inflected form. Each of these was checked not to be the opening of an
#: ordinary Hindi word.
NEPALI_MARKERS = frozenset({"तपाईं", "एउटा", "मैले", "हिजो", "भोलि", "हुन्छ"})

#: Matched only as **whole words**, because each is the opening of a real Hindi
#: word and prefix-matching them would flag correct Hindi:
#:
#:     छन्  ->  छन्द   (*chhand*, verse)
#:     गर्न ->  गर्नल  (colonel)
#:
#: They stay in the list because as complete words they are good Nepali signals;
#: they are separated because the matching rule that suits the others is unsafe
#: for them. Recall for their inflected forms is given up deliberately --- a miss
#: is today's behaviour, a false positive is a blocked request.
NEPALI_EXACT = frozenset({"छन्", "गर्न"})


#: Severities. `REFUSE` means the request should not be served at all; `WARN`
#: means serve it and say so.
#:
#: The two are set by how strong the evidence is, and it differs sharply between
#: the languages (`eval_out/devanagari/FINDINGS.md`):
#:
#:   * **Marathi refuses.** `ळ` came back 0 of 4 seeds and the model substituted
#:     Hindi words for Marathi ones on every one of them. That is not a
#:     degradation, it is a language the engine cannot say, and the detector is a
#:     single character that does not occur in standard Hindi.
#:   * **Nepali warns.** `र्` conjuncts survived 2 of 4 seeds --- unreliable
#:     rather than absent --- and the detector is a recall-limited word list
#:     rather than a definitive marker. Refusing on a weaker signal *and* weaker
#:     evidence would block text the engine handles acceptably.
REFUSE = "refuse"
WARN = "warn"


@dataclass(frozen=True)
class DevanagariNote:
    language: str
    severity: str
    message: str

    @property
    def refuses(self) -> bool:
        return self.severity == REFUSE


def _words(text: str) -> set[str]:
    """Devanagari word tokens, split on anything that is not a letter or mark.

    Danda and double danda (`।` `॥`) are Devanagari punctuation and are not
    letters, so the default split handles them; they are called out because a
    naive `str.split()` on whitespace leaves "छ।" attached and the match silently
    stops working at the end of every sentence.
    """
    out: set[str] = []
    current: list[str] = []
    for char in text:
        if char.isalpha() or "ऀ" <= char <= "ॏ" and not char.isspace():
            current.append(char)
        else:
            if current:
                out.append("".join(current))
                current = []
    if current:
        out.append("".join(current))
    return set(out)


def devanagari_language_note(text: str) -> DevanagariNote | None:
    """Identify Devanagari text that is a language we render badly.

    `detect` maps every Devanagari string to `hi`, which is the right default and
    is why Marathi and Nepali reach the Hindi engine *without tripping any
    guard*. They were not refused because they were not detected. This names the
    case and says how strongly.

    Returns None for ordinary Hindi. Precision over recall throughout: a missed
    Marathi sentence behaves as it did before this existed, while a false
    positive now blocks a paying customer's correct Hindi — which is exactly why
    the Nepali markers are matched as whole words rather than substrings.

    See `eval_out/devanagari/FINDINGS.md`. The short version: Chatterbox reads
    both languages intelligibly (0.77 and 0.80 mean round-trip overlap, nothing
    below the 0.50 alarm) and speaks neither natively.
    """
    detection = detect(text)
    if detection.script != "devanagari":
        return None

    if MARATHI_MARKER in text:
        return DevanagariNote(
            language="mr",
            severity=REFUSE,
            message=(
                f"This is Marathi. The engine cannot produce {MARATHI_MARKER!r} — "
                "measured 0 of 4 generations — and substitutes Hindi words for "
                "Marathi ones where the two collide, so it would return audio that "
                "is not Marathi rather than audio that is imperfect. Chatterbox "
                "Multilingual carries one Indic language token and it is Hindi. "
                "Marathi is roadmap, not configuration."
            ),
        )

    words = _words(text)
    looks_nepali = bool(NEPALI_EXACT & words) or any(
        word.startswith(marker) for word in words for marker in NEPALI_MARKERS
    )
    if looks_nepali:
        return DevanagariNote(
            language="ne",
            severity=WARN,
            message=(
                "This looks like Nepali. It will synthesize and be understandable, "
                "but the model is reading it with Hindi phonology; conjuncts such "
                "as 'र्' survive only some generations. Chatterbox Multilingual "
                "carries one Indic language token and it is Hindi."
            ),
        )

    return None
