"""Held-out Hindi sentences for the blind A/B benchmark.

Deliberately separate from `eval/sentences.py` and `train/prompts.py`, and
provably disjoint from both plus the training transcripts — a test set that leaks
into training measures memorisation and reports it as quality.
`tests/test_heldout.py` asserts the disjointness rather than trusting it.

WHY THESE SENTENCES AND NOT PRETTIER ONES

Each targets a failure mode that third-party reviewers report of the incumbent, so
a loss here says *what* to fix rather than just that we lost. The reported failures
are: accent bleed from English training data outside the top ten languages,
unnatural prosody, mispronunciation of region-specific terms, trouble with numbers
and abbreviations, and — most cited — accent drift and mid-sentence language
switching on long-form, needing two or three regenerations.

So the set over-weights code-mixing, digits, and long sentences with internal
clauses. It is a diagnostic instrument, not a showcase; some of these are meant to
be hard.

AUTHORSHIP CAVEAT, as in `train/prompts.py`: written by the assistant, not a native
speaker. Read them aloud before running a benchmark on them. A sentence that reads
oddly to a Hindi speaker invalidates whatever score it produces, and the fix is to
delete it.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class HeldOut:
    slug: str
    text: str
    targets: str
    """The reported failure mode this sentence is chosen to expose."""


SENTENCES: tuple[HeldOut, ...] = (
    HeldOut(
        "h1",
        "मैंने कल रात एक documentary देखी जो climate change के बारे में थी।",
        "code-mixing — mid-sentence language switch, the most cited long-form failure",
    ),
    HeldOut(
        "h2",
        "इस साल टीम ने 43 प्रोजेक्ट पूरे किए और 12 नए क्लाइंट जोड़े।",
        "digits — 'trouble with numbers and abbreviations'",
    ),
    HeldOut(
        "h3",
        "अगर आप चाहें तो मैं आपको पूरी report भेज देता हूँ, लेकिन उससे पहले एक meeting "
        "कर लेते हैं ताकि सब clear हो जाए।",
        "worst case — long form, three clauses, three code-switches",
    ),
    HeldOut(
        "h4",
        "बड़ी ठंड में लड़का पहाड़ी रास्ते पर चढ़ता रहा।",
        "retroflex series — ड़ ठ ड़ ढ़, where a model trained on English collapses the contrast",
    ),
    HeldOut(
        "h5",
        "ज़रा साफ़ आवाज़ में फ़ोन पर बात कीजिए।",
        "nuqta consonants — ज़ फ़ ज़ फ़, routinely flattened",
    ),
    HeldOut(
        "h6",
        "आपको क्या लगता है, यह तरीका सही रहेगा?",
        "question contour — prosody, not phonemes",
    ),
    HeldOut(
        "h7",
        "अरे! यह तो मैंने सोचा भी नहीं था।",
        "exclamation — emotional range on a short utterance",
    ),
    HeldOut(
        "h8",
        "बिल्कुल, हो जाएगा।",
        "short utterance — onset and final lengthening, no context to hide in",
    ),
    HeldOut(
        "h9",
        "समिति ने अपनी रिपोर्ट सरकार को सौंप दी है।",
        "formal register — news Hindi, exposes flat delivery",
    ),
    HeldOut(
        "h10",
        "देख यार, मुझे लगता है हमें एक बार फिर से सोचना चाहिए।",
        "colloquial with discourse markers — casual rhythm",
    ),
    HeldOut(
        "h11",
        "विद्यार्थियों ने प्रश्नपत्र ध्यान से पढ़ा और उत्तर लिखे।",
        "consonant clusters — द्य श्न प्र, where an inserted vowel is audible",
    ),
    HeldOut(
        "h12",
        "पिछले हफ़्ते जो फ़ाइल आपने भेजी थी, उसमें दो पेज गायब थे, इसलिए मैंने दोबारा "
        "मंगवा ली है।",
        "long form with relative clause and nuqta — accent drift across a long span",
    ),
)


def texts() -> list[str]:
    return [s.text for s in SENTENCES]


def by_slug(slug: str) -> HeldOut:
    for sentence in SENTENCES:
        if sentence.slug == slug:
            return sentence
    raise KeyError(slug)
