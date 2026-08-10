"""Test sentences for judging whether a voice sounds native.

Three registers, because they fail differently:

  * FORMAL -- news/announcement Hindi. Exposes wrong vowel length and flat,
    English-style stress.
  * COLLOQUIAL -- how people actually talk. Exposes stilted rhythm; a model
    trained on read-aloud corpora often cannot do casual pacing.
  * CODE_MIXED -- Hindi with English words mid-sentence, which is how most
    urban Indian speech works. Exposes the worst failure: the model switching
    accent mid-utterance, or reading English words with Hindi phonemes.

Also included are NUMERIC cases, since numbers and dates are where a missing
text-normalization pass becomes obvious.

Everything here is Devanagari. If any of it reaches a TTS model as Latin
characters, something upstream has romanized it and that is a bug.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TestSentence:
    slug: str
    text: str
    register: str
    note: str = ""
    """What this sentence is designed to expose."""


FORMAL: tuple[TestSentence, ...] = (
    TestSentence("f1", "नमस्ते, मैं आपकी क्या सहायता कर सकता हूँ?", "formal",
                 "long aa in सहायता; nasalized हूँ"),
    TestSentence("f2", "आज मौसम बहुत सुहावना है और आसमान बिल्कुल साफ़ है।", "formal",
                 "ऌ cluster in बिल्कुल; nuqta in साफ़"),
    TestSentence("f3", "कृपया अपना नाम और पता स्पष्ट रूप से बताइए।", "formal",
                 "vocalic कृ; retroflex ट in पता/बताइए"),
    TestSentence("f4", "भारत एक विशाल और विविधतापूर्ण देश है।", "formal",
                 "aspirated भ; long compound विविधतापूर्ण"),
    TestSentence("f5", "इस परियोजना का उद्देश्य ग्रामीण क्षेत्रों का विकास करना है।", "formal",
                 "conjunct क्ष in क्षेत्रों; द्दे in उद्देश्य"),
)

COLLOQUIAL: tuple[TestSentence, ...] = (
    TestSentence("c1", "अरे यार, कल क्या हुआ था वहाँ?", "colloquial",
                 "casual rhythm; nasal वहाँ"),
    TestSentence("c2", "मुझे लगता है कि हमें अभी निकल जाना चाहिए।", "colloquial",
                 "natural connected speech, not word-by-word"),
    TestSentence("c3", "थोड़ा रुको, मैं अभी आया।", "colloquial",
                 "retroflex ड़ (flap) -- a classic English-speaker failure"),
    TestSentence("c4", "क्या तुमने खाना खा लिया?", "colloquial",
                 "question intonation must rise naturally"),
    TestSentence("c5", "बस यही चाहिए था मुझे, और कुछ नहीं।", "colloquial",
                 "phrase-final fall; comma pause"),
    TestSentence("c6", "चलो ठीक है, जैसा तुम कहो।", "colloquial",
                 "retroflex ठ aspirated"),
)

CODE_MIXED: tuple[TestSentence, ...] = (
    TestSentence("m1", "मैंने अभी email भेज दिया है, please check कर लेना।", "code-mixed",
                 "must not switch accent mid-sentence"),
    TestSentence("m2", "कल meeting है सुबह दस बजे, calendar में डाल दो।", "code-mixed",
                 "English nouns inside Hindi syntax"),
    TestSentence("m3", "ये feature अभी beta में है, थोड़ा buggy हो सकता है।", "code-mixed",
                 "three English words, Hindi grammar"),
    TestSentence("m4", "मेरा laptop बहुत slow चल रहा है आजकल।", "code-mixed",
                 "common everyday mixing"),
    TestSentence("m5", "तुम्हारा presentation कैसा गया? sab theek?", "code-mixed",
                 "romanized Hindi at the end -- deliberately tests the pipeline"),
)

NUMERIC: tuple[TestSentence, ...] = (
    TestSentence("n1", "मेरा नंबर 98765 43210 है।", "numeric", "digits must be read in Hindi"),
    TestSentence("n2", "आज 15 अगस्त 2026 है।", "numeric", "date reading"),
    TestSentence("n3", "इसकी कीमत ₹1,299 है।", "numeric", "currency symbol + grouping"),
    TestSentence("n4", "मीटिंग 3:30 बजे शुरू होगी।", "numeric", "time reading"),
    TestSentence("n5", "लगभग 25% लोग सहमत थे।", "numeric", "percent"),
    TestSentence("n6", "तापमान 42.5 डिग्री तक पहुँच गया।", "numeric", "decimal"),
)

ENGLISH_CONTROL: tuple[TestSentence, ...] = (
    TestSentence("e1", "Hello, how can I help you today?", "english",
                 "control: confirms the English path is unaffected"),
)

ALL: tuple[TestSentence, ...] = (
    FORMAL + COLLOQUIAL + CODE_MIXED + NUMERIC + ENGLISH_CONTROL
)

HINDI_ONLY: tuple[TestSentence, ...] = FORMAL + COLLOQUIAL + CODE_MIXED + NUMERIC


def is_devanagari(text: str) -> bool:
    """True if the string contains any Devanagari codepoint."""
    return any("ऀ" <= ch <= "ॿ" for ch in text)


def script_report(text: str) -> dict[str, int]:
    """Count characters by script, to prove nothing was romanized in transit."""
    counts = {"devanagari": 0, "latin": 0, "digit": 0, "other": 0}
    for ch in text:
        if "ऀ" <= ch <= "ॿ":
            counts["devanagari"] += 1
        elif ch.isascii() and ch.isalpha():
            counts["latin"] += 1
        elif ch.isdigit():
            counts["digit"] += 1
        elif not ch.isspace():
            counts["other"] += 1
    return counts
