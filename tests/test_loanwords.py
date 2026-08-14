"""The loanword table, which is the difference between guessing and knowing.

The rule-based Latin->Devanagari fallback is "a floor, not a solution" by the
project's own note: it produces something pronounceable for any word and
something *right* only by luck. Every entry in LOANWORDS is a word the fallback
would otherwise guess at.

These tests are about the table's integrity rather than its taste. A wrong
spelling is a judgement call a native speaker has to make; a duplicate key or a
Latin character hiding in a value is a bug, and it is exactly the kind that
survives review of a 227-entry dict.
"""

from __future__ import annotations

import re

import pytest

from voiceagent.text.translit_en import LOANWORDS, ROMANIZED_HINDI, transliterate

DEVANAGARI = re.compile(r"^[ऀ-ॿ\s]+$")


def test_every_key_is_lowercase():
    """Lookup lowercases the word first, so an uppercase key is dead weight that
    silently never matches."""
    wrong = [k for k in LOANWORDS if k != k.lower()]
    assert not wrong, wrong


def test_every_value_is_devanagari():
    """A Latin character surviving in a value defeats the entire point: the
    engine would receive the thing transliteration exists to remove."""
    wrong = {k: v for k, v in LOANWORDS.items() if not DEVANAGARI.match(v)}
    assert not wrong, wrong


def test_no_value_is_empty():
    assert not [k for k, v in LOANWORDS.items() if not v.strip()]


def test_keys_do_not_collide_with_romanized_hindi():
    """The two tables are consulted for different reasons -- one is English
    spelled in Hindi, the other is Hindi spelled in Latin -- and a word in both
    means one of them is wrong about what language it is."""
    overlap = set(LOANWORDS) & set(ROMANIZED_HINDI)
    assert not overlap, overlap


def test_the_table_covers_the_verticals_the_plan_targets():
    """Not a size check for its own sake. These are the domains plan.md names,
    and an empty domain means a demo there falls back to guessing."""
    for domain, words in {
        "banking": ["loan", "premium", "statement", "transaction", "refund"],
        "health": ["medicine", "appointment", "prescription", "treatment"],
        "education": ["exam", "homework", "chapter", "syllabus"],
        "logistics": ["delivery", "discount", "product", "bill"],
        "the app itself": ["microphone", "recording", "notification", "language"],
    }.items():
        missing = [w for w in words if w not in LOANWORDS]
        assert not missing, f"{domain}: {missing}"


@pytest.mark.parametrize(
    "text,expected",
    [
        ("मेरा loan approve हो गया", "लोन"),
        ("exam का result आ गया", "एग्ज़ाम"),
        ("insurance premium भर दिया", "प्रीमियम"),
    ],
)
def test_known_words_use_the_table_not_the_fallback(text, expected):
    assert expected in transliterate(text)


def test_an_unknown_word_still_gets_something_speakable():
    """The floor. An unlisted word must not vanish -- a dropped word is worse
    than an approximate one, because the sentence changes meaning silently."""
    out = transliterate("यह एक quixotic विचार है")
    assert "quixotic" not in out
    assert DEVANAGARI.match(out.replace("यह एक ", "").replace(" विचार है", "").strip())


def test_devanagari_input_is_untouched():
    hindi = "आज मौसम बहुत सुहावना है।"
    assert transliterate(hindi) == hindi
