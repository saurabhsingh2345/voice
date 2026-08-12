"""Latin -> Devanagari transliteration for code-mixed Hindi.

The point of these tests is the *distinction* between the two mechanisms: the
loanword table must be exact, because those spellings are conventions a rule
cannot derive; the fallback is only required to be plausible and pronounceable,
because English spelling is not phonetic and no rule set gets it all right.
"""

from __future__ import annotations

import pytest

from voiceagent.text.translit_en import (
    LETTER_NAMES,
    LOANWORDS,
    ROMANIZED_HINDI,
    transliterate,
)

DEVANAGARI = range(0x0900, 0x0980)


def is_devanagari_or_punct(text: str) -> bool:
    return all(ord(ch) in DEVANAGARI or not ch.isalpha() for ch in text)


# --- the table path, which must be exact ---------------------------------


@pytest.mark.parametrize(
    "word,expected",
    [
        ("email", "ईमेल"),
        ("laptop", "लैपटॉप"),
        ("meeting", "मीटिंग"),
        ("calendar", "कैलेंडर"),
        ("presentation", "प्रेज़ेंटेशन"),
        ("feature", "फ़ीचर"),
    ],
)
def test_loanwords_use_conventional_spelling(word, expected):
    """These are the spellings Hindi actually uses, not derivable by rule."""
    assert transliterate(word) == expected


def test_case_is_ignored_on_lookup():
    assert transliterate("Email") == transliterate("EMAIL".lower()) == "ईमेल"


def test_romanized_hindi_is_not_treated_as_english():
    """"theek" is ठीक; read as English it would come out थीक."""
    assert transliterate("sab theek") == "सब ठीक"


# --- acronyms -------------------------------------------------------------


def test_uppercase_acronyms_are_spelled_out():
    assert transliterate("API") == "ए पी आई"
    assert transliterate("PDF") == "पी डी एफ़"


def test_acronyms_said_as_words_are_not_spelled_out():
    assert transliterate("NASA") != "एन ए एस ए"


def test_single_letters_are_letter_names():
    assert transliterate("x") == LETTER_NAMES["x"]


# --- the fallback, which only has to be plausible ------------------------


@pytest.mark.parametrize("word", ["dashboard", "kubernetes", "budget", "random", "python"])
def test_unknown_words_are_approximated_not_dropped(word):
    """The failure being prevented is silence: an unlisted word must still
    produce Devanagari, since IndicF5 skips Latin entirely."""
    out = transliterate(word)
    assert out and is_devanagari_or_punct(out)
    assert not any("a" <= ch.lower() <= "z" for ch in out)


@pytest.mark.parametrize(
    "word,expected",
    [
        ("bus", "बस"),        # short u is the inherent 'a'
        ("sun", "सन"),
        ("city", "सिटी"),      # soft c before i
        ("flight", "फ़्लाइट"),  # silent gh in "igh"
    ],
)
def test_fallback_rules_that_are_worth_pinning(word, expected):
    assert transliterate(word) == expected


# --- what must never be touched ------------------------------------------


def test_devanagari_is_untouched():
    text = "नमस्ते, आज मौसम बहुत सुहावना है।"
    assert transliterate(text) == text


def test_digits_and_symbols_are_left_for_the_number_pass():
    assert transliterate("15 अगस्त 2026") == "15 अगस्त 2026"
    assert transliterate("₹1,299") == "₹1,299"


def test_empty_input():
    assert transliterate("") == ""


# --- table hygiene --------------------------------------------------------


def test_tables_are_lowercase_keyed_and_devanagari_valued():
    """A stray uppercase key would never match, and a Latin value would defeat
    the whole point of the pass."""
    for table in (LOANWORDS, ROMANIZED_HINDI):
        for key, value in table.items():
            assert key == key.lower(), key
            assert is_devanagari_or_punct(value), (key, value)


def test_no_word_is_in_both_tables():
    """A word cannot be both English and romanized Hindi; the overlap would
    make which table wins depend on lookup order."""
    assert not set(LOANWORDS) & set(ROMANIZED_HINDI)
