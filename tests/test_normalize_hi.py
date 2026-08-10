"""Tests for Hindi text normalization.

Two properties matter most: numbers use the Indian system (लाख/करोड़, never
"hundred thousand"), and nothing is ever romanized on the way through.
"""

from __future__ import annotations

import pytest

from voiceagent.eval.sentences import is_devanagari
from voiceagent.text.normalize_hi import (
    digits_individually,
    normalize,
    number_to_hindi,
    time_to_hindi,
)


@pytest.mark.parametrize(
    "value,expected",
    [
        (0, "शून्य"),
        (1, "एक"),
        (19, "उन्नीस"),
        (20, "बीस"),
        (45, "पैंतालीस"),
        (51, "इक्यावन"),
        (99, "निन्यानवे"),
    ],
)
def test_irregular_numbers_below_100(value, expected):
    """0-99 are all irregular in Hindi and must come from the table."""
    assert number_to_hindi(value) == expected


def test_indian_numbering_system():
    """लाख and करोड़, not thousand/million."""
    assert number_to_hindi(100_000) == "एक लाख"
    assert number_to_hindi(10_000_000) == "एक करोड़"
    assert "लाख" in number_to_hindi(1_250_000)
    assert "मिलियन" not in number_to_hindi(1_000_000)


def test_compound_numbers():
    assert number_to_hindi(247) == "दो सौ सैंतालीस"
    assert number_to_hindi(1000) == "एक हज़ार"


# --- time -----------------------------------------------------------------


def test_colloquial_time_forms():
    """Speakers say सवा/साढ़े/पौने far more than the बजकर long form."""
    assert time_to_hindi(3, 0) == "तीन बजे"
    assert time_to_hindi(3, 15) == "सवा तीन बजे"
    assert time_to_hindi(3, 30) == "साढ़े तीन बजे"
    assert time_to_hindi(3, 45) == "पौने चार बजे"


def test_irregular_half_hours():
    """1:30 and 2:30 have their own words; साढ़े एक is wrong."""
    assert time_to_hindi(1, 30) == "डेढ़ बजे"
    assert time_to_hindi(2, 30) == "ढाई बजे"


def test_time_does_not_duplicate_baje():
    """The source text usually already has बजे after the digits."""
    out = normalize("मीटिंग 3:30 बजे शुरू होगी।")
    assert out.count("बजे") == 1
    assert "साढ़े तीन बजे" in out


def test_arbitrary_minutes_use_long_form():
    assert time_to_hindi(4, 20) == "चार बजकर बीस मिनट"


# --- full sentences -------------------------------------------------------


@pytest.mark.parametrize(
    "raw,must_contain",
    [
        ("इसकी कीमत ₹1,299 है।", "रुपये"),
        ("लगभग 25% लोग सहमत थे।", "प्रतिशत"),
        ("तापमान 42.5 डिग्री तक पहुँच गया।", "दशमलव"),
        ("आज 15 अगस्त 2026 है।", "दो हज़ार छब्बीस"),
    ],
)
def test_sentence_normalization(raw, must_contain):
    assert must_contain in normalize(raw)


def test_phone_numbers_are_read_digit_by_digit():
    """A phone number is not a quantity; 98765 is not अट्ठानवे हज़ार."""
    out = normalize("मेरा नंबर 98765 43210 है।")
    assert "नौ आठ सात छह पाँच" in out
    assert "हज़ार" not in out


def test_devanagari_digits_are_handled():
    assert normalize("मेरे पास १५ किताबें हैं।") == normalize("मेरे पास 15 किताबें हैं।")


def test_no_digits_survive():
    """Any leftover digit would be read in the model's default language."""
    for raw in [
        "मेरा नंबर 98765 43210 है।",
        "इसकी कीमत ₹1,299 है।",
        "मीटिंग 3:30 बजे शुरू होगी।",
        "तापमान 42.5 डिग्री तक पहुँच गया।",
    ]:
        assert not any(ch.isdigit() for ch in normalize(raw)), raw


def test_english_words_are_left_alone():
    """Code-mixed text must keep its English; this is not a translator."""
    out = normalize("मैंने अभी email भेज दिया है, please check कर लेना।")
    assert "email" in out and "please check" in out


def test_output_stays_devanagari():
    """Normalization must never romanize the Hindi it touches."""
    out = normalize("इसकी कीमत ₹1,299 है।")
    assert is_devanagari(out)


def test_empty_and_plain_text_pass_through():
    assert normalize("") == ""
    assert normalize("नमस्ते") == "नमस्ते"
