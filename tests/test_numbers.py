"""The replacement for num2words, and the parity that lets it replace it.

`num2words` was this project's only accepted licence exception: LGPL-2.1,
imported by `misaki.en` at module scope to turn digits into words for Kokoro. The
LGPL permits commercial use of an unmodified library, so it was never a risk
while Python stayed an ordinary importable tree. It becomes one the moment
Python is frozen into a single opaque binary -- recipients must be able to
replace the library, and they cannot replace what is welded shut. That is exactly
what the desktop bundle does, so the exception had to go rather than be renewed.

These cases were captured by diffing against the real library before it was
uninstalled: every integer to 10,000, thousands of large values, every ordinal to
2,000, every year to 20,000, and several thousand floats -- all matching. What is
kept here is the interesting subset, chosen so a regression shows up as a wrong
*word* rather than a wrong count.
"""

from __future__ import annotations

import sys

import pytest

from voiceagent.text.numbers import cardinal, decimal, num2words, ordinal, year


# --- cardinals --------------------------------------------------------------


@pytest.mark.parametrize(
    "number,expected",
    [
        (0, "zero"),
        (7, "seven"),
        (13, "thirteen"),
        (20, "twenty"),
        (21, "twenty-one"),
        (100, "one hundred"),
        # British "and", which US English usually drops. Kokoro's dictionary was
        # built with it, so dropping it would change the phonemes.
        (123, "one hundred and twenty-three"),
        (1000, "one thousand"),
        (1001, "one thousand and one"),
        # ... but a hundreds group turns the "and" into a comma-joined group.
        (1101, "one thousand, one hundred and one"),
        (1234, "one thousand, two hundred and thirty-four"),
        (1000000, "one million"),
        (1234567, "one million, two hundred and thirty-four thousand, five hundred and sixty-seven"),
        (-42, "minus forty-two"),
    ],
)
def test_cardinal(number, expected):
    assert cardinal(number) == expected


# --- ordinals ---------------------------------------------------------------


@pytest.mark.parametrize(
    "number,expected",
    [
        (1, "first"), (2, "second"), (3, "third"), (4, "fourth"), (5, "fifth"),
        (8, "eighth"), (9, "ninth"), (12, "twelfth"), (20, "twentieth"),
        (21, "twenty-first"), (100, "one hundredth"), (1000000, "one millionth"),
        # The suffix attaches to the last word across a hyphen, not a space.
        (42, "forty-second"),
    ],
)
def test_ordinal(number, expected):
    assert ordinal(number) == expected


# --- years ------------------------------------------------------------------


@pytest.mark.parametrize(
    "number,expected",
    [
        (1999, "nineteen ninety-nine"),
        (1900, "nineteen hundred"),
        (1905, "nineteen oh-five"),
        (2010, "twenty ten"),
        (2100, "twenty-one hundred"),
        (110, "one ten"),
        (101, "one oh-one"),
        # A x000-x009 band reads as a cardinal, not as a pair.
        (2000, "two thousand"),
        (2005, "two thousand and five"),
        # The boundary an earlier version of this got wrong: it used 1100 as the
        # cutoff and was wrong for the whole of 1010-1099.
        (1009, "one thousand and nine"),
        (1010, "ten ten"),
        # Outside the pair band entirely.
        (99, "ninety-nine"),
        (10010, "ten thousand and ten"),
    ],
)
def test_year(number, expected):
    assert year(number) == expected


# --- decimals ---------------------------------------------------------------


@pytest.mark.parametrize(
    "value,expected",
    [
        (1.5, "one point five"),
        (0.25, "zero point two five"),
        (2.0, "two"),
        (-3.5, "minus three point five"),
        (1234.56, "one thousand, two hundred and thirty-four point five six"),
    ],
)
def test_decimal(value, expected):
    assert decimal(value) == expected


# --- the entry point misaki imports -----------------------------------------


def test_the_four_forms_misaki_calls():
    assert num2words(21) == "twenty-one"
    assert num2words(21, to="cardinal") == "twenty-one"
    assert num2words(21, to="ordinal") == "twenty-first"
    assert num2words(1999, to="year") == "nineteen ninety-nine"
    assert num2words(1.5) == "one point five"


def test_an_unsupported_form_raises_rather_than_guessing():
    """A silently wrong reading is worse than a crash in a text frontend: it
    reaches the listener sounding confident."""
    with pytest.raises(NotImplementedError):
        num2words(5, to="currency")


def test_a_non_english_language_raises():
    with pytest.raises(NotImplementedError):
        num2words(5, lang="fr")


# --- the shim ---------------------------------------------------------------


def test_the_shim_satisfies_misakis_import():
    """`misaki.en` does `from num2words import num2words` at module scope."""
    from voiceagent.text.num2words_shim import install, is_installed

    install()
    assert is_installed()
    from num2words import num2words as imported

    assert imported(1999, to="year") == "nineteen ninety-nine"


def test_install_is_idempotent():
    from voiceagent.text.num2words_shim import install

    install()
    assert install() is False, "must not replace an already-registered module"


def test_the_lgpl_package_is_not_installed():
    """The point of all of this. If it comes back as a transitive dependency, a
    frozen bundle ships an LGPL relink obligation it cannot satisfy."""
    import importlib.metadata as md

    names = {(d.metadata["Name"] or "").lower() for d in md.distributions()}
    assert "num2words" not in names


def test_there_are_no_accepted_licence_exceptions_left():
    from voiceagent.models import ACCEPTED_EXCEPTIONS

    assert ACCEPTED_EXCEPTIONS == {}
