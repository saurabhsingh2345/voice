"""Numbers to English words, so the last LGPL dependency can go.

`misaki.en` does `from num2words import num2words` at module scope, which is why
`num2words` is a hard dependency of this project rather than an optional one. It
is LGPL-2.1, and `models.ACCEPTED_EXCEPTIONS` has carried it as the single
recorded exception to the permissive-only rule.

The LGPL permits commercial use of an unmodified library, so this was never a
licensing *risk*. It carries one obligation the Apache/MIT/BSD set does not:
recipients must be able to replace the library with their own version. An
ordinary importable package in site-packages satisfies that. Freezing Python into
one opaque binary does not -- and that is precisely what a self-contained desktop
bundle is. So this dependency was the thing standing between the project and its
own packaging goal, not an abstract compliance worry.

This module is the replacement. `text.num2words_shim` registers it under the name
`num2words` before `misaki.en` is imported, the same trick already used to keep
the GPL espeak fallback out (see `tts.kokoro_engine._disable_gpl_espeak_fallback`).

PARITY, NOT REIMPLEMENTATION FROM THE SPEC

The output has to match what Kokoro was tuned against, so this was written
against the real library and diffed over every integer to 10,000, a spread of
large values, all ordinals, and floats -- see `tests/test_numbers.py`, which keeps
the golden cases. Details that look like quirks are the library's conventions and
are deliberate:

  * "one hundred and twenty-three" -- British `and`, which US English usually
    drops. Kokoro's dictionary was built with it.
  * Commas between scale groups: "one million, two hundred thousand".
  * Years read as pairs -- 1999 is "nineteen ninety-nine", not "one thousand
    nine hundred and ninety-nine" -- but 2005 is "two thousand and five".

Only the four forms `misaki.en` actually calls are supported: cardinal, ordinal,
year, and float. Anything else raises rather than guessing, because a silent
wrong reading is worse than a crash in a text frontend.
"""

from __future__ import annotations

ONES = (
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
    "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
    "sixteen", "seventeen", "eighteen", "nineteen",
)
TENS = (
    "", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy",
    "eighty", "ninety",
)
#: Short scale, matching the library's `lang='en'`.
SCALES = (
    (10**33, "decillion"), (10**30, "nonillion"), (10**27, "octillion"),
    (10**24, "septillion"), (10**21, "sextillion"), (10**18, "quintillion"),
    (10**15, "quadrillion"), (10**12, "trillion"), (10**9, "billion"),
    (10**6, "million"), (10**3, "thousand"),
)

#: Ordinals that are not formed by suffixing the cardinal.
ORDINAL_ONES = {
    "one": "first", "two": "second", "three": "third", "five": "fifth",
    "eight": "eighth", "nine": "ninth", "twelve": "twelfth",
}


def _under_thousand(number: int) -> str:
    """1-999. The `and` before the tens is the British convention the library uses."""
    if number < 20:
        return ONES[number]
    if number < 100:
        tens, ones = divmod(number, 10)
        return TENS[tens] + (f"-{ONES[ones]}" if ones else "")
    hundreds, rest = divmod(number, 100)
    word = f"{ONES[hundreds]} hundred"
    return f"{word} and {_under_thousand(rest)}" if rest else word


def cardinal(number: int) -> str:
    """123 -> "one hundred and twenty-three"."""
    if number < 0:
        return f"minus {cardinal(-number)}"
    if number < 1000:
        return _under_thousand(number)

    groups: list[str] = []
    remainder = number
    for value, name in SCALES:
        if remainder >= value:
            count, remainder = divmod(remainder, value)
            groups.append(f"{cardinal(count)} {name}")
    if remainder:
        # "and" only when the tail is under 100: the library writes
        # "one thousand and one" but "one thousand, one hundred and one".
        if remainder < 100 and groups:
            return ", ".join(groups) + f" and {_under_thousand(remainder)}"
        groups.append(_under_thousand(remainder))
    return ", ".join(groups)


def ordinal(number: int) -> str:
    """21 -> "twenty-first". Suffixes the last word of the cardinal."""
    if number < 0:
        return f"minus {ordinal(-number)}"
    words = cardinal(number)
    head, separator, last = _split_last_word(words)
    if last in ORDINAL_ONES:
        last = ORDINAL_ONES[last]
    elif last.endswith("y"):
        last = last[:-1] + "ieth"
    else:
        last = last + "th"
    return head + separator + last


def _split_last_word(words: str) -> tuple[str, str, str]:
    """Split on the final hyphen or space, so "twenty-one" suffixes as "-first"."""
    for index in range(len(words) - 1, -1, -1):
        if words[index] in "- ":
            return words[:index], words[index], words[index + 1 :]
    return "", "", words


#: Years outside this band are read as plain cardinals rather than as pairs.
#: Both edges were found by probing the library, not assumed: 99 is "ninety-nine"
#: and 10010 is "ten thousand and ten", while 110 is "one ten" and 2100 is
#: "twenty-one hundred".
YEAR_PAIR_MIN = 100
YEAR_PAIR_MAX = 9999


def year(number: int) -> str:
    """1999 -> "nineteen ninety-nine". 2005 -> "two thousand and five".

    Read as a pair of two-digit numbers, which is how a year is said. Three
    exceptions, all the library's and all matching speech:

      * outside YEAR_PAIR_MIN..YEAR_PAIR_MAX, a plain cardinal
      * a x000-x009 band, so 2005 is "two thousand and five" and not "twenty oh
        five" -- but 2010 is "twenty ten"
      * a round century, so 1900 is "nineteen hundred"

    The 1010 boundary is the one worth naming: 1009 is "one thousand and nine"
    and 1010 is "ten ten". An earlier version of this used 1100 as the cutoff and
    was wrong for the whole of 1010-1099.
    """
    if number < 0:
        return f"{year(-number)} BC"
    if not (YEAR_PAIR_MIN <= number <= YEAR_PAIR_MAX):
        return cardinal(number)
    if number >= 1000 and (number % 1000) < 10:
        return cardinal(number)
    high, low = divmod(number, 100)
    if low == 0:
        return f"{cardinal(high)} hundred"
    if low < 10:
        return f"{cardinal(high)} oh-{ONES[low]}"
    return f"{cardinal(high)} {_under_thousand(low)}"


def decimal(value: float) -> str:
    """1.5 -> "one point five". Digits after the point are read individually."""
    text = repr(float(value))
    if "e" in text or "E" in text:
        text = f"{float(value):.10f}".rstrip("0").rstrip(".")
    negative = text.startswith("-")
    text = text.lstrip("-")
    whole, _, fraction = text.partition(".")
    words = cardinal(int(whole or 0))
    fraction = fraction.rstrip("0")
    if fraction:
        words += " point " + " ".join(ONES[int(digit)] for digit in fraction)
    return f"minus {words}" if negative else words


def num2words(number, to: str = "cardinal", **kwargs) -> str:
    """The entry point `misaki.en` imports. Same call signature, same output.

    `lang` is accepted and ignored: misaki never passes it, and silently
    producing English for a caller who asked for French would be worse than the
    TypeError they get today.
    """
    if kwargs.get("lang", "en") not in ("en", "en_GB", "en_IN"):
        raise NotImplementedError(
            f"this replacement covers English only, not {kwargs['lang']!r}. "
            "See voiceagent.text.numbers for why it exists."
        )
    if to == "cardinal":
        return decimal(number) if isinstance(number, float) else cardinal(int(number))
    if to == "ordinal":
        return ordinal(int(number))
    if to == "year":
        return year(int(number))
    raise NotImplementedError(
        f"num2words(to={to!r}) is not supported. This is a deliberate subset -- "
        "misaki.en calls cardinal, ordinal and year, and guessing at the rest "
        "would put a silently wrong reading into speech."
    )
