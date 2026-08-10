"""Hindi text normalization for TTS.

A TTS model reads what it is given. Hand it "₹1,299" or "3:30" and it will
either spell the symbols out in English order or skip them, which is a large
part of why synthesized Hindi sounds translated rather than spoken.

Hindi numbers are not compositional the way English ones are: every value from
0-99 has its own irregular name (इक्यावन, not "पचास एक"), so they have to be
tabulated rather than generated. Above that, Hindi uses the Indian system --
हज़ार, लाख, करोड़ -- not thousand/million/billion, so 1,00,000 is एक लाख and
never "one hundred thousand".

Devanagari in, Devanagari out. Nothing here romanizes anything.
"""

from __future__ import annotations

import re

# --- numbers --------------------------------------------------------------

#: Every Hindi number 0-99 is irregular; there is no rule to generate them.
ONES_TO_99 = [
    "शून्य", "एक", "दो", "तीन", "चार", "पाँच", "छह", "सात", "आठ", "नौ",
    "दस", "ग्यारह", "बारह", "तेरह", "चौदह", "पंद्रह", "सोलह", "सत्रह", "अठारह", "उन्नीस",
    "बीस", "इक्कीस", "बाईस", "तेईस", "चौबीस", "पच्चीस", "छब्बीस", "सत्ताईस", "अट्ठाईस", "उनतीस",
    "तीस", "इकतीस", "बत्तीस", "तैंतीस", "चौंतीस", "पैंतीस", "छत्तीस", "सैंतीस", "अड़तीस", "उनतालीस",
    "चालीस", "इकतालीस", "बयालीस", "तैंतालीस", "चवालीस", "पैंतालीस", "छियालीस", "सैंतालीस", "अड़तालीस", "उनचास",
    "पचास", "इक्यावन", "बावन", "तिरपन", "चौवन", "पचपन", "छप्पन", "सत्तावन", "अट्ठावन", "उनसठ",
    "साठ", "इकसठ", "बासठ", "तिरसठ", "चौंसठ", "पैंसठ", "छियासठ", "सड़सठ", "अड़सठ", "उनहत्तर",
    "सत्तर", "इकहत्तर", "बहत्तर", "तिहत्तर", "चौहत्तर", "पचहत्तर", "छिहत्तर", "सतहत्तर", "अठहत्तर", "उन्यासी",
    "अस्सी", "इक्यासी", "बयासी", "तिरासी", "चौरासी", "पचासी", "छियासी", "सत्तासी", "अठासी", "नवासी",
    "नब्बे", "इक्यानवे", "बानवे", "तिरानवे", "चौरानवे", "पचानवे", "छियानवे", "सत्तानवे", "अट्ठानवे", "निन्यानवे",
]

#: Devanagari digits map to the same values as ASCII ones.
DEVANAGARI_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")


def number_to_hindi(value: int) -> str:
    """Render an integer in spoken Hindi using the Indian numbering system."""
    if value < 0:
        return "ऋण " + number_to_hindi(-value)
    if value < 100:
        return ONES_TO_99[value]

    parts: list[str] = []
    # करोड़ = 10^7, लाख = 10^5 -- the Indian grouping, not thousand/million.
    for divisor, name in ((10_000_000, "करोड़"), (100_000, "लाख"), (1_000, "हज़ार"), (100, "सौ")):
        if value >= divisor:
            count = value // divisor
            value %= divisor
            parts.append(f"{number_to_hindi(count)} {name}")

    if value:
        parts.append(number_to_hindi(value))
    return " ".join(parts)


def digits_individually(digits: str) -> str:
    """Read a digit string one digit at a time, as phone numbers are spoken."""
    return " ".join(ONES_TO_99[int(d)] for d in digits if d.isdigit())


# --- dates and time -------------------------------------------------------

MONTHS = {
    1: "जनवरी", 2: "फ़रवरी", 3: "मार्च", 4: "अप्रैल", 5: "मई", 6: "जून",
    7: "जुलाई", 8: "अगस्त", 9: "सितंबर", 10: "अक्टूबर", 11: "नवंबर", 12: "दिसंबर",
}

#: Hindi has dedicated words for quarter/half past, and speakers use them far
#: more than "बजकर पंद्रह मिनट". Using the long form everywhere sounds robotic.
QUARTER_FORMS = {
    0: "{h} बजे",
    15: "सवा {h} बजे",
    30: "साढ़े {h} बजे",
}


def time_to_hindi(hour: int, minute: int) -> str:
    hour_12 = hour % 12 or 12
    if minute == 45:
        # पौने is "quarter to", so it takes the NEXT hour.
        nxt = (hour_12 % 12) + 1
        return f"पौने {ONES_TO_99[nxt]} बजे"
    if minute in QUARTER_FORMS:
        # साढ़े एक/दो are irregular in speech (डेढ़/ढाई), handled below.
        if minute == 30 and hour_12 == 1:
            return "डेढ़ बजे"
        if minute == 30 and hour_12 == 2:
            return "ढाई बजे"
        return QUARTER_FORMS[minute].format(h=ONES_TO_99[hour_12])
    return f"{ONES_TO_99[hour_12]} बजकर {number_to_hindi(minute)} मिनट"


# --- the normalizer -------------------------------------------------------

#: Swallow a बजे that already follows the digits. time_to_hindi always emits
#: one, and without this "3:30 बजे" becomes "साढ़े तीन बजे बजे".
_TIME = re.compile(r"\b(\d{1,2}):(\d{2})\b(?:\s*बजे)?")
_DATE_DMY = re.compile(r"\b(\d{1,2})[/-](\d{1,2})[/-](\d{4})\b")
_CURRENCY = re.compile(r"₹\s?([\d,]+(?:\.\d+)?)")
_PERCENT = re.compile(r"(\d+(?:\.\d+)?)\s?%")
_PHONE = re.compile(r"\b(\d{5})[\s-]?(\d{5})\b")
_DECIMAL = re.compile(r"\b(\d+)\.(\d+)\b")
_GROUPED = re.compile(r"\b\d{1,3}(?:,\d{2,3})+\b")
_PLAIN = re.compile(r"\b\d+\b")

#: Abbreviations that a TTS would otherwise spell out letter by letter.
ABBREVIATIONS = {
    "डॉ.": "डॉक्टर",
    "श्री": "श्री",
    "प्रो.": "प्रोफ़ेसर",
    "आदि": "आदि",
    "ई.": "ईस्वी",
}


def normalize(text: str) -> str:
    """Convert a Hindi string into something a TTS model can read aloud."""
    if not text:
        return text

    # Devanagari digits behave like ASCII ones for every rule below.
    text = text.translate(DEVANAGARI_DIGITS)

    for abbrev, expansion in ABBREVIATIONS.items():
        text = text.replace(abbrev, expansion)

    # Order matters: phone numbers and times before generic digits, otherwise
    # 98765 becomes "अट्ठानवे हज़ार..." instead of being read digit by digit.
    text = _PHONE.sub(lambda m: digits_individually(m.group(1) + m.group(2)), text)
    text = _TIME.sub(lambda m: time_to_hindi(int(m.group(1)), int(m.group(2))), text)
    text = _DATE_DMY.sub(
        lambda m: f"{number_to_hindi(int(m.group(1)))} {MONTHS.get(int(m.group(2)), '')} "
        f"{number_to_hindi(int(m.group(3)))}",
        text,
    )
    text = _CURRENCY.sub(
        lambda m: f"{number_to_hindi(int(float(m.group(1).replace(',', ''))))} रुपये", text
    )
    text = _PERCENT.sub(lambda m: f"{_spoken_number(m.group(1))} प्रतिशत", text)
    text = _DECIMAL.sub(
        lambda m: f"{number_to_hindi(int(m.group(1)))} दशमलव {digits_individually(m.group(2))}",
        text,
    )
    text = _GROUPED.sub(lambda m: number_to_hindi(int(m.group(0).replace(",", ""))), text)
    text = _PLAIN.sub(lambda m: number_to_hindi(int(m.group(0))), text)

    return re.sub(r"\s{2,}", " ", text).strip()


def _spoken_number(raw: str) -> str:
    if "." in raw:
        whole, frac = raw.split(".", 1)
        return f"{number_to_hindi(int(whole))} दशमलव {digits_individually(frac)}"
    return number_to_hindi(int(raw))
