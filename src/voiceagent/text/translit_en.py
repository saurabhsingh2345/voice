"""Latin script -> Devanagari, so an Indic TTS can read code-mixed Hindi.

Real Indian speech mixes English into Hindi constantly ("मेरा laptop बहुत slow
चल रहा है"). IndicF5 cannot pronounce those words. Its vocab *does* contain all
52 ASCII letters, so nothing is out of vocabulary and nothing errors -- the
model simply has no acoustic mapping for English orthography, having been
trained on Indic script. Measured by round-trip transcription, the English words
came back dropped or mangled while the surrounding Hindi scored 88-100%:

    "कल meeting है सुबह दस बजे, calendar में डाल दो।"
      -> heard: "सुकबह दस बजे एउननर में डाल दो"   (meeting gone, calendar -> एउननर)

So the words are rewritten into the script the model can actually read.

Two mechanisms, in this order, because they fail differently:

  1. A LOANWORD table. This carries the load, and it is not a shortcut: English
     loanwords in Hindi have *conventional* Devanagari spellings that reflect how
     Indians actually say them, not how a transliterator would derive them.
     "email" is ईमेल, never एमैल. No rule produces that from the spelling.
  2. A rule-based fallback for anything unlisted, so an unknown word is
     approximated rather than silently dropped. English spelling is not phonetic,
     so this is genuinely lossy -- "though" and "tough" cannot both come out
     right. It is a floor, not a solution; add frequent words to the table.

Romanized *Hindi* is handled too ("sab theek" -> "सब ठीक"), since users type it
and it is indistinguishable from English by script alone.

Devanagari input is never touched.
"""

from __future__ import annotations

import re

# --- 1. conventional spellings -------------------------------------------

#: English loanwords by their established Hindi spelling. Lowercase keys.
#: These are the spellings Hindi speakers use, which a transliterator cannot
#: derive: "feature" -> फ़ीचर depends on knowing the vowel is long.
#:
#: The rule-based fallback below is "a floor, not a solution": it produces
#: something pronounceable for any Latin word, and something *right* only by
#: luck. Every entry here is a word the fallback would otherwise guess at, and
#: the measured cost of guessing is real -- transliterating before synthesis
#: moved the code-mixed held-out sentence from 94 % to 98 % round-trip overlap.
#:
#: Grouped by domain rather than alphabetically, because the useful question
#: when extending this is "what does this vertical talk about" and not "what
#: comes after M".
#:
#: AUTHORSHIP CAVEAT, as in `train/prompts.py` and `eval/heldout.py`: the
#: spellings below were written by the assistant, not a native speaker. They
#: follow common usage, but a Hindi speaker should read the list -- a wrong
#: spelling here is worse than no entry, because it silently overrides the
#: fallback and is never revisited.
LOANWORDS: dict[str, str] = {
    # communication / office
    "email": "ईमेल",
    "mail": "मेल",
    "message": "मैसेज",
    "meeting": "मीटिंग",
    "calendar": "कैलेंडर",
    "presentation": "प्रेज़ेंटेशन",
    "report": "रिपोर्ट",
    "office": "ऑफ़िस",
    "call": "कॉल",
    "please": "प्लीज़",
    "check": "चेक",
    "update": "अपडेट",
    "schedule": "शेड्यूल",
    "reminder": "रिमाइंडर",
    "document": "डॉक्युमेंट",
    "file": "फ़ाइल",
    "folder": "फ़ोल्डर",
    "project": "प्रोजेक्ट",
    "deadline": "डेडलाइन",
    "team": "टीम",
    "client": "क्लाइंट",
    # tech
    "laptop": "लैपटॉप",
    "computer": "कंप्यूटर",
    "mobile": "मोबाइल",
    "phone": "फ़ोन",
    "internet": "इंटरनेट",
    "wifi": "वाईफ़ाई",
    "network": "नेटवर्क",
    "server": "सर्वर",
    "software": "सॉफ़्टवेयर",
    "hardware": "हार्डवेयर",
    "app": "ऐप",
    "website": "वेबसाइट",
    "browser": "ब्राउज़र",
    "password": "पासवर्ड",
    "screen": "स्क्रीन",
    "battery": "बैटरी",
    "charger": "चार्जर",
    "feature": "फ़ीचर",
    "beta": "बीटा",
    "buggy": "बगी",
    "bug": "बग",
    "code": "कोड",
    "data": "डेटा",
    "online": "ऑनलाइन",
    "offline": "ऑफ़लाइन",
    "download": "डाउनलोड",
    "upload": "अपलोड",
    "link": "लिंक",
    "video": "वीडियो",
    "photo": "फ़ोटो",
    "camera": "कैमरा",
    "system": "सिस्टम",
    "version": "वर्ज़न",
    "setting": "सेटिंग",
    "settings": "सेटिंग्स",
    # everyday
    "slow": "स्लो",
    "fast": "फ़ास्ट",
    "ok": "ओके",
    "okay": "ओके",
    "sorry": "सॉरी",
    "thanks": "थैंक्स",
    "bye": "बाय",
    "hello": "हैलो",
    "market": "मार्केट",
    "ticket": "टिकट",
    "train": "ट्रेन",
    "bus": "बस",
    "car": "कार",
    "doctor": "डॉक्टर",
    "hospital": "हॉस्पिटल",
    "school": "स्कूल",
    "college": "कॉलेज",
    "book": "बुक",
    "party": "पार्टी",
    "birthday": "बर्थडे",
    "weekend": "वीकेंड",
    "holiday": "हॉलिडे",
    "problem": "प्रॉब्लम",
    "tension": "टेंशन",
    "time": "टाइम",
    "minute": "मिनट",
    "hour": "आवर",
    "morning": "मॉर्निंग",
    "night": "नाइट",
    "water": "वॉटर",
    "tea": "टी",
    "coffee": "कॉफ़ी",
    "food": "फ़ूड",
    "money": "मनी",
    "bank": "बैंक",
    "account": "अकाउंट",
    "payment": "पेमेंट",
    "order": "ऑर्डर",
    "address": "एड्रेस",
    "number": "नंबर",

    # --- banking and money -----------------------------------------------
    # The vertical `plan.md` puts first: BFSI is where "the data cannot leave
    # the building" is a procurement rule rather than a preference, so this is
    # the vocabulary a demo there actually needs.
    "loan": "लोन",
    "interest": "इंटरेस्ट",
    "credit": "क्रेडिट",
    "debit": "डेबिट",
    "card": "कार्ड",
    "balance": "बैलेंस",
    "transfer": "ट्रांसफ़र",
    "statement": "स्टेटमेंट",
    "insurance": "इंश्योरेंस",
    "policy": "पॉलिसी",
    "premium": "प्रीमियम",
    "claim": "क्लेम",
    "branch": "ब्रांच",
    "cheque": "चेक",
    "transaction": "ट्रांज़ैक्शन",
    "refund": "रिफ़ंड",
    "deposit": "डिपॉज़िट",
    "savings": "सेविंग्स",
    "salary": "सैलरी",
    "tax": "टैक्स",
    "invoice": "इनवॉइस",
    "budget": "बजट",
    "fund": "फ़ंड",
    "investment": "इन्वेस्टमेंट",
    "installment": "इंस्टॉलमेंट",
    "limit": "लिमिट",
    "penalty": "पेनल्टी",
    "verification": "वेरिफ़िकेशन",
    "application": "एप्लिकेशन",
    "approval": "अप्रूवल",
    "form": "फ़ॉर्म",
    "signature": "सिग्नेचर",
    "customer": "कस्टमर",
    "service": "सर्विस",
    "support": "सपोर्ट",
    "complaint": "कंप्लेंट",
    "request": "रिक्वेस्ट",
    "status": "स्टेटस",

    # --- health ------------------------------------------------------------
    "medicine": "मेडिसिन",
    "tablet": "टैबलेट",
    "appointment": "अपॉइंटमेंट",
    "clinic": "क्लिनिक",
    "test": "टेस्ट",
    "patient": "पेशेंट",
    "treatment": "ट्रीटमेंट",
    "surgery": "सर्जरी",
    "nurse": "नर्स",
    "emergency": "इमरजेंसी",
    "prescription": "प्रिस्क्रिप्शन",
    "checkup": "चेकअप",
    "injection": "इंजेक्शन",

    # --- education ---------------------------------------------------------
    "class": "क्लास",
    "teacher": "टीचर",
    "student": "स्टूडेंट",
    "exam": "एग्ज़ाम",
    "subject": "सब्जेक्ट",
    "chapter": "चैप्टर",
    "question": "क्वेश्चन",
    "answer": "आंसर",
    "notes": "नोट्स",
    "admission": "एडमिशन",
    "homework": "होमवर्क",
    "course": "कोर्स",
    "degree": "डिग्री",
    "result": "रिजल्ट",
    "marks": "मार्क्स",
    "syllabus": "सिलेबस",
    "practice": "प्रैक्टिस",
    "assignment": "असाइनमेंट",
    "library": "लाइब्रेरी",

    # --- shops, orders, logistics ------------------------------------------
    "price": "प्राइस",
    "discount": "डिस्काउंट",
    "delivery": "डिलीवरी",
    "shop": "शॉप",
    "product": "प्रोडक्ट",
    "quality": "क्वालिटी",
    "stock": "स्टॉक",
    "bill": "बिल",
    "cash": "कैश",
    "offer": "ऑफ़र",
    "brand": "ब्रांड",
    "size": "साइज़",
    "model": "मॉडल",

    # --- travel --------------------------------------------------------------
    "station": "स्टेशन",
    "flight": "फ़्लाइट",
    "hotel": "होटल",
    "airport": "एयरपोर्ट",
    "platform": "प्लेटफ़ॉर्म",
    "booking": "बुकिंग",
    "seat": "सीट",
    "luggage": "लगेज",
    "driver": "ड्राइवर",
    "road": "रोड",
    "traffic": "ट्रैफ़िक",
    "signal": "सिग्नल",
    "parking": "पार्किंग",
    "license": "लाइसेंस",

    # --- work ----------------------------------------------------------------
    "manager": "मैनेजर",
    "target": "टारगेट",
    "review": "रिव्यू",
    "feedback": "फ़ीडबैक",
    "training": "ट्रेनिंग",
    "interview": "इंटरव्यू",
    "contract": "कॉन्ट्रैक्ट",
    "department": "डिपार्टमेंट",
    "staff": "स्टाफ़",
    "shift": "शिफ़्ट",
    "experience": "एक्सपीरियंस",
    "company": "कंपनी",
    "business": "बिज़नेस",
    "plan": "प्लान",
    "process": "प्रोसेस",
    "task": "टास्क",
    "list": "लिस्ट",
    "group": "ग्रुप",

    # --- using the app itself ------------------------------------------------
    # What a user says *to* this assistant, which is the one domain guaranteed
    # to come up no matter which vertical it is sold into.
    "search": "सर्च",
    "save": "सेव",
    "delete": "डिलीट",
    "share": "शेयर",
    "click": "क्लिक",
    "install": "इंस्टॉल",
    "backup": "बैकअप",
    "login": "लॉगिन",
    "logout": "लॉगआउट",
    "profile": "प्रोफ़ाइल",
    "notification": "नोटिफ़िकेशन",
    "microphone": "माइक्रोफ़ोन",
    "speaker": "स्पीकर",
    "recording": "रिकॉर्डिंग",
    "voice": "वॉइस",
    "language": "लैंग्वेज",
}

#: Romanized Hindi. Users type this, and it is not English -- transliterating it
#: as English gives the wrong vowels ("theek" -> थीक rather than ठीक).
ROMANIZED_HINDI: dict[str, str] = {
    "sab": "सब",
    "theek": "ठीक",
    "thik": "ठीक",
    "hai": "है",
    "haan": "हाँ",
    "han": "हाँ",
    "nahi": "नहीं",
    "nahin": "नहीं",
    "kya": "क्या",
    "kaise": "कैसे",
    "kaisa": "कैसा",
    "accha": "अच्छा",
    "acha": "अच्छा",
    "bahut": "बहुत",
    "thoda": "थोड़ा",
    "yaar": "यार",
    "bhai": "भाई",
    "arre": "अरे",
    "abhi": "अभी",
    "kal": "कल",
    "aaj": "आज",
    "matlab": "मतलब",
    "bilkul": "बिल्कुल",
    "shukriya": "शुक्रिया",
    "namaste": "नमस्ते",
    "dhanyavaad": "धन्यवाद",
    "chalo": "चलो",
    "karo": "करो",
    "kar": "कर",
    "hua": "हुआ",
    "mera": "मेरा",
    "tera": "तेरा",
    "tumhara": "तुम्हारा",
}

#: Letter names, for acronyms read one letter at a time ("API" -> ए पी आई).
LETTER_NAMES: dict[str, str] = {
    "a": "ए", "b": "बी", "c": "सी", "d": "डी", "e": "ई", "f": "एफ़",
    "g": "जी", "h": "एच", "i": "आई", "j": "जे", "k": "के", "l": "एल",
    "m": "एम", "n": "एन", "o": "ओ", "p": "पी", "q": "क्यू", "r": "आर",
    "s": "एस", "t": "टी", "u": "यू", "v": "वी", "w": "डब्ल्यू",
    "x": "एक्स", "y": "वाई", "z": "ज़ेड",
}

#: Acronyms that are said as words, so they must NOT be spelled out. Note that
#: PDF, ATM and GPS are *not* here: Hindi speakers spell those out letter by
#: letter (पी डी एफ़), which is the default path.
SPOKEN_AS_WORD = {"nasa", "unesco", "wifi", "jpeg", "sim"}

# --- 2. rule-based fallback ----------------------------------------------

#: Multi-character graphemes, longest first. Order is significant.
_DIGRAPHS: tuple[tuple[str, str, str], ...] = (
    # (spelling, consonant-with-halant, independent form)
    ("tion", "श्", "शन"),
    ("sion", "श्", "शन"),
    # "ough" is deliberately absent: it is the most inconsistent string in
    # English spelling ("though", "tough", "through" all differ), so any single
    # mapping is wrong more often than right. Let it fall through.
    ("ch", "च्", "च"),
    ("sh", "श्", "श"),
    ("th", "थ्", "थ"),
    ("ph", "फ़्", "फ़"),
    ("gh", "घ्", "घ"),
    ("kh", "ख्", "ख"),
    ("ck", "क्", "क"),
    ("ng", "ंग्", "ंग"),
    ("qu", "क्व्", "क्व"),
    ("wh", "व्", "व"),
)

#: Vowel graphemes -> (independent form, matra). Longest first.
_VOWELS: tuple[tuple[str, str, str], ...] = (
    ("eau", "ओ", "ो"),
    # "igh" is silent-gh: "flight" is फ़्लाइट, not फ़्लिघ्ट. Must precede the
    # bare vowels so the 'i' is not consumed first.
    ("igh", "आइ", "ाइ"),
    ("ee", "ई", "ी"),
    ("ea", "ई", "ी"),
    ("oo", "ऊ", "ू"),
    ("ou", "आउ", "ाउ"),
    ("ow", "ओ", "ो"),
    ("oa", "ओ", "ो"),
    ("ai", "ए", "े"),
    ("ay", "ए", "े"),
    ("ei", "ए", "े"),
    ("ey", "ए", "े"),
    ("ie", "ई", "ी"),
    ("oi", "ऑय", "ॉय"),
    ("oy", "ऑय", "ॉय"),
    ("au", "ऑ", "ॉ"),
    ("aw", "ऑ", "ॉ"),
    ("a", "अ", "ा"),
    ("e", "ए", "े"),
    ("i", "इ", "ि"),
    ("o", "ओ", "ो"),
    # Short English 'u' is /ʌ/, which Devanagari writes as the consonant's
    # inherent 'a' -- so the matra is empty. That is what makes "bus" -> बस and
    # "sun" -> सन rather than बास/सान.
    ("u", "अ", ""),
    ("y", "इ", "ी"),
)

#: Single consonants -> (with halant, bare). Bare form carries inherent 'a'.
_CONSONANTS: dict[str, tuple[str, str]] = {
    "b": ("ब्", "ब"), "c": ("क्", "क"), "d": ("ड्", "ड"), "f": ("फ़्", "फ़"),
    "g": ("ग्", "ग"), "h": ("ह्", "ह"), "j": ("ज्", "ज"), "k": ("क्", "क"),
    "l": ("ल्", "ल"), "m": ("म्", "म"), "n": ("न्", "न"), "p": ("प्", "प"),
    "q": ("क्", "क"), "r": ("र्", "र"), "s": ("स्", "स"), "t": ("ट्", "ट"),
    "v": ("व्", "व"), "w": ("व्", "व"), "x": ("क्स्", "क्स"), "y": ("य्", "य"),
    "z": ("ज़्", "ज़"),
}


def _transliterate_word(word: str) -> str:
    """Approximate an unlisted Latin word in Devanagari.

    Walks the spelling left to right, emitting a consonant with a halant unless
    a vowel follows, in which case the vowel becomes a matra. Word-initial
    vowels take their independent form. A trailing silent 'e' is dropped, since
    English almost always leaves it unspoken ("code", "file", "please").
    """
    w = word.lower()
    if len(w) > 2 and w.endswith("e") and w[-2] not in "aeiou":
        w = w[:-1]  # silent final e

    out: list[str] = []
    i = 0
    at_start = True

    while i < len(w):
        consonant = _match_consonant(w, i)
        if consonant is not None:
            bare, with_halant, length = consonant
            i += length
            vowel = _match_vowel(w, i)
            if vowel is None:
                if i >= len(w):
                    # End of word: the bare form reads more naturally than a
                    # trailing halant.
                    out.append(bare)
                elif bare in ("न", "म"):
                    # A nasal before another consonant is written as anusvara in
                    # Hindi, which is why "random" is रैंडम and not रन्डोम.
                    out.append("ं")
                else:
                    # Keep the halant so the consonant is not given a spurious
                    # inherent 'a'.
                    out.append(with_halant)
            else:
                (_, matra), consumed = vowel
                # A bare Devanagari consonant already carries inherent 'a', so
                # a following short 'a' needs no matra.
                out.append(bare)
                if not (consumed == 1 and w[i] == "a"):
                    out.append(matra)
                i += consumed
            at_start = False
            continue

        vowel = _match_vowel(w, i)
        if vowel is not None:
            (independent, matra), consumed = vowel
            out.append(independent if at_start else matra)
            i += consumed
            at_start = False
            continue

        i += 1  # punctuation or digit: left for the other passes

    return "".join(out)


def _match_consonant(w: str, i: int) -> tuple[str, str, int] | None:
    """Return (bare, with_halant, consumed) if a consonant starts at i."""
    for spelling, with_halant, bare in _DIGRAPHS:
        if w.startswith(spelling, i):
            return bare, with_halant, len(spelling)
    char = w[i]
    # 'y' is a consonant at the start of a word ("yaar") but a vowel elsewhere
    # ("city", "syntax"), so hand it to the vowel matcher in that position.
    if char == "y" and i > 0:
        return None
    # Soft c: before e/i/y English says /s/, so "city" is सिटी, not किटि.
    if char == "c" and i + 1 < len(w) and w[i + 1] in "eiy":
        return "स", "स्", 1
    if char in _CONSONANTS:
        with_halant, bare = _CONSONANTS[char]
        return bare, with_halant, 1
    return None


def _match_vowel(w: str, i: int) -> tuple[tuple[str, str], int] | None:
    """Return ((independent, matra), consumed) if a vowel starts at i."""
    for spelling, independent, matra in _VOWELS:
        if w.startswith(spelling, i):
            return (independent, matra), len(spelling)
    return None


# --- 3. entry point -------------------------------------------------------

#: A run of Latin letters, optionally with internal apostrophes.
_LATIN_WORD = re.compile(r"[A-Za-z][A-Za-z']*")


def transliterate(text: str) -> str:
    """Rewrite Latin-script words in `text` as Devanagari.

    Devanagari, digits, and punctuation pass through untouched.
    """
    if not text:
        return text
    return _LATIN_WORD.sub(lambda m: _convert(m.group(0)), text)


def _convert(word: str) -> str:
    lower = word.lower().strip("'")
    if not lower:
        return word

    if lower in LOANWORDS:
        return LOANWORDS[lower]
    if lower in ROMANIZED_HINDI:
        return ROMANIZED_HINDI[lower]

    # An all-caps run of 2-5 letters is an acronym: say the letter names,
    # unless it is one of the ones people pronounce as a word.
    if word.isupper() and 2 <= len(word) <= 5 and lower not in SPOKEN_AS_WORD:
        return " ".join(LETTER_NAMES.get(c, c) for c in lower)

    # A single letter is always a letter name, never a syllable.
    if len(lower) == 1:
        return LETTER_NAMES.get(lower, word)

    return _transliterate_word(lower)
