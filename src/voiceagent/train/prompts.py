"""Reading prompts for building a fine-tuning dataset.

`eval/sentences.py` has 22 sentences and exists to *test* synthesis. These exist to
*train* it, which is a different job with different requirements:

  * **Phonetic coverage.** A model can only learn sounds it hears. Hindi's retroflex
    series, the aspirates, the nuqta consonants and the common clusters each need to
    appear enough times to be learned, and they do not turn up reliably in whatever
    someone happens to say. Each group below targets one of them.
  * **Prosodic range.** Thirty minutes of flat declaratives teaches flat
    declaratives. Questions, exclamations, lists and asides carry different
    intonation contours, so they are here deliberately.
  * **Length spread.** Clips are batched by frames, so a dataset of uniformly long
    sentences batches badly. These run from three words to about twenty.

WHEN THE SPEAKER READS ONE OF THESE, THE PROMPT IS THE TRANSCRIPT. That is the whole
reason to read rather than improvise: it removes transcription error from the
dataset entirely. Whisper reads Hindi at about 4.8% CER, and on the first eight real
clips it turned कृपया into क्रिपय and स्पष्ट into इस पर्ष्ट -- six of eight wrong. Text we
already know should never be guessed.

AUTHORSHIP CAVEAT, stated plainly: these sentences were written by the assistant,
not by a native speaker, and this project already records that Hindi *quality*
beyond intelligibility is unverified. Skim them before recording. A sentence that
reads oddly to you will sound odd in the model, and the cost of dropping one is
zero.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Prompt:
    slug: str
    text: str
    register: str
    focus: str
    """What this sentence is here to cover -- a phoneme group, or a contour."""


#: Retroflex stops. Hindi contrasts these with the dentals, and a model that has
#: not heard enough of them collapses the distinction, which is the single most
#: recognisable way synthetic Hindi sounds foreign.
RETROFLEX: tuple[Prompt, ...] = (
    Prompt("r1", "टोकरी में दो बड़े आम रखे हैं।", "neutral", "ट ड ब़"),
    Prompt("r2", "पानी की बूँदें छत से टपक रही थीं।", "neutral", "ट ँ"),
    Prompt("r3", "उसने ठंडा पानी माँगा और चुपचाप बैठ गया।", "neutral", "ठ ठं ठ"),
    Prompt("r4", "डाकिया रोज़ सुबह चिट्ठी लेकर आता है।", "neutral", "ड ट्ठ"),
    Prompt("r5", "लड़के ने गेंद उठाई और दीवार की तरफ़ फेंक दी।", "neutral", "ड़ ठ"),
    Prompt("r6", "बड़ी मुश्किल से वह पहाड़ी रास्ता पार हुआ।", "neutral", "ड़ ड़ी"),
    Prompt("r7", "कड़ाही में तेल गरम हो गया है।", "neutral", "ड़ ह"),
    Prompt("r8", "मैंने अपना सामान अलमारी में रख दिया।", "neutral", "ण-ish, ल"),
)

#: Aspirated consonants. The unaspirated/aspirated pair is phonemic in Hindi and
#: routinely flattened by models trained mostly on English.
ASPIRATE: tuple[Prompt, ...] = (
    Prompt("a1", "खाना खाकर वह खिड़की के पास खड़ा हो गया।", "neutral", "ख ख ख"),
    Prompt("a2", "घर के पीछे घना जंगल था।", "neutral", "घ घ"),
    Prompt("a3", "छोटी बच्ची छत पर छाता लेकर चली गई।", "neutral", "छ छ छ"),
    Prompt("a4", "झूठ बोलना ठीक नहीं है।", "neutral", "झ ठ"),
    Prompt("a5", "थाली में थोड़ा और चावल डाल दो।", "neutral", "थ थ"),
    Prompt("a6", "धीरे धीरे धुआँ पूरे कमरे में भर गया।", "neutral", "ध ध ँ"),
    Prompt("a7", "फूल की खुशबू पूरे बगीचे में फैल गई।", "neutral", "फ फ"),
    Prompt("a8", "भाई ने भारी बक्सा अकेले उठा लिया।", "neutral", "भ भ"),
)

#: Nuqta consonants. Borrowed sounds, frequently written without the dot and
#: therefore under-represented in scraped text. Read them as written.
NUQTA: tuple[Prompt, ...] = (
    Prompt("n1", "काग़ज़ पर साफ़ अक्षरों में लिखो।", "neutral", "ग़ ज़ फ़"),
    Prompt("n2", "ज़रूरत पड़े तो ज़रूर बताना।", "colloquial", "ज़ ज़"),
    Prompt("n3", "थोड़ी देर में बारिश शुरू हो जाएगी।", "neutral", "ड़ श"),
    Prompt("n4", "उसका तज़ुर्बा इस काम में बहुत है।", "neutral", "ज़"),
    Prompt("n5", "फ़ोन की घंटी लगातार बज रही थी।", "neutral", "फ़ घ"),
    Prompt("n6", "क़िताब मेज़ पर रखी हुई है।", "neutral", "क़ ज़"),
)

#: Consonant clusters. These are where synthesis most often inserts a vowel that
#: should not be there.
CLUSTER: tuple[Prompt, ...] = (
    Prompt("c1", "क्षेत्र के सभी विद्यालय कल बंद रहेंगे।", "formal", "क्ष द्य"),
    Prompt("c2", "विज्ञान और गणित दोनों ज़रूरी विषय हैं।", "formal", "ज्ञ ण"),
    Prompt("c3", "श्रीमान जी ने प्रस्ताव स्वीकार कर लिया।", "formal", "श्र स्त्र स्व"),
    Prompt("c4", "स्थिति स्पष्ट होने पर ही निर्णय लिया जाएगा।", "formal", "स्थ स्प"),
    Prompt("c5", "द्वार पर खड़े व्यक्ति ने प्रश्न पूछा।", "formal", "द्व व्य प्र"),
    Prompt("c6", "त्रिपुरा और महाराष्ट्र दोनों राज्यों में बारिश हुई।", "formal", "त्र ष्ट्र ज्य"),
)

#: Vowel length. Short and long vowels are contrastive; getting them wrong is the
#: failure the round-trip eval catches most often.
VOWEL: tuple[Prompt, ...] = (
    Prompt("v1", "दिन भर बैठे रहने से पीठ में दर्द हो गया।", "neutral", "इ ई"),
    Prompt("v2", "सुबह की धूप कमरे में आ रही थी।", "neutral", "उ ऊ"),
    Prompt("v3", "नानी अपनी कहानी सुनाने लगीं।", "neutral", "आ ई"),
    Prompt("v4", "मीठा खाने का मन कर रहा है।", "colloquial", "ई आ"),
    Prompt("v5", "पूरा दिन बीत गया और काम अधूरा रह गया।", "neutral", "ऊ ई ू"),
)

#: Questions. A rising contour the model will not learn from declaratives.
QUESTION: tuple[Prompt, ...] = (
    Prompt("q1", "तुम कल कहाँ गए थे?", "colloquial", "rising"),
    Prompt("q2", "क्या आपको यह किताब चाहिए?", "neutral", "yes/no rise"),
    Prompt("q3", "यह कैसे हुआ, किसने बताया तुम्हें?", "colloquial", "double question"),
    Prompt("q4", "आप कितने बजे पहुँचेंगे?", "neutral", "wh-question"),
    Prompt("q5", "अच्छा, तो अब क्या करना चाहिए?", "colloquial", "discourse marker + rise"),
    Prompt("q6", "क्यों, कोई दिक्कत है?", "colloquial", "short rise"),
)

#: Exclamation and emphasis. Where a flat model is most obviously flat.
EMPHATIC: tuple[Prompt, ...] = (
    Prompt("e1", "अरे वाह, यह तो कमाल हो गया!", "colloquial", "exclamation"),
    Prompt("e2", "नहीं, ऐसा बिल्कुल नहीं करना है!", "colloquial", "emphatic negation"),
    Prompt("e3", "बहुत खूब, तुमने कमाल कर दिया।", "colloquial", "praise"),
    Prompt("e4", "अरे यार, मैं भूल ही गया था।", "colloquial", "self-correction"),
    Prompt("e5", "सुनो, ज़रा इधर आओ।", "colloquial", "imperative call"),
    Prompt("e6", "ओह, यह तो बहुत बुरा हुआ।", "colloquial", "sympathy"),
)

#: Lists and enumerations. Internal boundaries and a characteristic contour.
LIST: tuple[Prompt, ...] = (
    Prompt("l1", "बाज़ार से आलू, प्याज़, टमाटर और धनिया ले आना।", "colloquial", "comma list"),
    Prompt("l2", "पहले नहाओ, फिर नाश्ता करो, उसके बाद निकलो।", "colloquial", "sequence"),
    Prompt("l3", "सोमवार, मंगलवार और बुधवार को छुट्टी रहेगी।", "neutral", "day list"),
)

#: Numbers, dates and times. Read them as words, the way you would say them.
NUMERIC: tuple[Prompt, ...] = (
    Prompt("d1", "मीटिंग सुबह साढ़े दस बजे शुरू होगी।", "neutral", "time"),
    Prompt("d2", "इस महीने की पंद्रह तारीख़ को छुट्टी है।", "neutral", "date"),
    Prompt("d3", "कुल मिलाकर बत्तीस लोग आए थे।", "neutral", "number"),
    Prompt("d4", "किराया हर महीने आठ हज़ार रुपये है।", "neutral", "money"),
    Prompt("d5", "दो हज़ार पच्चीस में यह काम पूरा होगा।", "neutral", "year"),
)

#: Code-mixed. How people actually speak, and what the transliteration layer in
#: text/translit_en.py exists to handle.
CODE_MIXED: tuple[Prompt, ...] = (
    Prompt("m1", "मैंने कल शाम को meeting cancel कर दी थी।", "colloquial", "loanwords"),
    Prompt("m2", "मुझे एक ईमेल भेज देना, मैं calendar में डाल लूँगा।", "colloquial", "mixed clause"),
    Prompt("m3", "यह project अगले month में खत्म हो जाएगा।", "colloquial", "mixed nouns"),
    Prompt("m4", "ज़रा screenshot भेजो, मैं देख लेता हूँ।", "colloquial", "imperative + loan"),
    Prompt("m5", "उसका interview ठीक गया था।", "colloquial", "single loan"),
)

#: Long-form. Longer sentences with internal clauses, for the rhythm of connected
#: speech rather than isolated utterances. Kept under the clip cap.
CONNECTED: tuple[Prompt, ...] = (
    Prompt("f1", "जब मैं छोटा था, तब हम हर साल गर्मियों में नानी के घर जाते थे।",
           "neutral", "subordinate clause"),
    Prompt("f2", "अगर कल मौसम ठीक रहा, तो हम सुबह जल्दी निकल जाएँगे।", "neutral", "conditional"),
    Prompt("f3", "उसने कहा कि वह थोड़ी देर में आ जाएगा, लेकिन अब तक नहीं आया।",
           "neutral", "reported speech"),
    Prompt("f4", "इस काम में समय लगेगा, इसलिए जल्दबाज़ी करने का कोई फ़ायदा नहीं है।",
           "neutral", "causal"),
    Prompt("f5", "मुझे लगता है कि हमें पहले पूरी बात समझ लेनी चाहिए, फिर कोई फ़ैसला करना चाहिए।",
           "neutral", "opinion + sequence"),
    Prompt("f6", "पिछले हफ़्ते जो किताब तुमने दी थी, वह मैंने पूरी पढ़ ली।",
           "colloquial", "relative clause"),
)

#: Short utterances. Three to five words. These batch well and cover onsets and
#: final lengthening, which longer sentences bury in the middle.
SHORT: tuple[Prompt, ...] = (
    Prompt("s1", "ठीक है, समझ गया।", "colloquial", "short"),
    Prompt("s2", "अभी आता हूँ।", "colloquial", "short"),
    Prompt("s3", "बहुत धन्यवाद।", "neutral", "short"),
    Prompt("s4", "कोई बात नहीं।", "colloquial", "short"),
    Prompt("s5", "एक मिनट रुको।", "colloquial", "short"),
    Prompt("s6", "हाँ, बिल्कुल सही।", "colloquial", "short"),
    Prompt("s7", "मुझे नहीं पता।", "colloquial", "short"),
    Prompt("s8", "चलो, निकलते हैं।", "colloquial", "short"),
)

ALL: tuple[Prompt, ...] = (
    RETROFLEX + ASPIRATE + NUQTA + CLUSTER + VOWEL + QUESTION
    + EMPHATIC + LIST + NUMERIC + CODE_MIXED + CONNECTED + SHORT
)


def coverage() -> dict[str, int]:
    """Prompts per register, so the UI can say what is being covered."""
    out: dict[str, int] = {}
    for prompt in ALL:
        out[prompt.register] = out.get(prompt.register, 0) + 1
    return out
