"""Does Qwen3 write Hindi someone would actually say, or translated Hindi?

    uv run python -m voiceagent.eval.hindi_llm
    uv run python -m voiceagent.eval.hindi_llm --variant fewshot --show

Three things can go wrong when a Hindi turn reaches an English-tuned model, and
they need separating because the fixes differ:

  1. It answers in the wrong script -- English, or romanized Hindi. Fatal, since
     the Indic TTS reads Devanagari and Kokoro cannot speak Hindi at all.
  2. It answers in Devanagari but in *translated* register: Sanskritized formal
     vocabulary that is correct, stiff, and not how anyone speaks. This is the
     failure the phase was actually about.
  3. It emits text the synthesizer mangles -- markdown, emoji, digits.

Register is scored with a lexicon of formal/colloquial synonym pairs (सहायता vs
मदद, आवश्यकता vs ज़रूरत). Stating the obvious limit: this is a proxy. It measures
word choice, not fluency or idiom, and a native speaker's ear remains the real
test. It is here because it is reproducible and it moves in the right direction
when the prompt improves, which is what makes prompt changes decidable.

Compares prompting strategies so the cheapest sufficient one can be chosen --
fine-tuning is off the table on this machine anyway.
"""

from __future__ import annotations

import argparse
import asyncio
import statistics
from dataclasses import dataclass, field

from rich.console import Console
from rich.table import Table

from voiceagent.llm.agent import SYSTEM_PROMPT
from voiceagent.llm.base import Message

console = Console()

# --- what we ask ----------------------------------------------------------

#: Colloquial Hindi turns, the way someone would actually speak to an assistant.
TURNS: tuple[str, ...] = (
    "अरे यार, आज मौसम कैसा है?",
    "मुझे कल सुबह जल्दी उठना है, कोई सलाह दो।",
    "थोड़ा बताओ, ये काम कैसे करूँ?",
    "मेरा मन नहीं कर रहा कुछ करने का, क्या करूँ?",
    "तुम क्या कर सकते हो मेरे लिए?",
)

# --- prompting strategies -------------------------------------------------

LANGUAGE_RULE = (
    " Reply in the same language the user wrote in. When they write Hindi, reply "
    "in Devanagari script -- never romanized Hindi and never English."
)

COLLOQUIAL_RULE = (
    " Write Hindi the way people actually speak it, not textbook or news Hindi. "
    "Prefer everyday words over Sanskritized ones: say मदद not सहायता, ज़रूरत not "
    "आवश्यकता, कोशिश not प्रयास, सवाल not प्रश्न, काम not कार्य, लेकिन not परंतु."
)

#: Few-shot examples carry register in a way instructions cannot: they show the
#: rhythm and length of a spoken reply, not just the vocabulary.
FEWSHOT: tuple[tuple[str, str], ...] = (
    ("नमस्ते, तुम कौन हो?",
     "नमस्ते! मैं आपका असिस्टेंट हूँ। बताइए, क्या मदद चाहिए?"),
    ("यार ये फ़ाइल कहाँ रखी है?",
     "एक मिनट, देख लेता हूँ। आपके फ़ोल्डर में ही होनी चाहिए।"),
)

VARIANTS: dict[str, str] = {
    "baseline": SYSTEM_PROMPT,
    "language": SYSTEM_PROMPT + LANGUAGE_RULE,
    "colloquial": SYSTEM_PROMPT + LANGUAGE_RULE + COLLOQUIAL_RULE,
    "fewshot": SYSTEM_PROMPT + LANGUAGE_RULE + COLLOQUIAL_RULE,
}

# --- register lexicon -----------------------------------------------------

#: (formal/Sanskritized, everyday) pairs. Both members are correct Hindi; the
#: left one is what a translation engine reaches for and a person rarely says.
REGISTER_PAIRS: tuple[tuple[str, str], ...] = (
    ("सहायता", "मदद"),
    ("आवश्यकता", "ज़रूरत"),
    ("प्रयास", "कोशिश"),
    ("प्रश्न", "सवाल"),
    ("उत्तर", "जवाब"),
    ("कार्य", "काम"),
    ("परंतु", "लेकिन"),
    ("किंतु", "लेकिन"),
    ("तथा", "और"),
    ("यदि", "अगर"),
    ("संभव", "मुमकिन"),
    ("शीघ्र", "जल्दी"),
    ("प्रतीक्षा", "इंतज़ार"),
    ("स्मरण", "याद"),
    ("उपलब्ध", "मौजूद"),
    ("प्रारंभ", "शुरू"),
    ("समाप्त", "खत्म"),
    ("अत्यंत", "बहुत"),
    ("संपूर्ण", "पूरा"),
    ("वार्तालाप", "बातचीत"),
    ("धन्यवाद", "शुक्रिया"),
    ("इच्छा", "मन"),
)

FORMAL_WORDS = {formal for formal, _ in REGISTER_PAIRS}
COLLOQUIAL_WORDS = {casual for _, casual in REGISTER_PAIRS}

#: Characters the synthesizer reads literally or drops. Digits are excluded --
#: normalize_hi turns those into words before synthesis.
TTS_HOSTILE = set("*#`_[]{}|<>~^\\") | {"•", "→", "—"}


def devanagari_share(text: str) -> float:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    return sum(1 for c in letters if "ऀ" <= c <= "ॿ") / len(letters)


def latin_words(text: str) -> int:
    import re

    return len(re.findall(r"[A-Za-z]{2,}", text))


def max_ngram_repeat(text: str, n: int = 3) -> int:
    """Longest run of the same n-gram repeated back to back.

    This is the metric that caught the real defect. The register lexicon happily
    scored a reply 100% "everyday" while its actual content was
    "ऊपर से ऊपर ऊपर से ऊपर ..." repeated until the token budget ran out. Word
    choice was never the problem; coherence was.
    """
    words = text.split()
    if len(words) < 2 * n:
        return 1

    grams = [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]
    best = run = 1
    for i in range(1, len(grams)):
        # Step by n so runs are counted as whole non-overlapping repeats.
        if grams[i] == grams[i - 1]:
            run += 1
            best = max(best, run)
        else:
            run = 1
    return best


def distinct_word_ratio(text: str) -> float:
    """Unique words over total. A degenerate loop drives this toward zero."""
    words = text.split()
    if not words:
        return 1.0
    return len(set(words)) / len(words)


def register_score(text: str) -> tuple[float | None, int, int]:
    """Share of register-marked words that are the everyday form.

    None when the reply used none of the marked words, which is common for short
    replies and must not be read as a zero.
    """
    formal = sum(text.count(w) for w in FORMAL_WORDS)
    casual = sum(text.count(w) for w in COLLOQUIAL_WORDS)
    total = formal + casual
    return (casual / total if total else None), formal, casual


@dataclass
class VariantResult:
    name: str
    replies: list[str] = field(default_factory=list)
    deva: list[float] = field(default_factory=list)
    registers: list[float] = field(default_factory=list)
    formal_hits: int = 0
    casual_hits: int = 0
    latin: int = 0
    hostile: int = 0
    sentence_words: list[float] = field(default_factory=list)
    distinct: list[float] = field(default_factory=list)
    #: Replies whose content degenerated into a repeated phrase.
    looped: int = 0
    loop_examples: list[str] = field(default_factory=list)

    @property
    def median_deva(self) -> float:
        return statistics.median(self.deva) if self.deva else 0.0

    @property
    def mean_register(self) -> float | None:
        return statistics.mean(self.registers) if self.registers else None


async def run_variant(engine, name: str, prompt: str, show: bool, repeat: int) -> VariantResult:
    result = VariantResult(name)

    # Each turn is sampled `repeat` times because the defect this eval exists to
    # catch is intermittent: one run showed a reply degenerating into
    # "ऊपर से ऊपर ..." until the token budget ran out, and the very next run of
    # the same prompt showed none at all. A single pass measures luck.
    for turn in [t for t in TURNS for _ in range(repeat)]:
        messages = [Message(role="system", content=prompt)]
        if name == "fewshot":
            for user, assistant in FEWSHOT:
                messages.append(Message(role="user", content=user))
                messages.append(Message(role="assistant", content=assistant))
        messages.append(Message(role="user", content=turn))

        text = ""
        async for chunk in engine.stream(messages, max_tokens=160):
            text += chunk.text
        reply = text.strip()
        result.replies.append(reply)

        result.deva.append(devanagari_share(reply))
        score, formal, casual = register_score(reply)
        if score is not None:
            result.registers.append(score)
        result.formal_hits += formal
        result.casual_hits += casual
        result.latin += latin_words(reply)
        result.hostile += sum(1 for c in reply if c in TTS_HOSTILE)

        result.distinct.append(distinct_word_ratio(reply))
        # Two back-to-back repeats of a trigram is still plausible prose; three
        # is not something a person writes.
        if max_ngram_repeat(reply) >= 3 or distinct_word_ratio(reply) < 0.5:
            result.looped += 1
            result.loop_examples.append(reply)

        sentences = [s for s in reply.replace("।", ".").split(".") if s.strip()]
        if sentences:
            result.sentence_words.append(
                statistics.mean(len(s.split()) for s in sentences)
            )

        if show:
            console.print(f"[dim]{turn}[/]\n  [bold]{reply}[/]\n")

    return result


async def main_async(variant: str | None, show: bool, repeat: int) -> int:
    from voiceagent.llm.mlx_engine import MLXLLMEngine

    chosen = VARIANTS if variant is None else {variant: VARIANTS[variant]}

    console.print("loading Qwen3-4B ...")
    engine = MLXLLMEngine()
    engine.load()
    console.print(f"loaded ({engine.resident_bytes / 2**30:.2f} GiB)\n")

    results: list[VariantResult] = []
    for name, prompt in chosen.items():
        console.print(f"running [bold]{name}[/] ...")
        # The prefix cache is keyed on the prompt; reset so one variant's cache
        # cannot serve another's and skew what we attribute to the prompt.
        engine.reset_cache()
        results.append(await run_variant(engine, name, prompt, show, repeat))

    table = Table(title="Qwen3 Hindi register", title_justify="left", header_style="bold")
    table.add_column("Prompt")
    table.add_column("Devanagari", justify="right")
    table.add_column("Everyday words", justify="right")
    table.add_column("formal:everyday", justify="center")
    table.add_column("Latin words", justify="right")
    table.add_column("TTS-hostile", justify="right")
    table.add_column("Words/sentence", justify="right")
    table.add_column("Distinct", justify="right")
    table.add_column("Looped", justify="right")

    for r in results:
        register = r.mean_register
        deva = r.median_deva
        table.add_row(
            r.name,
            f"{deva:.0%}" if deva > 0.9 else f"[red]{deva:.0%}[/]",
            "[dim]n/a[/]" if register is None
            else (f"{register:.0%}" if register >= 0.6 else f"[red]{register:.0%}[/]"),
            f"{r.formal_hits}:{r.casual_hits}",
            str(r.latin) if r.latin == 0 else f"[yellow]{r.latin}[/]",
            str(r.hostile) if r.hostile == 0 else f"[red]{r.hostile}[/]",
            f"{statistics.mean(r.sentence_words):.1f}" if r.sentence_words else "-",
            f"{statistics.mean(r.distinct):.2f}" if r.distinct else "-",
            f"{r.looped}/{len(r.replies)}" if r.looped == 0
            else f"[red]{r.looped}/{len(r.replies)}[/]",
        )

    console.print(table)
    console.print(
        "\n[dim]Devanagari: share of letters in Devanagari -- below 100% means it "
        "leaked English or romanized Hindi.\n"
        "Everyday words: of the register-marked words it used, how many were the "
        "spoken form rather than the Sanskritized one.\n"
        "Looped: replies that degenerated into a repeated phrase. This is "
        "intermittent, so it needs --repeat to measure as a rate.[/]"
    )

    for r in results:
        for example in r.loop_examples[:1]:
            console.print(f"\n[red]degenerate reply under {r.name}:[/]\n  {example[:300]}")

    engine.unload()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=sorted(VARIANTS), help="only this prompt")
    parser.add_argument("--show", action="store_true", help="print every reply")
    parser.add_argument("--repeat", type=int, default=1,
                        help="samples per turn; >1 to measure intermittent degeneration")
    args = parser.parse_args()
    return asyncio.run(main_async(args.variant, args.show, args.repeat))


if __name__ == "__main__":
    raise SystemExit(main())
