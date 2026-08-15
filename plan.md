# Plan — ElevenLabs for India

Companion to `whatwehave.md` (current state). Evidence gathered 2026-08-14 with
web access; **strategy rewritten 2026-08-15** on the founder's call.

Read §3 first. The product is a paid Indian-language voice platform — website,
API, credits in rupees — with the fully offline desktop app as its premium tier
rather than as the whole company.

**Sections 0, 1, 2 and 10 are evidence and are unchanged.** They are what a
week of verification found, and they remain true under the new strategy: Indian
TTS is commoditising, ElevenLabs is genuinely strong at Hindi, Sarvam sells at
₹2.70 a finished minute. The platform has to be built with those facts true, not
by forgetting them. §13 lists what else did not change and why.

Two things in §3 are new and neither is optional: **we speak one Indian
language and the plan needs several** (§3.1), and **hosted inference retires
the zero-recurring-cost constraint the codebase was built on** (§3.2).

---

## 0. Confidence and sourcing

I checked things directly where I could. Confidence is not uniform:

- **High** — read from the primary source: Hugging Face model cards, arXiv papers,
  Sarvam's own API docs, ElevenLabs' own India page, and the code in this repo.
- **Medium** — vendor pages and reputable secondary reporting.
- **Low** — SEO listicles. Used only for corroboration, never as the sole basis
  for a claim. Where a number is low-confidence I say so inline.

Nothing below carries a `⚠️ VERIFY`. Section 10 is the replacement table: every
guessed number from the draft, with what it actually is.

---

## 1. Four findings that invalidate the old plan

### 1.1 IndicF5 ranks last among current Hindi TTS systems — measured at scale

AI4Bharat published **SpeechArenaBench**: over **120,000 pairwise comparisons from
over 1,900 native raters** across 10 Indian languages, 5,357 sentences, analysed
with Bradley-Terry. Seven systems compared.

Result, Bradley-Terry score:

| System | Score |
| --- | --- |
| Gemini 2.5 Pro TTS | 1128.53 ± 3 (1st in 9 of 10 languages) |
| ElevenLabs v3 | statistically tied for 2nd |
| Sonic 3 (Cartesia) | statistically tied for 2nd |
| Bulbul v3 Beta, Speech 2.8 HD, GPT-4o-mini TTS | middle |
| **IndicF5** | **805.75 ± 3 — last** |

Source: [Preferences of a Voice-First Nation, arXiv 2604.21481](https://arxiv.org/html/2604.21481);
data at [ai4bharat/SpeechArenaBench](https://huggingface.co/datasets/ai4bharat/SpeechArenaBench/).

This matters more than any other single fact in this document. The current Hindi
path fine-tunes the **worst-ranked** system in the largest independent Indic TTS
evaluation that exists — and the ranking was produced by the same lab that built
the model. The benchmark even includes **4,164 code-mixed sentences** (intra-sentential
English insertion, transliteration mixing, mixed script), and rankings were
**stable across code-mixed input**. So the intended wedge — "we win on Hinglish" —
was specifically tested and does not open up.

### 1.2 "ElevenLabs is bad at Hindi" is no longer true

Independent MOS pilot (50 samples/language) inside
[VoiceAgentBench, arXiv 2510.07978](https://arxiv.org/pdf/2510.07978), Hindi:

| System | Naturalness | Prosody | Pronunciation | Clarity |
| --- | --- | --- | --- | --- |
| ElevenLabs | **3.84** | 3.56 | 3.70 | 3.96 |
| Sarvam | 3.76 | 3.06 | 3.64 | 3.86 |
| Krutrim | 3.20 | **3.60** | **3.72** | **3.96** |
| Google | 2.24 | 2.34 | 3.08 | 3.46 |

Google is the only one still bad. ElevenLabs leads Hindi naturalness. And
commercially they are not dabbling — [elevenlabs.io/india](https://elevenlabs.io/india)
advertises 12 Indian languages, voices "tuned for Indian accents and code-switched
speech," **India data residency**, SOC 2 / ISO 27001 / PCI DSS, telephony
integrations with Ozonetel, Exotel and Plivo, and named Indian customers:
**Meesho, Kuku FM, PocketFM, Cars24, hoichoi, TVS Motor**.

Kuku FM and PocketFM are exactly the audiobook/long-form buyers the draft listed
as wedge #3. They are already signed.

### 1.3 The price floor collapsed — pre-rendered audio is not a business

[Sarvam's published pricing](https://docs.sarvam.ai/api-reference-docs/pricing):

- Bulbul v2 — **₹15 per 10,000 characters**
- Bulbul v3 — **₹30 per 10,000 characters** (beta)
- STT — ₹30/hour; with diarization ₹45/hour

At ~900 characters per minute of Hindi speech, Bulbul v3 is about **₹2.70 per
finished minute**. Bulbul v4 was shown at Epoch on 30 July 2026.

[Bhashini](https://www.bhashini.ai/pricing), the government stack, subscribes
speech synthesis at roughly **₹250/month for 50,000 characters/day**.

The draft proposed selling rendered audio at ₹100–300/minute against a human-VO
anchor of ₹500–3000/min. The human anchor is roughly right —
[e-learning Hindi VO runs about ₹1,000/minute](https://www.cosmicsounds.in/e-learning-voice-over-rates-in-india-a-comprehensive-guide/),
~₹60,000 for an hour of finished audio. But that anchor is irrelevant now,
because your buyer's realistic alternative is no longer a human. It is a
₹2.70/minute API from a funded Indian company, or a ₹250/month government
subscription. Any ed-tech buyer can call those directly.

Selling rendered audio means reselling a commodity whose supplier is better
funded than you, with a marginal cost you cannot beat, using a slower pipeline
(RTF 3.40). **That business is closed.** Wedges 1–4 in the draft all depend on it.

### 1.4 The dubbing and narration lanes are crowded with funded Indian players

- **Murf AI** — Indian, [$10M Series A](https://murf.ai/blog/series-a-announcement),
  shipped *Falcon* (Nov 2025, 55 ms model latency, 130 ms time-to-first-audio,
  35+ languages) and **MultiNative**, which switches language mid-sentence
  English↔Hindi without stitching. That is the Hinglish wedge, productised.
- **Dubverse** — Indian, video dubbing plus cloning and emotion control.
- **Cartesia Sonic 3** — tied for 2nd on SpeechArenaBench.
- **Krutrim** (Ola) — best Hindi prosody/pronunciation/clarity in the pilot above.

---

## 2. Three findings that open something better

### 2.1 The licence fork is nearly free — and the fix is already on disk

The draft budgeted 1–2 weeks and three options for Phase 0. There is a fourth
option it did not know about, and it costs about a day.

`mlx-audio` 0.4.7 — **already installed in `.venv`** — ships Resemble AI's
**Chatterbox Multilingual**, separate from the `chatterbox_turbo` you use for
cloning. Verified locally:

```
.venv/lib/python3.12/site-packages/mlx_audio/tts/models/chatterbox/chatterbox.py:43
    "hi": "Hindi",
```

Chatterbox Multilingual is **MIT**, supports 23 languages including Hindi,
does zero-shot cloning, has emotion/intensity control, and watermarks output
with PerTh. Sources:
[Resemble AI](https://www.resemble.ai/learn/models/chatterbox-multilingual),
[HF card](https://huggingface.co/ResembleAI/chatterbox).

Switching the Hindi path from IndicF5 to Chatterbox Multilingual removes
`f5-tts` and with it **`encodec` (CC-BY-NC), `Unidecode` (GPL), `frozendict`
(LGPL-3), `soxr` (LGPL-2.1)** in a single move. Every blocking licence item in
`whatwehave.md` closes at once. `voice-doctor` should go green with no new
dependency, no vocoder surgery, and no migration to Indic Parler-TTS.

Three secondary benefits:

- **Speed.** Chatterbox measures RTF ~1.6 in your own notes vs IndicF5's 3.40.
  Roughly 2× faster, and materially closer to the RTF < 1 that a bilingual live
  loop needs.
- **One engine, two languages.** Hindi and cloning collapse onto the same model.
  The `chatterbox_turbo` / `indic_engine` split can go away.
- **The 0.75 speed bug may evaporate.** It is an f5-tts artifact — duration set
  arithmetically from the enrolment clip. Different architecture, likely no bug.
  Re-measure before applying the derived correction; you may be correcting
  something that is no longer there.

Independent corroboration that Chatterbox is a sound Hindi base:
[Praxy Voice, arXiv 2604.25441](https://arxiv.org/abs/2604.25441) builds
commercial-class Indic TTS on frozen Chatterbox using BUPS (ISO-15919
romanisation) plus a LoRA, releasing
[Apache-2.0 weights](https://huggingface.co/Praxel/praxy-voice-r6) and
[MIT code](https://github.com/praxelhq/praxy). Note the model card's own advice:
the LoRA helps Telugu and Tamil but **regresses Hindi — use vanilla Chatterbox
for Hindi.** That is a free, specific, load-bearing hint. Take it.

Also on the shelf if Chatterbox disappoints:
[OpenF5-TTS-Base](https://huggingface.co/mrfakename/OpenF5-TTS-Base) — Apache-2.0,
F5 architecture, permissive training data — but English-only and its own card
admits it is inferior to the NC-licensed original. Fallback, not first choice.

### 2.2 The evaluation you have not run has largely been run for you

You have a 96-item blind harness stuck at zero listeners, and `results()` correctly
refuses a verdict under 20 ratings/system. Recruiting 20–30 raters was scored as
2–3 weeks of work.

SpeechArenaBench already spent 1,900 raters and 120K comparisons on this exact
question, and **published the sentences and the preference data**. Use it: take
the code-mixed subset as your held-out set, generate your conditions against
theirs, and you inherit a methodology plus a reference ranking. Your own harness
stops being a data-collection project and becomes a regression test — which is
what it is genuinely good at.

Related, worth reading before designing any more evaluation:
[PSP: per-dimension accent benchmark for Indic TTS](https://arxiv.org/pdf/2604.25476)
and [Phir Hera Fairy](https://arxiv.org/html/2505.20693v1) (English F5 as a strong
faker for low-resource Indian languages).

### 2.3 Regulation is manufacturing demand for exactly what this repo is

The DPDP Rules were **notified 13 November 2025**. Phased: Consent Manager
framework live **13 November 2026**; full operational compliance by
**13 May 2027**; penalties to ₹250 crore. Biometric data needs explicit,
unbundled, purpose-specific consent. Cross-border transfer needs an adequacy
framework or contractual safeguards.
([Fisher Phillips](https://www.fisherphillips.com/en/insights/insights/indias-new-data-privacy-rules-are-here),
[Scrut](https://www.scrut.io/post/dpdp-rules))

Consequence: for BFSI, insurance and healthcare, **on-premise or in-India
deployment is becoming the clean answer**, and voice is biometric-adjacent.
Most regulated Indian enterprises are answering to DPDP *plus* TRAI *plus* a
sectoral regulator simultaneously.
([Caller Digital regulatory map](https://www.caller.digital/blog/voice-ai-india-regulatory-map-2026))

And every system in §1 is a cloud API. ElevenLabs offers India *data residency* —
still their cloud, still your data leaving your perimeter. Sarvam's public
pricing page lists **no self-hosted or on-prem option at all**.

The nearest thing to a competitor is **Sarvam Edge** (announced 14 Feb 2026):
74M parameters, ~294 MB, ASR + TTS + translation for 10 Indic languages, under
300 ms on a Snapdragon 8 Gen 3, 8.5× faster than real time. It is genuinely
impressive and genuinely narrow — **speech I/O only. No LLM. No tool calling.
No memory. No agent.** As of now Sarvam has announced no public availability
date and no OEM partnerships.
([Sarvam](https://www.sarvam.ai/products/edge),
[Analytics Vidhya](https://www.analyticsvidhya.com/blog/2026/03/sarvam-edge/))

That gap — speech-only edge models on one side, cloud agents on the other — is
where this repo already sits, alone.

---

## 3. The thesis — rewritten 2026-08-15

**Superseded:** the offline-only thesis this section used to hold. It argued that
rendered audio is a commodity and the only defensible position is the private
on-prem agent. Sections 1 and 2 above are the evidence for that and they are
unchanged and still true. The founder's call is that they support a *narrower*
conclusion than was drawn from them, and the product is being built as follows.

> **ElevenLabs for India.** A paid Indian-language voice platform: a website
> where anyone signs up, clones or picks a voice, generates speech, and pays;
> an API behind it for developers; and a desktop app that runs the same engine
> **entirely offline** for the buyers who cannot use a cloud at all.
>
> Not "beat ElevenLabs at Hindi." Be the one built *for* this market —
> Indian languages, Indian pricing, UPI and rupees, Indian support hours — and
> own the one capability none of them have: it also runs with the network off.

The offline path is not abandoned; it is **demoted from the whole company to the
premium tier**. That is the honest version of this strategy, and it is stronger
than either half alone: the web product is how the business is funded and
distributed, and the offline desktop app is the thing a competitor with a
datacenter cannot copy without rebuilding on small models.

What changes concretely:

| | Offline-only plan | Now |
| --- | --- | --- |
| Product | An on-prem agent, sold by hand | A self-serve website + API, plus a desktop app |
| Revenue | ₹50k–2L pilots, hand-installed | Credits and subscriptions, self-serve, plus enterprise |
| Buyer | CISOs at regulated enterprises | Creators, developers, studios, SMBs — *and* those CISOs |
| Inference | On the user's machine, always | **Hosted**, with on-device as a tier |
| Recurring cost | Zero, by hard constraint | **A real COGS line.** See §3.2 |
| Languages | Hindi was enough | **Hindi is nowhere near enough.** See §3.1 |
| Offline agent | The entire thesis | The premium differentiator |

**What survives, unchanged and still load-bearing:** licence discipline enforced
in code; round trip is an alarm and never a ranking; publish your own ceilings;
a measurement that contradicts the speaker loses; charge early. None of that is
strategy-dependent — it is how this codebase avoids being confidently wrong.

### 3.1 The critical gap: we speak one Indian language

This is the single biggest issue with the reframe, and it is now the critical
path rather than a footnote.

"For the Indian market what ElevenLabs did for the world" is a claim about
**coverage**. ElevenLabs' own India page advertises **12 Indian languages**.
Sarvam covers 10. Bhashini covers the schedule. This repo speaks **Hindi**, and
raises `UnsupportedLanguage` on every other Indic script.

That was a deliberate, correct trade when the product was one offline agent for
Hindi-belt enterprises (§4: eleven languages traded for a clean licence tree).
It is disqualifying for a consumer platform. A visitor who cannot find Tamil,
Telugu, Bengali, Marathi or Gujarati does not become a smaller customer — they
leave.

So language coverage moves to **Phase A, first, ahead of everything**, and it is
the one item on this plan that could fail on technical grounds. Options, in the
order they should be tried:

1. **Chatterbox Multilingual already speaks 23 languages** — but only Hindi
   among Indic. Check what its tokenizer actually does with Marathi and Nepali,
   which share Devanagari; that is a free experiment and might yield two more.
2. **Indic Parler-TTS** — Apache-2.0, **21 languages**, Hindi at 84.79% native
   satisfaction (§10). Description-based control, *no cloning*. This is the most
   likely answer for breadth, and it is licence-clean. The cost is that cloned
   voices would be Hindi/English-only for a while.
3. **IndicF5** — speaks 11, MIT weights, but ranks last of seven and drags in a
   dependency tree that fails our own audit. Only as a stopgap, and only if the
   audit rule is consciously suspended, which §2.1's history says it should not
   be.
4. **Fine-tune on IndiaAI compute** — ₹65–92/GPU-hour, up to 40% subsidy. The
   real answer at scale, and the reason to apply now rather than later.

**Do not sell a language before it is measured.** Shipping bad Tamil is worse
than shipping no Tamil: §1.1 is what a public quality ranking does to a company
that got ahead of its evidence.

### 3.2 The architectural inversion: hosted inference

Constraint 3 in `README.md` and `whatwehave.md` reads *"Zero recurring cost —
local inference only."* A paid website and an API mean **someone else's request
runs on our hardware**, on our electricity bill, at our latency. That constraint
is retired for the hosted product and **kept for the desktop app**, which is
what makes the two tiers genuinely different rather than the same thing priced
twice.

The awkward part is specific: **the entire stack is MLX, which runs only on
Apple Silicon.** There is no Linux or CUDA path today. Three ways out, and the
first is better than it sounds:

1. **Serve from Mac hardware.** MLX runs fine; a Mac mini or Studio fleet is a
   real, boring, cheap way to serve early volume, and it needs *zero* porting.
   This is the launch answer.
2. **Port to PyTorch/CUDA.** Chatterbox's original implementation *is* PyTorch —
   `mlx-audio` is a port of it, and the weights are the same MIT checkpoint. So
   this is a rewrite of our engine wrapper, not of the model. This is the scale
   answer, and it unlocks IndiaAI GPUs.
3. Both: Macs for launch, GPUs when queue depth says so.

Two things follow immediately. **Cost per generation becomes a number we must
know**, because Sarvam sells at ₹2.70/finished minute and Bhashini is ~free —
that is the price ceiling and the margin has to live under it. And **the spec
sheet's job changes**: it stops being a buyer-facing credibility artifact and
becomes capacity planning.

### 3.3 What the competition actually sells, which is not just a model

From §1.2, ElevenLabs' India page: 12 Indian languages, voices tuned for Indian
accents and code-switched speech, India data residency, SOC 2 / ISO 27001 / PCI
DSS, telephony integrations with Ozonetel, Exotel and Plivo, and named customers
including Meesho, Kuku FM, PocketFM, Cars24, hoichoi and TVS Motor.

Almost none of that is the model. It is coverage, compliance paper,
integrations, and proof. A better checkpoint does not win this; a better
*product* might. What we can credibly beat them on, in order:

- **Price in rupees**, with UPI, and no dollar-denominated card wall.
- **Offline desktop**, which none of them has at all.
- **Support in the same timezone**, from the person who wrote it.
- **A licence audit** that any customer's legal team can rerun.

What we cannot beat them on today: languages, compliance certifications, and
enterprise integrations. Those are build items, not talking points.

---

## 4. Phase 0 — Close the licence fork — **DONE**

Budgeted 1–2 weeks in the draft, then 3 days here. Took one session, because the
fix was already installed. `voice-doctor` now exits **0**:

```
PHASE 0 PASS (metal=True, licenses=True, budget=True)
```

**Measured before switching** — 12 held-out sentences, same reference clip, the
project's own round-trip scorer. Full write-up in
`eval_out/chatterbox_spike/FINDINGS.md`:

| | human (ceiling) | IndicF5 | Chatterbox |
| --- | --- | --- | --- |
| Mean round-trip overlap | 90.2 % | 88.7 % | **94.0 %** |
| Code-mixed subset | 91.5 % | 86.0 % | **95.4 %** |
| Aggregate RTF | — | 3.40 | **1.24** |

Better on 9 of 12, tied on 2, worse on none. Above the human anchor means at the
metric's ceiling, not "sounds better" — intelligibility is not naturalness.

**What shipped:**

- New `tts/chatterbox_indic.py` on `mlx-community/chatterbox-multilingual-v3`
  (MIT), with the Praxy Voice sampling recipe.
- Deleted `tts/indic_engine.py`, `tts/remote_engine.py`, `web/tts_service.py`,
  `train/prepare_indic.py`, the `indic` and `remote` extras' f5-tts contents, and
  the `voice-tts-service` / `voice-train-prep` entry points. `uv sync` removed
  `f5-tts`, `encodec`, `Unidecode`, `frozendict`, `soxr`, `vocos` and `pydub`.
- Vocoder control condition reimplemented on Chatterbox's own HiFiGAN, so the
  benchmark's best idea survives the engine swap.
- 395 tests pass (was 427 across a larger surface; 29 new ones cover the engine).
- README Phase 12, `whatwehave.md` and `models.py` all updated.

**Four findings worth carrying forward:**

- **The 0.75 speed correction was never shipped, and must not be.** It was an
  f5-tts duration artifact. Chatterbox runs at 0.81× and exposes no speed knob.
- **The round-trip scorer mislabels clips under ~2 s** — a 1.7 s Hindi clip
  auto-detected as Korean and scored 0 %; pinned to `hi`, 88 %. Fix this first;
  every gate downstream depends on it.
- **`translit_en` still earns its place** even though the multilingual tokenizer
  takes raw Latin (h1 94 → 98 %).
- **MLX arrays are thread-affine and MLX is lazy**, so an off-thread bug
  surfaces far from its cause. The engine owns one thread for load and generate.

**What it cost, stated plainly:** ten Indic languages (Chatterbox speaks Hindi
only; the rest now raise `UnsupportedLanguage`), the per-voice fine-tuning path,
and 1.5 GiB of headroom (`MIN_FREE_GIB` 2.5 → 4.0).

## 5. Phase 1 — Make the local agent the product — **DONE**

All five ship-blockers below are closed. `voice-doctor` exits 0, 529 tests pass,
and the measured state is in `eval_out/SPECSHEET.md`:

| | before | now |
| --- | --- | --- |
| Licence audit | failing | **clean**, zero exceptions |
| Hindi TTS | RTF 3.40 | **RTF 0.63** (0.70 median idle) |
| Live loop | English only | **bilingual**, both directions |
| Qwen3 repetition | ~1 in 80, unbounded | detected and stopped |
| Desktop app | launcher needing a checkout | **1.3 GB self-contained `.app`** |
| Loanword table | 96 entries | **227**, +4.2 pts on code-mixed |

Two things are deliberately still open and named rather than glossed: the `.app`
is **unsigned**, so Gatekeeper refuses it on another Mac; and there is still **no
quality verdict** — intelligibility and latency are measured, naturalness is not.

The original plan for this phase follows.

### The original ship-blockers

The gap between this repo and a product is not voice quality. It is that
`whatwehave.md` describes a launcher, an English-only live loop, and a repetition
bug.

**Ship-blockers, in order:**

1. **Bilingual live loop.** Currently English-only because Hindi is RTF 3.40. At
   Chatterbox's ~1.6 this becomes an engineering problem instead of a research
   one. Getting Hindi under RTF 1.0 — streaming/chunked synthesis, warm model,
   no per-turn reload — is the single highest-value technical task in this plan.
   A Hindi agent you can *talk to* is the demo. Type-and-listen is not.
2. **Fix the Qwen3 repetition loop** (~1 in 80). Fatal in a live demo, invisible
   in every quality metric. Repetition penalty / n-gram blocking / stop
   heuristics — cheap fix, ship it.
3. **Self-contained desktop bundle.** Today the Tauri app still needs a checkout
   and a `.venv`. Nobody in procurement will do that. `num2words` (LGPL) is the
   stated blocker for opaque freezing — either replace it with a small Hindi/English
   number-to-words routine you own (a few hundred lines, and you need Indian
   numbering — lakh/crore — anyway), or comply by keeping it dynamically linked
   and documenting relink rights. Replacing it is cleaner and removes your last
   recorded licence exception.
4. **Memory ceiling honesty.** 18 GB dev machine with 2–5 GB free is your
   constraint, not the customer's. Publish a measured spec sheet: minimum RAM,
   resident footprint per language, what evicts what. Enterprise buyers ask this
   first and nobody else in Indic voice can answer it with measurements.
5. **Grow the loanword table.** Latin→Devanagari fallback is "a floor, not a
   solution" by your own note. Hand-curate the top few hundred code-mixed terms
   for your first vertical. Unglamorous, high-impact, and it is the part of
   Hinglish handling you control regardless of which TTS you use.

**Deliberately deferred:** the recording sprint, any fine-tuning, other languages.

---

## 6. Phase 2 — Evidence (2 weeks, overlapping Phase 1) — **MOSTLY DONE**

Done 2026-08-14: the spec sheet (`eval_out/SPECSHEET.md`), the licence audit as a
green build gate, and the quality position — `eval_out/arena/FINDINGS.md`.

Outstanding: the offline demo recording, which needs a camera and a person.

The quality result went further than "run our conditions against the code-mixed
subset" anticipated, in a direction worth reading before writing any marketing.
We land **inside the intelligibility band of the six working cloud systems** on
held-out arena code-mixed sentences, and separate cleanly from IndicF5 (92%
paired wins; those raters approved IndicF5 13% of the time). That is the claim.

But we also came top of our own table, above Gemini and ElevenLabs, and that is
**not** a result. Calibrating our round-trip scorer against 654 human-rated arena
clips put it at AUC 0.671, and 0.625 with IndicF5 removed, against 0.500 for a
coin flip — inside the working band its ordering inverts relative to the raters.
The instrument that produced our number cannot rank the band it places us in.
Publishing that top row would be §1.1 committed with our own counter-evidence
next to it. See [[roundtrip-cannot-rank-quality]] in the project memory.

The claim to earn is no longer "better than ElevenLabs." It is falsifiable and
much easier to defend:

> "A complete Hindi+English voice agent — speech in, reasoning with tools,
> speech out, encrypted memory — running fully offline on a consumer Mac, with
> every dependency under a permissive licence, verified by an audit that fails
> the build. Nothing leaves the device. Here are the measurements."

What to produce:

- **The offline demo.** Record it with the network physically off. Airplane mode
  in frame. This is the artifact; it argues by itself.
- **A measured spec sheet.** Latency (currently 1.8 s warm / 2.4 s cold to first
  audio), RTF per language, resident memory per configuration, STT CER
  (Hindi 4.8%), round-trip intelligibility against its real ~90% ceiling.
  Publish the ceiling. Publishing your own ceiling is the credibility move.
- **Quality position, honestly stated.** Run your conditions against the
  SpeechArenaBench code-mixed subset. If Chatterbox Hindi lands mid-pack against
  cloud leaders, **say so** — "cloud-competitive quality, entirely offline" is a
  strong and true claim. Do not claim a quality win you cannot support; §1.1 is
  what happens to people who do.
- **The licence audit as a product feature.** `voice-doctor` exiting non-zero on
  a CC-BY-NC transitive dependency is a diligence artifact almost nobody in this
  space can produce. Screenshot it. Put it in the deck.
- **Keep the blind harness** as a regression test across engine swaps. That is
  its real job and it is a good one.

---

## 7. The roadmap, rebuilt around the platform

Phases are lettered to keep them distinct from the old numbered ones, which are
done and are referenced throughout this document.

### Phase A — Languages (the critical path)

Nothing else on this plan matters if the answer is "Hindi." See §3.1.

1. Test Chatterbox Multilingual on Marathi and Nepali. Same script, free
   experiment, possibly two languages for nothing.
2. Spike **Indic Parler-TTS** (Apache-2.0, 21 languages) as the breadth engine,
   behind the same `<module>/base.py` seam every other engine sits behind. Route
   on script, exactly as the TTS router already does.
3. Decide the split honestly: **cloning in Hindi and English; catalogue voices
   in everything else.** That is a shippable product and a truthful pricing
   page. Pretending otherwise is §1.1 waiting to happen.
4. Measure each language before it goes on the pricing page. Round trip is an
   alarm only; for a *ranking* the only validated instrument in this project is
   pairwise comparison, which is what `arena_bt` fits.

**Exit test:** five Indian languages, each measured, each with a named ceiling.

### Phase B — The product surface

The website is the company now. Concretely:

- **Landing page** that states what it is in one screen, with audio you can play
  before signing up. Audio-first: a voice product whose landing page cannot be
  *heard* is failing at the only demonstration that matters.
- **Studio** — the generation UI. The existing `voice-web` is a developer tool;
  this is the paid product and it needs to be as good as the thing it competes
  with.
- **Voice library** — catalogue voices per language, plus cloning where we have
  it. Consent records stay mandatory; that machinery already exists and is one
  of the few places we are ahead.
- **Accounts, credits, billing.** Rupees, UPI, Razorpay. Credits metered per
  character, the unit the whole market prices in.
- **API** — keys, rate limits, usage, docs. Developers are the compounding
  channel; a creator churns, an integration does not.
- **Desktop app** as the offline tier, signed and notarised. It already exists
  (§5) and is 90% of a premium SKU nobody else can offer.

### Phase C — Serving

- Launch on **Mac hardware**; it needs no porting (§3.2).
- Instrument **cost per generation** from day one. The price ceiling is Sarvam's
  ₹2.70/finished minute and the margin has to live under it.
- Queue, per-key rate limits, and a hard concurrency cap. The synthesis lock in
  `web/server.py` is already the right shape and already refuses rather than
  queues — that decision survives.
- Port to PyTorch/CUDA when queue depth demands it, not before. The weights are
  the same MIT checkpoint; it is our wrapper that is MLX-specific.

### Phase D — The recording flywheel

The founder's idea, and it is a good one: after a generation, let the speaker
record the same line, and keep the pair. See §14 for what it is and is not.

### Phase E — Enterprise and offline, as a tier

Everything in `outreach/` survives, repositioned: the regulated-enterprise
material now sells the **top tier of a product** rather than the whole company.
That is an easier sale, not a harder one, because the buyer can see a working
public product first. The DPDP deadline of 13 May 2027 is unchanged.

---

## 8. Pricing

Anchored on the market's own units, not on ours.

| Tier | Shape | Notes |
| --- | --- | --- |
| Free | A few thousand characters/month, watermarked | Enough to hear it; not enough to ship with |
| Creator | Monthly, in ₹, credits per character | Undercut a dollar-priced competitor on purchasing power, not on cost |
| Developer | API keys, usage-metered | The compounding channel |
| Desktop / offline | One-time or annual licence | **Nobody else sells this.** Price on capability, not on volume |
| Enterprise | On-prem, hand-installed | `outreach/PRICING.md` still applies |

Two rules that do not bend. **Rupees and UPI**, because a card wall in dollars
loses the market this is named after. And **the free tier must be small enough
to have a real cost ceiling** — hosted inference is now a COGS line, and a
generous free tier on someone else's GPU bill is how this dies quietly.

---

## 9. Risks, restated for the platform

| Risk | Likelihood | Mitigation |
| --- | --- | --- |
| **We cannot get past Hindi cleanly** | **Medium-high** | Phase A is first for this reason. Parler is the fallback; catalogue-not-cloning is the honest degraded product |
| Serving costs exceed price | **High at first** | Measure per-generation cost before launch, cap the free tier, Macs before GPUs |
| Competing head-on with better-funded incumbents | **High** | Do not fight on model quality (§1.1). Fight on price in ₹, offline, and support |
| Quality claim outruns evidence | Medium | `outreach/CLAIMS.md` already governs this and applies unchanged to marketing copy |
| MLX lock-in blocks scale | Medium | PyTorch original exists; port is a wrapper rewrite, not a model rewrite |
| Solo founder, now with an uptime obligation | **High** | A paid API is a promise to be awake. Status page, honest SLAs, and do not sell enterprise support that cannot be staffed |
| Offline tier cannibalises hosted | Low | Different buyers. The offline buyer cannot use the hosted one by law |

---

## 10. The verification table — every guess, replaced

| Draft's guess | Actual | Source |
| --- | --- | --- |
| ElevenLabs ~$5/22/99/330 per month | Free / Starter / Creator $22 (121k credits) / Pro $99 (600k) / Scale $299 / Business $990 (6M) | vendor + secondary, medium confidence |
| "ElevenLabs Hindi sounds like accented English" | **False.** Highest Hindi naturalness (3.84 MOS) in independent pilot; India page with data residency + 6 named Indian customers | arXiv 2510.07978; elevenlabs.io/india |
| Sarvam "check what they charge" | Bulbul v2 ₹15/10k chars; v3 ₹30/10k chars; STT ₹30/hr; ₹100 free credits; API-only, no self-serve cloning, no public on-prem | docs.sarvam.ai — high confidence |
| Indian Hindi VO ₹500–3000/recorded min | ~₹1,000/min e-learning; ~₹60,000 per finished hour; 15–25% volume discounts | industry guides, medium |
| Kaggle ~30 GPU-h/week | Confirmed ~30 h/week, T4/P100, 12 h/session; Colab Pro can now be linked to raise it | medium |
| "Cheapest GPU-hour ~$1–2" | **IndiaAI Mission: ₹65–92/hour, 38,000+ GPUs, up to 40% subsidy for Indian startups** | TechDodo, medium-high |
| DPDP "rules were being phased in" | Rules notified 13 Nov 2025. Consent Manager 13 Nov 2026. Full compliance 13 May 2027. Penalties to ₹250 cr. Biometric consent must be explicit and unbundled | multiple legal, high |
| GST threshold ₹20L services | Confirmed: ₹20L services (₹10L special-category states); ₹40L goods | high |
| Pvt Ltd incorporation ₹6–15k | ₹7,000–25,000 all-in; SPICe+ filing free up to ₹15L capital; DIN/PAN/TAN included | medium |
| IndicF5 licence | Weights **MIT**. The problem was never the weights — it is the `f5-tts` dependency tree. Ranks **last** of 7 on SpeechArenaBench | HF card + arXiv 2604.21481 |
| Indic Parler-TTS | **Apache-2.0**, 21 languages, Hindi 84.79% native-speaker satisfaction, description-based control only, no cloning | HF card, high |
| "Vocos MIT / BigVGAN check NVIDIA terms" | Moot — Chatterbox Multilingual (MIT) removes the whole f5-tts subtree | verified locally |
| Sarvam AI position | Open-sourced Sarvam-30B/105B LLMs (Apache) Mar 2026; **Bulbul/Saarika stay API-only**; Bulbul v4 shown 30 Jul 2026; Sarvam Edge (14 Feb 2026) is speech-only, 74M params, ~294 MB, no agent layer, no announced availability | vendor + secondary |
| "Krutrim, Cartesia, Camb.ai, Smallest.ai" | Krutrim best Hindi prosody/pronunciation/clarity in pilot; Cartesia Sonic 3 tied 2nd overall; **Murf AI** ($10M Series A, Falcon 55 ms, MultiNative mid-sentence Hinglish) and **Dubverse** are the Indian dubbing incumbents | mixed |
| Bhashini "not a competitor" | It is a price floor: ~₹250/month for 50k chars/day; DIBD exists specifically to onboard startups; commercial terms by email | vendor, medium |
| Indian pre-seed range | Pre-seed ₹25L–2Cr; angel ₹50L–3Cr; institutional seed ₹1–10Cr; angel tax abolished 2024 | medium |

---

## 11. The next two weeks

**Days 1–4 — Phase A, languages.** The critical path and the only item that can
fail on technical grounds. Test Chatterbox on Marathi and Nepali (same script,
free). Spike Indic Parler-TTS behind the existing engine seam. Decide the
cloning-vs-catalogue split and write it down honestly.

**Days 5–9 — Phase B, the surface.** Landing page that can be *heard* before
signup. Rebuild the studio UI. Accounts and credits in rupees.

**Days 10–12 — Phase C, serving.** Run it on one Mac behind a queue. Instrument
cost per generation. Do not port to CUDA yet.

**Days 13–14 — Phase D.** The recording endpoint (§14). It costs little and it
starts accumulating the one asset that compounds.

**Not in the next two weeks:** CUDA port, fine-tuning, SOC 2, telephony
integrations, the enterprise motion. All real, none of them first.

---

## 12. If you only do five things

1. **Get past Hindi.** Five Indian languages, each measured. Everything else on
   this plan is downstream of it, and it is the one that can fail.
2. **Make the landing page playable.** A voice product that cannot be heard
   before signup is failing at its only demonstration.
3. **Charge in rupees, over UPI.** The market this is named after does not carry
   dollar-denominated cards.
4. **Know your cost per generation before you price.** Sarvam's ₹2.70/minute is
   the ceiling and the margin lives under it. Hosted inference is a COGS line
   now, not a constraint the code enforces away.
5. **Keep the licence audit and the claims register.** They cost nothing to
   maintain and they are why this project has never had to retract a number.
   §1.1 is what happens to people who skip them.

---

## 13. What did not change

Worth stating plainly, because a reframe this size invites throwing out things
that were never strategy-dependent:

- **Sections 1 and 2 are evidence, not opinion**, and they still hold. Rendered
  audio *is* commoditising; ElevenLabs *is* strong at Hindi; Sarvam *is* ₹2.70 a
  minute. The platform has to be built with those true, not by forgetting them.
- **`outreach/CLAIMS.md` governs marketing copy**, and applies harder now that
  copy will be public and permanent rather than in ten emails.
- **Round trip is an alarm, never a ranking** — AUC 0.625 among working systems.
  This does not become less true because the product is now a website.
- **The licence audit stays a build gate.** It is the cheapest diligence artifact
  in the company and the reason the dependency tree is clean.
- **Consent records stay mandatory for cloning.** They are now a *product*
  feature and a legal necessity, not an internal nicety.
- **Publish your own ceilings.** It was the credibility move for enterprise
  buyers and it is the credibility move for developers reading docs.

---

## 14. The recording flywheel — assessment

The founder's proposal: after a generation in a given voice, let that speaker
record the same line, keep both, compare, and use it to improve the voice.

**The data shape is excellent, and this is the strongest idea in this document.**
It produces `(text, synthetic, real, same speaker, same sentence)` — an aligned
parallel corpus, collected from real usage, with consent already attached
because the speaker is the one recording. That is precisely the corpus a
fine-tune wants and it is normally expensive to buy. Every generation becomes a
chance to acquire one, and the people most motivated to record are the ones who
care most about their own voice quality. It compounds, and no competitor gets it
for free.

**Three corrections to how it was described, none fatal:**

1. **Comparison is not training.** Nothing trains from a diff. This builds a
   *dataset* and a *measurement*; the training is a separate, later, GPU-bound
   step (IndiaAI, §8 of the old plan). Expect the flywheel to pay off in months,
   not on the next generation.
2. **Not on this machine.** An 18 GiB Mac cannot fine-tune a TTS model, and the
   per-voice fine-tune path was retired with IndicF5 (§4). It has to be rebuilt
   on Chatterbox and run on rented compute.
3. **Beware the obvious metric.** Round trip cannot judge naturalness, so it
   cannot score the pair. What a *matched* pair genuinely supports, and what
   nothing else in this project has, is: duration ratio per line (we know
   synthetic runs 0.81× — this measures it per sentence rather than in
   aggregate), F0 and prosody contour comparison against a real reference of the
   *same words*, and a proper A/B where a listener is asked which is the human.
   That last one is the identity test `eval/abtest.py` already implements.

**So it is worth building now**, for the data and the measurement, provided the
UI does not promise the user that their voice improves immediately. It does not.

**Do it as a first-class product feature, not a hidden telemetry hook.** Framed
as *"help your voice get better — read this line"* it is a consented,
transparent, opt-in contribution with a visible payoff. Framed as silent
collection it is a privacy incident in a company whose entire credibility rests
on the opposite. The consent machinery in `voice_clone/store.py` already models
this correctly and the endpoint should reuse it rather than invent a second path.