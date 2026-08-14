# Plan — revised against what is actually true in August 2026

Companion to `whatwehave.md` (current state) and the bootstrap-to-funded draft.
Written 2026-08-14, **with** web access, after running the verification list that
the draft correctly said had to be run first.

The draft asked to be checked before being acted on. It has now been checked.
Most of the strategy does not survive contact with the evidence. The engineering
assets do, and they point somewhere better.

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

## 3. The revised thesis

> Indian-language **speech quality** is solved and commoditising fast: Gemini,
> ElevenLabs and Cartesia lead on quality, Sarvam sells Hindi at ₹2.70/minute,
> Bhashini gives it away. Competing there is competing on someone else's
> strength at someone else's price.
>
> Indian-language **agents that run where the data cannot leave** are not solved
> by anyone. Sarvam Edge is speech-only. Everyone else is cloud. DPDP, RBI cloud
> guidance and sectoral regulators are converting "we'd prefer local" into
> "procurement requires local" on a deadline of May 2027.
>
> This repo is a working, tested, licence-audited, memory-budgeted, offline
> Hindi+English voice agent with tool calling and encrypted memory on consumer
> hardware. That is the product. The TTS is a component, and it should be the
> best permissively-licensed component available — not a research project.

What changes concretely:

| | Old plan | Revised |
| --- | --- | --- |
| Asset | Proprietary Hindi audio | The local agent runtime + licence/eval discipline |
| Wedge | Best Hindi voice | Only private on-prem/on-device Indic voice agent |
| Buyer | Ed-tech content teams | Regulated enterprises; OEM/ISV embedders |
| Priced against | Human VO ₹1000/min | Cloud voice-agent stacks + compliance risk |
| Model work | Fine-tune IndicF5 | Swap to MIT Chatterbox, drive RTF < 1 |
| Evaluation | Recruit 20 listeners | Ride SpeechArenaBench; keep harness as regression |
| Blocking risk | Licence violation | Same — but now a 1-day fix, not a 2-week fork |

**What survives from the draft, unchanged and still right:** licence discipline
enforced in code; verify by round trip not by ear; the starting loss tells you a
fine-tune loaded; control conditions make results interpretable; a measurement
that contradicts the speaker loses; charge someone early. Those are the reasons
this pivot is even available.

**The recording sprint is demoted, not deleted.** 60–90 minutes of consented
expressive Hindi is still a real asset and still cheap — but it is no longer the
highest-value task, because a better speaker cannot fix a commodity market. Do it
in Phase 2, on the Chatterbox path, when it can be judged.

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

## 7. Phase 3 — First paying customer (6–10 weeks, starts during Phase 1)

**Wedges, re-ranked against the evidence:**

**1. Regulated-enterprise private voice agent.** ⭐ BFSI, insurance, healthcare,
government-adjacent. Their voice data legally should not leave the perimeter, and
the May 2027 DPDP deadline gives them a date. On-prem agentic voice normally
implies H100/L40S clusters; yours runs on a workstation. That is the pitch.
Slow procurement — start now, in parallel with everything.

**2. OEM / ISV embedding.** Device makers, kiosk vendors, POS, in-vehicle,
medical devices, rural-connectivity products. They need offline Indic voice and
cannot ship a cloud dependency. Sarvam Edge targets this and has announced no
partnerships or availability; you have a working stack today. Licence per unit
or per deployment.

**3. Privacy-first prosumer agent.** Doctors, lawyers, journalists, therapists —
professionals with confidentiality duties and no IT department. One-time licence
or modest subscription. Lowest procurement friction, fastest to a first invoice,
and it directly exercises the desktop bundle from Phase 1.

**4. Ed-tech narration.** Demoted from #1 to last. The draft's own logic —
"they're already paying for TTS, so you're a switch not a new budget line" —
now cuts the other way: they can switch to Sarvam at ₹2.70/minute. Take this
work only as opportunistic cash, never as the strategy.

**Pricing.** Not per-minute — you are not selling minutes. Sell deployment:
a pilot at ₹50k–2L for a scoped on-prem installation; per-seat or per-device for
OEM; ₹2,000–5,000 one-time or ₹500/month for prosumer. The comparison is not
₹2.70/minute of audio, it is the cost and risk of the compliance path they would
otherwise take. First deal small but **never free** — the first invoice is still
the hinge, and that part of the draft was right.

**Do the work by hand.** No self-serve platform. You, on a call, installing it.

---

## 8. Phase 4 — Scale (3–6 months)

- **Services first, then product** — the draft's sequencing holds. Two or three
  on-prem deployments teach you the workflow; productise what repeats.
- **Compute.** The draft assumed free tiers were the ceiling. They are not.
  [Kaggle is ~30 GPU-h/week](https://www.kaggle.com/) (T4/P100) and fine for
  small runs, but **IndiaAI Mission has 38,000+ GPUs available to Indian
  startups, researchers and MSMEs at ₹65–92/hour** via the IndiaAI Compute
  Portal, with up to 40% subsidy for approved projects, and **AI Kosh** hosts
  1,000+ machine-readable datasets.
  ([TechDodo](https://techdodo.in/articles/india-ai-impact-summit-2026-gpu-access-guide))
  Apply. This removes compute as a constraint and is also a credibility signal.
- **More voices / more languages** via the same script-detect → engine-route
  pattern. Now genuinely cheap: Chatterbox Multilingual covers 23 languages in
  one model, so Marathi/Bengali become configuration and evaluation, not new
  engines.
- **Metrics from day one:** deployments, devices, per-device licence revenue,
  memory/latency envelope per release, retention, revenue concentration.
  Keep the quality trendline — a measured trendline is a moat narrative.

**Incorporate when:** a client's procurement demands it, you need a payment
gateway, you approach the GST threshold, or an investor conversation gets real.
Private Limited. Sole proprietorship + invoices is fine until then.

---

## 9. Phase 5 — Raise

The pitch is not "better Hindi voice." It is:

> Every Indian-language voice system is a cloud API. DPDP, RBI and sectoral
> regulators are making that untenable for the sectors with the most voice
> traffic, on a May 2027 deadline. We run the whole agent — speech, reasoning,
> tools, memory — offline on hardware people already own, with a licence audit
> that fails the build. Here it is with the network off. Here are the customers.

Realistic numbers: Indian pre-seed runs **₹25 lakh – ₹2 crore**; angel networks
write **₹50 lakh – ₹3 crore**; institutional seed **₹1–10 crore**, with AI and
deep-tech at the upper end. Angel tax was abolished for domestic investors by the
Finance Act 2024. Investors in 2026 expect traction before the cheque.
([Startup Movers](https://www.startup-movers.com/blog/pre-seed-vs-seed-funding-india),
[myHQ](https://myhq.in/guides/seed-funding-guide))

Non-dilutive first: IndiaAI Mission, IIT incubators, NVIDIA Inception. An
AI4Bharat relationship is worth cultivating — they are a supplier and a
credibility source, and you should tell them their own benchmark redirected you.

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

## 11. Risks

| Risk | Likelihood | Mitigation |
| --- | --- | --- |
| Chatterbox Hindi is worse than IndicF5 | **Medium** | Measure in Phase 0 before committing. BUPS recipe, then Indic Parler-TTS, then reconsider. Note IndicF5 ranked last anyway — the bar is lower than it feels |
| Sarvam Edge ships an agent layer | **Medium-high** | Your lead is a working tool-calling agent with memory *today*. Ship the bundle fast. Being small and shipping is the only edge |
| ElevenLabs/Google ship true on-device Indic | Low-medium | Neither has an on-device story; both monetise cloud inference. Watch quarterly |
| Enterprises say "India data residency is enough" | **High** | Many will. Target the ones where it genuinely is not — RBI-regulated, healthcare, government. Fewer buyers, bigger deals |
| On-prem sales cycles outrun your runway | High | That is why prosumer (#3) and opportunistic narration (#4) exist — near-term cash while enterprise ripens |
| The pivot is wrong and quality is the wedge after all | Low | SpeechArenaBench is 120K comparisons and 1,900 raters. Do not argue with it using 12 sentences |
| Solo-founder burnout | High | This plan is now mostly engineering you already know how to do, on a codebase you already have. Ship narrow |

---

## 12. The next two weeks

**Days 1–3 — Phase 0.**
Spike Chatterbox Multilingual on Hindi. Round-trip score against IndicF5. Measure
RTF and the duration ratio. If it holds: rip out `f5-tts`, retire `indic_engine`
and the remote-TTS split, update `models.py`, get `voice-doctor` to exit 0 with
everything installed. Write the decision into the README.

**Days 4–7 — the demo that sells.**
Fix the Qwen3 repetition loop. Drive Hindi toward RTF < 1 and get the live loop
bilingual. Record the offline demo with the network off.

**Days 8–10 — evidence.**
Pull the SpeechArenaBench code-mixed subset. Generate your conditions. Produce
the measured spec sheet — latency, RTF, memory, CER, round-trip against its real
ceiling. Publish the ceiling.

**Days 11–14 — first conversations.**
Apply to the IndiaAI Compute Portal. Write to AI4Bharat about SpeechArenaBench
and what it changed here. Find ten people in regulated Indian enterprises or at
Indic-hardware OEMs and send them a link to the offline demo. Ask for fifteen
minutes.

**Not in the next two weeks:** recording sprint, fine-tuning, other languages,
self-serve platform, incorporation.

---

## 13. If you only do five things

1. **Swap Hindi to Chatterbox Multilingual.** It is MIT, already installed,
   ~2× faster, and it closes every blocking licence item in one commit.
2. **Make the live loop bilingual.** A Hindi agent you can talk to is the
   product; type-and-listen is a demo of a component.
3. **Ship a self-contained bundle.** Nobody installs a `.venv` to evaluate you.
4. **Record the offline demo with the network off**, and publish the spec sheet
   including your own ceilings. The honesty is the differentiator.
5. **Stop trying to beat ElevenLabs at Hindi.** That race has 120,000 pairwise
   comparisons saying it is over. Win the one nobody has entered.

---

*Every number here was checked on 2026-08-14 against the sources linked inline.
The Chatterbox finding was verified against the code in this repo's own `.venv`.
Re-check the pricing and competitive claims before quoting them to a customer —
Bulbul v4 is already announced and this field moves in weeks.*
