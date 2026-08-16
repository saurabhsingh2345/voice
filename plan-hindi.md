# Plan — the best Hindi voice in the world

Companion to `plan.md`. That document is the business; this one is the single
technical goal it now rests on: **make Hindi sound like a real person, prove it,
and only then do the other languages.**

Written 2026-08-16 with web research. Every licence in §3 was checked against the
Hugging Face API rather than a search summary, because this project has been
burned twice by a permissive model card sitting on a tree that was not.

---

## 0. The honest version of the goal, before anything is built

"Best Hindi TTS in the world" needs splitting, because one half is realistic and
the other is not.

**Not realistic: beating Gemini and Bulbul at general zero-shot Hindi.** The
Hindi-only Bradley-Terry table we fitted ourselves (`eval_out/arena/FINDINGS.md`,
10,268 native-rater votes) puts Gemini 2.5 Pro at 1131 and Sarvam's Bulbul V3 at
1056, above ElevenLabs. Those are large models from funded teams with data we
cannot match. A 0.5B model on an 18 GiB Mac is not going to win that on general
quality, and a plan that assumes otherwise is `plan.md` §1.1 repeating — the
mistake of shipping a ranking claim the evidence contradicts.

**Realistic, and better business: the best Hindi *voice*, for voices we own.** A
single-speaker model fine-tuned on enough clean audio routinely beats a general
zero-shot system *on that speaker*. That is how the good ElevenLabs voices work
too. It is also defensible in a way a model choice never is: anyone can download
the same open weights and the same open corpus; nobody else has ten hours of your
studio-quality Hindi with matched transcripts and a consent record.

So the target that should govern this work:

> **On held-out Hindi sentences, native listeners cannot reliably tell our
> flagship voice from a real recording of that speaker, and prefer it to Bulbul
> in paired comparison more often than not.**

Both halves are falsifiable, both are measured by people, and neither requires
beating Gemini at everything.

---

## 1. The one thing blocking all of it

**Naturalness has never been measured here.** Not once. The round-trip scorer is
blind to prosody by construction — a flat monotone with every phoneme correct
scores at ceiling — and Phase 2 measured what it is worth: AUC 0.625 against
native judgement among working systems, against 0.500 for a coin flip. It is an
alarm for broken speech and cannot rank anything.

So today there is no way to tell whether a change made the voice better. Every
improvement below is unfalsifiable until this is fixed, and **no other work
should start first.**

Two things make this cheaper than it sounds, and both are already here:

- **The calibration set exists.** SpeechArenaBench ships 654 Hindi clips *with
  the native rater's six-axis judgement attached*, MIT-licensed. Any candidate
  naturalness metric can be scored against real human verdicts on identical
  audio — exactly the procedure that exposed round-trip. We do not have to trust
  a metric's paper; we can measure it on our own language.
- **The human harness exists and has never been run.** `eval/abtest.py` already
  implements the right experiment — a blind identity test reporting a *fooled
  rate*, "how often was a synthetic sample called real". It refuses a verdict
  below 20 ratings per system and it currently has **zero**. `eval/expressive.py`
  likewise renders an expressiveness grid nobody has listened to.

**The bottleneck is not engineering. It is that nobody has ever listened.** That
is the single highest-value hour available in this project.

---

## 2. What actually makes speech sound human

From the research, and it is consistent across sources:

1. **Prosody is the dominant tell** — pitch contour, phrase breaks, emphasis,
   pacing. Flat intonation and wrong emphasis are what read as robotic. This is
   the biggest lever and the one we have never touched deliberately.
2. **Breathing and micro-pauses.** Systems that insert real breaths and
   intra-sentence pauses stop sounding like recitation. Breath modelling
   measurably improves rated prosody and rhythm.
3. **Phrase-break prediction** is a solved-ish subproblem with published
   approaches, and it is where a small model can buy naturalness cheaply — it is
   a text-side decision, not a vocoder one.
4. **Disfluencies are a trap.** Inserted "um"s can *reduce* naturalness by
   breaking rhythm; fluent delivery with good pausing beats fake hesitation.
   Do not chase this.

Hindi-specific failure modes, from this repo's own held-out set: schwa deletion,
retroflex collapse, nuqta consonants, numbers, and above all **mid-sentence
accent drift on code-mixed English**, which is the most-reported long-form
failure and the thing urban Indian speech does constantly.

A useful calibration on ambition: in a February 2026 blind test the best AI clips
were taken for human by **40%** of listeners. That is the number to beat, and
"fooled rate" in `abtest.py` measures exactly it.

---

## 3. The assets, with licences verified

| Asset | Licence | Size | Use |
| --- | --- | --- | --- |
| **SpeechArenaBench** | **MIT** ✓ | 654 rated Hindi clips, 7 systems | Calibrate any metric; paired comparison against Bulbul/ElevenLabs on identical sentences |
| **IndicVoices-R** | **CC-BY-4.0** ✓ | 74.6 h Hindi, 399 speakers | General Hindi prosody adaptation. Attribution required |
| **IndicVoices** | **CC-BY-4.0** ✓ | large | Same |
| Our own recordings | ours | **9.7 min** | The flagship voice — and it is far too small |
| SPRINGLab/IndicTTS-Hindi | **none declared** ✗ | ~40 h studio | **Do not use.** Undeclared licence is refused for the same reason community requants are |
| IndicMOS (code) | **none stated** ✗ | — | Read the paper, do not vendor the repo |

Two notes that matter.

**IndicVoices-R is mostly extempore** — 70.3 h spontaneous against 4.3 h read —
which is *good* for prosody realism and *bad* for a flagship voice: 74.6 hours
across 399 speakers is about eleven minutes each. It is a corpus for teaching the
model how Hindi is actually spoken, not for building one great voice.

**Our own data is the differentiator and it is 9.7 minutes.** That is the real
gap. A convincing single-speaker fine-tune wants hours, not minutes.

---

## 4. The plan

### H0 — Build the instrument (do this first, nothing else)

Nothing below can be judged without it, and it is mostly measurement, not code.

1. **Calibrate candidate metrics on Hindi.** Score UTMOSv2, NISQA, DNSMOS and
   SQUIM-MOS over the 654 rated arena clips and report Spearman and AUC against
   the native verdicts. Published MOS predictors are trained on English and
   research explicitly reports degradation on Indian languages — so treat every
   one as guilty until measured. Reject anything that cannot rank within the
   working band, exactly as round-trip was rejected.
2. **Run the human panel that already exists.** Recruit 5–10 native Hindi
   speakers, put them through `eval/abtest.py` until it clears its 20-per-system
   bar, and get the first *fooled rate* this project has ever had.
3. **Add a paired comparison against the incumbents.** We hold Bulbul,
   ElevenLabs, Gemini and Sonic audio from the arena. Same sentences, blind, ours
   against theirs, fitted with the `arena_bt` Bradley-Terry code already written.

**Exit test:** a naturalness metric with a *measured* correlation to Hindi native
judgement, and a repeatable listening panel that produces a number in a day.
Until both exist, treat every quality claim as unsupported.

### H1 — Fix what the instrument says is broken (cheap, once you can see)

Ordered by expected naturalness gain per unit of work, not by ease:

1. **Phrase breaks and pausing.** The largest lever per the research and entirely
   untouched here. Text-side, so it costs no inference budget.
2. **Re-run the expressiveness sweep with ears.** `EXAGGERATION = 0.7` was
   inherited from the Praxy recipe and has never been chosen by listening. The
   grid is already rendered.
3. **Fix streamed prosody.** Streaming is deliberately ungrouped (RTF 0.86–1.12
   against batch 0.63) so it can start at the first sentence boundary — but
   ungrouped means no cross-sentence prosody. Measure what that costs in
   naturalness and decide the trade knowingly.
4. **Code-mix accent drift** — the most-cited long-form failure, and our
   held-out set already targets it.
5. **Native review of the 227 loanwords.** Assistant-authored, never read by a
   Hindi speaker, and a wrong entry silently overrides the fallback.

### H2 — The flagship voice (the part that actually wins)

This is where the differentiator is, and it is a **data** project, not a modelling
one.

1. **Record 5–10 hours** of one speaker: consistent mic, consistent room,
   phonetically balanced script, plenty of code-mixed and numeric material
   because that is where the failures are. The dataset builder and consent
   machinery already exist. 9.7 minutes is not close.
2. **LoRA fine-tune Chatterbox on it.** Community toolkits exist and report LoRA
   as more stable than full fine-tuning at this scale, which also fits an 18 GiB
   machine.
   **Heed the warning already in our own code:** the Praxy Voice work found its
   Indic LoRA improved Telugu and Tamil and *regressed Hindi*, and its own model
   card says to use vanilla Chatterbox for Hindi. So a generic Indic LoRA is not
   the move; a single-speaker fine-tune measured against H0 is.
3. **Consider a prosody pass on IndicVoices-R** — multi-speaker adaptation for
   Hindi rhythm, then the single-speaker LoRA on top. Attribution to AI4Bharat
   required by CC-BY-4.0.

**Exit test:** on held-out sentences, our flagship voice beats vanilla Chatterbox
on the panel, and its fooled rate is meaningfully above zero.

### H3 — Prove it, in a form that survives scrutiny

Run the arena protocol properly and publish only what it supports. `CLAIMS.md`
governs every sentence. The claim we would be entitled to — and it is a strong
one — is of the shape:

> "On held-out Hindi sentences judged blind by native speakers, our voice is
> preferred to Bulbul in N% of pairings, and taken for a real recording M% of the
> time."

Not "best Hindi TTS". The narrow claim is the defensible one, and it is the only
kind this project has ever shipped.

---

## 5. What I would deliberately not do

- **Do not switch base models before H0 exists.** Fish Audio S2 Pro (Apache-2.0,
  the top open-weight model on general arenas) and Higgs Audio V2 are worth a
  bounded spike *afterwards* — but without an instrument you cannot tell whether
  a swap helped, and Chatterbox is MIT, integrated, and runs at RTF 0.62 in
  1.33 GiB. Switching costs a licence audit and an integration; do it on
  evidence.
- **Do not chase architecture.** The gap between us and Bulbul is data and
  prosody, not attention layers.
- **Do not train on undeclared-licence data.** IndicTTS is the tempting one —
  studio quality, 40 hours — and it is exactly the `f5-tts` mistake in a new
  costume.
- **Do not add disfluencies.** The research says they hurt more than they help.
- **Do not let Parler back in yet.** Breadth is proven and RTF 3.9–18.9 is not
  serviceable. Hindi first, as you said.

---

## 6. Risks

| Risk | Likelihood | What it does | Mitigation |
| --- | --- | --- | --- |
| **No listeners recruited** | **High** | The whole plan stalls at H0 | It is 5–10 people for an afternoon. Start with people you know; the harness is already built |
| Every open MOS predictor fails on Hindi | Medium | H0.1 yields nothing usable | H0.2 and H0.3 are human and do not depend on it. A metric is a convenience; the panel is the ground truth |
| Recording 5–10 h stalls | Medium-high | H2 never starts | It is the single highest-value asset in the company. Book it like a product deadline |
| Single-speaker fine-tune regresses | Medium | Wasted weeks | Exactly what Praxy hit. H0 catches it in a day instead of at a customer |
| Best-in-world is unreachable | Medium | Positioning breaks | §0 is written so the claim degrades gracefully to "best on our voices", which is still true and still sells |

---

## 7. The order, in one line

**Measure → listen → fix prosody → record hours → fine-tune → prove it.**

The temptation will be to start at "fine-tune", because it feels like the part
that makes it better. It is the part that cannot be evaluated, and this project
has already spent months with a metric that could not rank. Build the instrument
first.
