# Phase 2: a quality verdict, and the limits of the instrument that produced it

Measured 2026-08-14 on an idle M3 Pro (`specsheet --check` green: 0.25 load per
core, 6.6 GiB free). Reproduce with:

    uv run python -m voiceagent.eval.arena votes    # fit the ranking
    uv run python -m voiceagent.eval.arena clips    # audio + human ratings
    uv run python -m voiceagent.eval.arena score    # round-trip their clips
    uv run python -m voiceagent.eval.arena ours     # render ours
    uv run python -m voiceagent.eval.arena compare  # the verdict

Source: AI4Bharat **SpeechArenaBench** (arXiv 2604.21481, dataset MIT, gated).
120K pairwise comparisons, 1,900 native raters, seven TTS systems, audio and
preferences both released. Its README names this use as intended: *"Benchmarking
new TTS systems against the released leaderboard using Bradley-Terry-style
modeling."*

This project's own blind harness has sat at zero listeners for months and
`abtest.results()` refuses a verdict below 20 ratings per system. This is the
way around that, and it stops short of where the data stops.

---

## 1. The Hindi ranking, which the paper did not publish

Table 4 pools all ten languages; the per-language view is a figure with no table
behind it. So Hindi-only, code-mixed-only Bradley-Terry scores had to be fitted:
10,268 Hindi votes, 209 raters, 43 of 70 shards, 500 bootstrap resamples.

| system | Hindi code-mixed BT | published, pooled |
| --- | --- | --- |
| Gemini 2.5 Pro TTS | 1131.3 ±13 | 1135.45 |
| **Bulbul V3 Beta** | **1056.4 ±13** | 1031.28 |
| Sonic 3 | 1034.8 ±12 | 1054.74 |
| Eleven Labs v3 | 1031.0 ±16 | 1054.00 |
| GPT 4o Mini TTS | 1017.3 ±13 | 951.42 |
| Speech 2.8 HD | 1014.6 ±14 | 982.76 |
| Indic F5 | 714.7 ±23 | 812.54 |

Gemini landing within 4 points of its published pooled score is the check that
the pipeline reads the data correctly. It is not a finding.

Two things are:

**Bulbul V3 Beta is second at Hindi**, above ElevenLabs and Sonic, and 25 points
above its own pooled average. Sarvam is materially stronger on Hindi than its
all-language number suggests, and Sarvam is the competitor this project has.

**IndicF5 does not merely rank last on Hindi code-mixed, it falls off**: 714.7
against its own pooled 812.54, three hundred points below the pack. That is
exactly the input this repo routes through Hindi TTS, which makes the migration
off IndicF5 a larger correction than it was argued as at the time.

---

## 2. What our own scorer is worth, measured against 1,900 raters

Every arena clip ships with the audio *and* that rater's six-axis judgement. So
the project's round-trip scorer can be run over their clips and calibrated
against human verdicts on identical audio. 654 clips, all seven systems.

**First, a trap.** The six axes are documented as 1–5 scales. They are binary —
only 1 and 5 ever occur:

    intelligibility {1: 70, 5: 160}   expressiveness {1: 100, 5: 130}
    voice_quality   {1: 94, 5: 136}   liveliness     {1: 96, 5: 134}
    hallucinations  {1: 59, 5: 171}   noise          {1: 42, 5: 188}

A per-system "mean of 4.48" is an 87% good-rate in disguise. Reading differences
between such means as quality gradations is reading a unit that does not exist,
and Pearson on a two-valued outcome is a point-biserial coefficient capped by
class balance. The right instrument is AUC.

**The result:**

| | AUC | n |
| --- | --- | --- |
| all systems | 0.671 | 654 |
| **IndicF5 removed** | **0.625** | 615 |

A coin flip is 0.500. Our round-trip overlap separates broken speech from
working speech, and is close to guessing among systems that work.

The per-system table shows why the pooled number flatters it:

| system | our overlap | human good-rate |
| --- | --- | --- |
| GPT 4o Mini TTS | 0.872 | 80% |
| Bulbul V3 Beta | 0.865 | 75% |
| Gemini 2.5 Pro TTS | 0.859 | 77% |
| Eleven Labs v3 | 0.859 | 57% |
| Sonic 3 | 0.852 | 75% |
| **Speech 2.8 HD** | **0.891** | **70%** |
| Indic F5 | 0.486 | 13% |

Speech 2.8 HD has the highest overlap of any system and the second-lowest
good-rate of the working six. The ordering inside the band is not weak, it
inverts. A system-level Pearson across all seven reads r=0.948, p=0.001 — and it
is a line drawn through a cluster and one distant dot.

**What the scorer *is* good for.** Every clip scoring below 0.5 overlap was also
rejected by humans — 11 of 11 for IndicF5, and no other system produced a single
clip that low. `overlap < 0.5` is a trustworthy alarm. `overlap = 0.88` is not a
trophy.

---

## 3. Where this project's engine lands

40 arena code-mixed sentences, never seen before, rendered by
`ChatterboxIndicEngine` (8-bit, MIT) with the enrolled reference voice, scored
by the same round-trip scorer.

| system | n | overlap | human good-rate |
| --- | --- | --- | --- |
| **Chatterbox Multilingual 8-bit (this repo)** | 40 | **0.879** | no rater has heard it |
| Speech 2.8 HD | 86 | 0.879 | 60% |
| GPT 4o Mini TTS | 47 | 0.860 | 79% |
| Gemini 2.5 Pro TTS | 43 | 0.857 | 72% |
| Bulbul V3 Beta | 85 | 0.851 | 74% |
| Eleven Labs v3 | 173 | 0.850 | 54% |
| Sonic 3 | 43 | 0.849 | 72% |
| Indic F5 | 29 | 0.505 | 10% |

Paired on identical sentences, how often we score higher:

    vs Indic F5          12/13  (92%)
    vs Sonic 3           10/15  (67%)
    vs Bulbul V3 Beta    14/25  (56%)
    vs Eleven Labs v3    23/45  (51%)
    vs Speech 2.8 HD     19/47  (40%)
    vs Gemini 2.5 Pro    7/18   (39%)
    vs GPT 4o Mini TTS   8/23   (35%)

Median RTF 0.58, p90 0.64, one sentence of forty above real time at 1.18. No
clip fell below 0.57 overlap; no synthesis failed.

---

## 4. The verdict, stated as narrowly as the evidence allows

> A fully offline Hindi voice engine, on a consumer Mac, under permissive
> licences, lands **inside the intelligibility band of six cloud systems** on
> held-out code-mixed sentences from a benchmark rated by 1,900 native speakers,
> and is **decisively separated from the one system that fails** (92% paired
> wins over IndicF5, which those raters approved 13% of the time). It runs at
> RTF 0.58.

**What may not be claimed, and why it is tempting.** Our engine sits at the top
of table 3 on our own metric, above Gemini and ElevenLabs. That is not a quality
win and must never be published as one: section 2 is the proof that this
instrument cannot rank inside the working band — it puts Speech 2.8 HD first
while humans put it fifth of six. Claiming the top row would be the exact
failure `plan.md` §1.1 documents, committed with our own evidence sitting next
to it.

**Naturalness remains unmeasured.** Round-trip overlap is blind to prosody: a
monotone reading with every phoneme correct scores at ceiling. Preference in
this arena is driven by expressiveness and voice quality, and we have no
instrument for either. That gap needs listeners, and nothing here substitutes.

**Caveats on our own number.** 40 sentences against their 43–173. Our engine is
conditioned on one amateur reference recording of one speaker; the cloud systems
are not. And one arena row contains the literal string `sentence` as its text,
a placeholder in the released data, which we rendered and scored at 0.57.

---

## Notes for whoever runs this next

- The dataset is **gated** (click-through, MIT). Granted to `singhenfec`.
- The Hindi config is **32.5 GiB** because every pairwise row embeds both clips;
  the full release is 241 GiB. The module projects to text columns over HTTP
  range requests — the entire vote fetch grew the local HF cache by **28 KB**.
  `datasets` was rejected for materializing the split before it will filter.
- **Expect the CDN to stall** with a live socket delivering ~880 bytes/second and
  no timeout anywhere in the pyarrow stack. Reads have a 120s deadline and every
  shard and clip is cached as it lands, so a restart costs one shard.
- 43 of 70 vote shards were fetched. The intervals were already ±13 and the link
  had degraded to roughly one shard per several minutes; the rest is resumable
  and would not move a ±13 interval.
