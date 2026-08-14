# Phase 0 spike — Chatterbox Multilingual as the Hindi path

Run 2026-08-14. Question: can the Hindi path move off IndicF5/`f5-tts` onto an
MIT engine without losing quality? Method: the 12 held-out sentences, the same
reference clip (`fixtures/hi/reference_lekha.wav`, 8.3 s), the project's own
round-trip scorer — same normalization, same `ur`→`hi` re-decode — so these
numbers are comparable to everything already in the README.

Engine: `mlx-community/chatterbox-multilingual-v3` (MIT) via `mlx_audio`
`0.4.7`, already installed. Sampling from the Praxy Voice recipe
(arXiv 2604.25441): `exaggeration=0.7, temperature=0.6, min_p=0.1`.

## Answer: yes, and it is faster

| | human (ceiling) | IndicF5 | Chatterbox |
| --- | --- | --- | --- |
| Mean overlap | 90.2 % | 88.7 % | **94.0 %** |
| Worst sentence | 81 % | 75 % | **84 %** |
| ≥70 % | 12/12 | 12/12 | 12/12 |
| Code-mixed subset (h1, h3, h12) | 91.5 % | 86.0 % | **95.4 %** |
| Aggregate RTF | — | 3.40 | **1.24** |
| Total audio for the 12 | 60.8 s | 45.4 s | 49.2 s |

Per sentence, Chatterbox beats IndicF5 on 9 of 12, ties on 2 (h5 nuqta 84 %,
h7 exclamation 92 %), and loses on none.

The largest gains land exactly where the draft plan hoped to compete:

- **h2, digits** — 75 % → 95 %
- **h1, mid-sentence code-switch** — 81 % → 94 %
- **h3, three clauses and three code-switches** — 85 % → 97 %

**Read the 94.0 % correctly.** It is *above* the 90.2 % human anchor, which
means the engine is at this metric's ceiling and round trip can no longer
discriminate. It says Chatterbox is not worse and is much faster. It does **not**
say it sounds better — intelligibility is not naturalness. That is what the
listening harness is for, now as a regression test rather than a verdict.

## No new dependencies, and the licence tree clears

`mlx-audio` is already in the `tts` and `clone` extras. The spike imported
nothing from `f5-tts`. Routing `hi` to Chatterbox therefore drops the `indic`
extra entirely and with it `encodec` (CC-BY-NC), `Unidecode` (GPL),
`frozendict` (LGPL-3) and `soxr` (LGPL-2.1).

`voice-doctor` today: `PHASE 0 FAIL (metal=True, licenses=False, budget=True)`.
Licences are the only failing gate.

## Four things the spike found that were not expected

**1. The 0.75 speed correction does not transfer — do not ship it.**
Chatterbox runs at **0.81×** human duration, not IndicF5's 0.75×. The mechanism
was an f5-tts artifact (duration set arithmetically from the enrolment clip) and
a different architecture has a different offset. Applying the derived 0.75
correction here would make output ~8 % too *slow*. Re-derive or drop.

**2. A real bug in the round-trip scorer: short clips get mislabelled.**
h8 ("बिल्कुल, हो जाएगा।", 1.7 s) auto-detected as **Korean** and scored 0 %.
The Korean string `밀쿨 호자에가` romanises to *milkul hojaega* — it is the
target sentence. Re-decoded with Hindi pinned, the same file scores **88 %**,
identical to IndicF5 on h8. This is the Hindi/Urdu bug already documented in
`EQUIVALENT_LANGUAGES`, in a new script. Across 10 repeats of h8 it fired twice,
at both temperature 0.6 and 0.3.

*Fix:* re-decode pinned whenever the clip is under ~2.5 s, or whenever
auto-detect returns a language the text's script cannot produce. Otherwise the
harness will keep scoring short utterances at 0 % and blaming the engine.
The table above uses the pinned 88 % for h8.

**3. Keep `translit_en`.** Chatterbox's multilingual tokenizer handles raw Latin
better than IndicF5 did, but transliterating first is still equal or better:

| | raw Latin | transliterated |
| --- | --- | --- |
| h1 | 94 % | **98 %** |
| h3 | 97 % | **98 %** |
| h12 | 96 % | 96 % |

**4. Generation is stochastic and unseeded.** `mlx_audio`'s Chatterbox exposes no
seed. Five repeats of h7 scored 88–96 %; five of h8 scored 69–88 % pinned. The
per-sentence numbers above are single samples and carry roughly ±5 points of
noise. IndicF5 had the same problem and the engine solved it with a per-call
`DEFAULT_SEED` — do the same here before treating any of this as a regression
gate.

## Caveats

- n = 12, one sample per sentence. Run 3 samples/sentence and take the median
  before this becomes a gate.
- Round trip measures intelligibility only. No claim about prosody or naturalness.
- The checkpoint is 2.7 GB. `aufklarer/Chatterbox-Multilingual-MLX-fp16` (1.29 GB,
  MIT) and a Hindi-specific `aufklarer/Chatterbox-Multilingual-hi-MLX-fp16`
  export of upstream `ResembleAI/Chatterbox-Multilingual-hi` both exist and are
  worth testing against the memory budget — the fp16 export is not in mlx-audio's
  expected layout, so it needs a shim.
- The machine had ~14 GB of swap in use during the run; wall-clock RTF is
  therefore pessimistic, not optimistic.

## Reproduce

Scripts used for this run: `spike_gen.py`, `spike_score.py`, `spike_h8.py`
(job scratch). Audio in this directory; raw scores in `scores.json` — note that
file records h8 at its unpinned 0 %, corrected to 88 % here and in the table.
