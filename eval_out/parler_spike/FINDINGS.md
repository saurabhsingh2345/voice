# Spike — Indic Parler-TTS as the breadth engine

**Date:** 2026-08-16. **Verdict: the model is right, the package is not, and the
weights are gated.** No code was written and nothing was installed into the
project environment. The resolution was done in a throwaway venv precisely so
this document could be written without touching `voice-doctor`'s answer.

Phase A item 1 closed the free path (`eval_out/devanagari/FINDINGS.md`): the
Devanagari family does not come for free, so every further language costs a
model. This spike asks what the plan's designated breadth engine actually costs.

## 1. The model is exactly what Phase A needs

`ai4bharat/indic-parler-tts` — Apache-2.0, 937.8M parameters, **18 languages**:

> en, as, bn, gu, hi, kn, ks, or, ml, mr, ne, pa, sa, sd, ta, te, ur, om

That includes Marathi and Nepali, which Chatterbox reads but cannot speak, and
Tamil, Telugu, Bengali, Gujarati, Kannada, Malayalam, Odia and Punjabi, which
Chatterbox cannot even *encode* — zero tokens of those scripts exist in its
vocab. On paper this closes the entire gap in one model.

**Correction to `plan.md`: 18 languages, not 21.** The 21 figure is in the
verification table at §10 and in Phase A. The model card's own metadata lists 18.
Small, but this project's rule is to read the data rather than the card, and the
number appears in material that goes to customers.

At 937.8M parameters this is roughly Chatterbox's weight class, so the memory
budget is not the obstacle — a bf16 or 8-bit checkpoint lands near the 1.33 GiB
Chatterbox occupies today. Nothing about the size rules it out.

## 1b. The port is far smaller than section 4 assumed — UPDATE 2026-08-16

Access was granted (§2 below is now history), and the config and weights say the
vendoring job is mostly assembly of parts transformers already ships.

    text_encoder    t5, 24 layers, d=1024      -> transformers T5, native
    audio_encoder   dac                        -> transformers DacModel, native
    decoder         parler_tts_decoder, 24 x   -> see below

Dumping `model.safetensors` (840 tensors: 219 text_encoder, 223 audio_encoder,
397 decoder, 1 `embed_prompts`) and comparing decoder layer 0 against
`MusicgenDecoderLayer` built from the installed transformers 5.14.1 gives an
**exact match, name for name and shape for shape**:

    self_attn.{q,k,v,out}_proj.weight      self_attn_layer_norm.{weight,bias}
    encoder_attn.{q,k,v,out}_proj.weight   encoder_attn_layer_norm.{weight,bias}
    fc1.weight  fc2.weight                 final_layer_norm.{weight,bias}

Which is not a coincidence — Parler-TTS is MusicGen's architecture with a
different codec. MusicGen already uses T5 for text; Parler swaps EnCodec for DAC.
So the Parler-specific delta is three things, not a modeling file:

  * **`embed_prompts` and a prefixed description.** `prompt_cross_attention` is
    `false` in this checkpoint, so the style description is embedded and
    *prepended* to the decoder input rather than cross-attended. Cross-attention
    carries the T5 encoding of the text to speak.
  * **Fused LM heads.** `use_fused_lm_heads: true` — one `decoder.lm_heads.weight`
    where MusicGen has a ModuleList of nine. A reshape, not a rewrite.
  * **The usual MusicGen machinery around 9 codebooks** — learned positions
    (`rope_embeddings: false`), the delay pattern, and DAC decoding.

**What this does and does not establish.** It establishes that the weights fit
native classes. It does **not** establish identical forward semantics: matching
names and shapes cannot see a difference in the delay pattern, in how the
prefixed prompt is masked, or in positional handling. Those are exactly the
places a port goes subtly wrong and still produces plausible-sounding audio.
Treat this as "much cheaper than feared", not as "done".

## 2. Both official checkpoints are gated — RESOLVED 2026-08-16

    ai4bharat/indic-parler-tts              403 GatedRepoError
    ai4bharat/indic-parler-tts-pretrained   403 GatedRepoError

Access is restricted and must be requested from the model page with a Hugging
Face account. **This blocks all hands-on work** — no config, no weights, no
measurement. It is a form to fill in, not a negotiation, but it needs the user.

Gating is not a licence problem: the licence is Apache-2.0 and stays Apache-2.0.
It is a provisioning problem, and it has one consequence worth planning for — the
desktop bundle already downloads ~7.6 GB of weights on first run, and a gated
repo cannot be fetched by an end user's machine without credentials. Whatever
ships would need the weights pre-staged or re-hosted, which is permitted by
Apache-2.0 but is a step nobody has costed.

Third-party mirrors exist (`RXD03/indic-parler-tts`, `Yogi5/indic-parler-tts-backup`,
`naklitechie/indic-parler-tts`, and an 8-bit `smdesai/indic-parler-tts-hybrid-8bit`).
They are re-uploads by unknown parties with no licence declared, which is the same
reason `models.py` already refuses community Chatterbox requants. Do not route
around the gate this way.

## 3. The `parler-tts` package is ruled out, on the project's own rule

This is `f5-tts` happening again, and it was caught the same way — by resolving
the tree before installing it. Four independent problems, the first alone fatal:

### 3.1 It introduces `soxr`, which is LGPL, and structurally so

    parler-tts
      └─ descript-audiotools-unofficial
           └─ librosa            (requires soxr>=1.0.0)
                └─ soxr          LGPL-2.1-or-later

`soxr` is not currently installed in this project. Its published metadata
declares `License-Expression: LGPL-2.1-or-later`, and run against this project's
own `COPYLEFT_PATTERN` it matches on `LGPL` and is not in `ACCEPTED_EXCEPTIONS`:

    voice-doctor would report ->
      soxr: non-permissive license detected (LGPL-2.1-or-later)

`soxr` is one of the exact four packages named in `pyproject.toml` as the reason
`f5-tts` had to go (with encodec, Unidecode and frozendict). It arrives through
`librosa`, which hard-requires it, so it cannot be dropped with an extras flag.

### 3.2 It pins `transformers` to a single old version

    "transformers>=4.46.1,<=4.46.1"

An exact pin, and this project runs **5.14.1**. Installing the package would
force a major-version downgrade underneath everything else that uses
transformers. Independent of the licence question, the two cannot share a venv.

### 3.3 The published wheel declares no licence at all

The source repository is Apache-2.0 (verified — `LICENSE`, and the header in
`setup.py`). The PyPI artifact declares nothing. The audit reads distribution
metadata, so an empty licence field **passes silently** — worse than a violation,
because it produces a green check with no evidence behind it.

### 3.4 Its DAC dependencies are unofficial republishes

`descript-audio-codec-unofficial` and `descript-audiotools-unofficial` — both
declare MIT, neither is published by Descript. The upstream `setup.py` installs
`descript-audiotools` straight from a git URL with the comment "temporary fix as
long as 0.7.4 is not published". A commercial product should not have a git-URL
dependency of uncertain provenance in its tree.

## 3b. It has been heard — UPDATE 2026-08-16

Generated in a **throwaway venv**, never in the project tree. The licence rule
governs what we install and ship; LGPL constrains distribution, not evaluation,
and `voice-doctor` still audits an environment where `soxr` is absent. Deciding
whether a model is worth adopting is not the same act as adopting it.

Eight sentences, fp32 on MPS, scored with this project's own round-trip scorer
pinned per language. Audio in `audio/`.

| | Language | Overlap | RTF | Chatterbox can |
| --- | --- | --- | --- | --- |
| hi1 | Hindi | 0.69 | 3.90 | yes |
| hi2 | Hindi | 0.74 | 12.02 | yes |
| mr1 | Marathi | 0.58 | 18.41 | no (refused) |
| mr2 | Marathi | 0.90 | 18.86 | no (refused) |
| ne1 | Nepali | 0.55 | 12.03 | badly |
| **ta1** | **Tamil** | **0.82** | 13.59 | **cannot encode** |
| **bn1** | **Bengali** | **0.76** | 14.12 | **cannot encode** |
| **te1** | **Telugu** | **0.83** | 16.80 | **cannot encode** |

**The result that matters is the bottom three.** Tamil, Bengali and Telugu are
scripts with *zero tokens* in Chatterbox's vocab — not badly supported,
unrepresentable — and Indic Parler-TTS reads them at overlaps comparable to what
it manages in Hindi. Nothing here is below the 0.50 alarm. The breadth is real.

**Two things temper it, and neither is small.**

*It is slower than the model we retired for being slow.* RTF 3.9–18.9 against
Chatterbox's 0.62 and IndicF5's 3.40 — and IndicF5's 3.40 was one of the reasons
it went. This is unoptimised fp32 PyTorch on MPS with no quantization and no MLX
port, so it is a ceiling rather than a verdict: Chatterbox went 1.17 → 0.63 on
8-bit alone. But 18x is a long way from 1.0 and there is no evidence yet that the
gap closes.

*Its Hindi is worse than ours.* 0.69 and 0.74 against the 0.91 the Chatterbox
control scores on the same scorer. Read as "do not replace the Hindi path with
it" — this is a breadth engine to sit *beside* Chatterbox, not instead of it,
which is also what the two-engine routing in `tts/router.py` already assumes.

**And it corrected one of our own findings.** Parler's Marathi `सकाळी` came back
as `सकार` — no `ळ` — from a model that genuinely speaks Marathi. That is what
exposed the confound now recorded in `eval_out/devanagari/FINDINGS.md`: `ळ` has
never appeared in *any* Whisper transcript this project has produced, so "0 of 4
seeds" was never evidence about Chatterbox alone. The Marathi refusal stands on
the lexical substitution (शाळेत → शायद, a Hindi word, every seed), which this
model does *not* do.

## 4. There is one clean path, and it is real work

The blocking dependency is only ever used to turn DAC codes into audio — and
**the installed transformers already does that natively**:

    transformers 5.14.1
      DacModel     True
      DacConfig    True

So the `descript-*` → `librosa` → `soxr` subtree, the whole reason the package
fails the audit, is replaceable by a class already in a dependency we ship.

What remains is the Parler modeling code itself. `ParlerTTSForConditionalGeneration`
is not in transformers, and the model repo ships no modeling `.py` and no
`auto_map`, so there is no `trust_remote_code` shortcut — the files are exactly
`config.json`, `model.safetensors`, the tokenizer and the preprocessor config.

That leaves vendoring, which the licence permits (Apache-2.0, with attribution)
and for which this project has precedent — `text/numbers.py` replaced an LGPL
`num2words`, and the espeak fallback is stubbed to keep GPL out. The difference
in scale is the honest caveat: this is a modeling file written against
transformers 4.46 that would have to be ported across a major version, where
generation and cache APIs changed. That is a project, not an afternoon, and it
carries the maintenance cost of code we own but did not write.

**Recommendation.** Request access first — it is free, it unblocks everything,
and until the weights can be fetched none of the above can be validated by
running it. Decide on the vendoring only once the model can be heard, because a
port is only worth paying for if the voices are good, and this project's own rule
is that a model card is not evidence.

## What was actually verified here, and what was not

Verified: the language list and licence from the model card API; the gate (two
403s); the full 96-package resolution of `parler-tts`; `soxr`'s real published
metadata; that string matching this project's `COPYLEFT_PATTERN`; the exact
transformers pin from upstream `setup.py`; the Apache-2.0 `LICENSE`; and that
`DacModel` exists in the installed transformers.

Not verified, because the gate prevents it: that the model loads, what it sounds
like in any language, its real memory footprint, its RTF on this machine, or
whether its 18 languages are each usable. **No quality claim about Indic
Parler-TTS may be made from this document.** It is a packaging and provisioning
assessment only.
