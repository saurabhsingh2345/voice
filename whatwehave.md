# What we have

Living summary of the project. Amend this after each change; keep it concise —
the README holds the long-form evidence, this holds the current state.

Last updated: 2026-08-14 (Phase 3 outreach material in `outreach/`, built on the
Phase 2 evidence — `eval_out/arena/FINDINGS.md`)

## What it is

A fully local, privacy-first AI voice agent for Apple Silicon. Speech in, LLM
reasoning with tool use, speech out — no cloud APIs, no recurring cost, nothing
leaving the machine. English is conversational; Hindi is type-and-listen.
Includes consent-gated voice cloning and a fine-tuning path for one speaker's
voice.

Hard constraints the code enforces:

1. **Permissive licences only** (Apache/MIT/BSD) — `models.py` + `voice-doctor`,
   exits non-zero on violation.
2. **Memory budget** — resident pipeline under 12 GiB on an 18 GiB M3 Pro.
3. **Zero recurring cost** — local inference only.
4. **Modular** — every layer sits behind `<module>/base.py` and is swappable.

## What it does today

| Capability | Status | Numbers |
| --- | --- | --- |
| STT (English, streaming) | working | Moonshine small, RTF 0.12, 228 MiB |
| STT (Hindi, batch) | working | whisper-large-v3-turbo pinned `hi`, CER 4.8 %, RTF 0.24 |
| LLM + tool calling | working | Qwen3-4B 4-bit MLX, 2.16 GiB, TTFT 165 ms warm (prefix cache) |
| TTS English | working | Kokoro-82M, RTF ~0.1, first audio 280 ms streamed |
| TTS Hindi | working | Chatterbox Multilingual 8-bit (MIT), **RTF 0.63**, 93.5 % round-trip |
| Live voice loop | working, **bilingual** | `voice-chat --language en\|hi\|auto` |
| Tools | working | files / shell / HTTP, sandboxed, confirmation-gated |
| Memory | working | SQLite + Fernet, key in macOS Keychain, wipeable |
| Voice cloning (zero-shot) | working | Chatterbox Turbo (MIT), RTF ~1.6, consent required |
| Voice dataset builder | working | 96 clips / 9.7 min; the IndicF5 fine-tune it fed is retired |
| Web UI | working | `voice-web` → 127.0.0.1:8823 |
| Desktop app | **self-contained** | 1.3 GB `.app`, embeds Python; no checkout, no uv, no venv |
| Blind A/B listening test | **built, unrated** | 12 held-out sentences, real / vocoded / ours |
| Quality position (Hindi) | **measured** | inside the band of 6 cloud systems on arena code-mixed; 92% paired wins over IndicF5 |
| Licence audit | **green** | `voice-doctor` exits 0 with every extra installed |
| Outreach material | **written, unsent** | `outreach/` — claims register, one-pager, emails, call guide, pricing, target filter |

Entry points: `voice-doctor`, `voice-web`, `voice-chat`. 566 tests.

`voice-web` prints the source revision it loaded at startup and shows a banner
when the checkout has moved on underneath it — see the note in Limitations.

## Architecture

```
mic → VAD (silero) → STT → LLM (+tools, +memory) → TTS router → speaker
                                                      ├─ Kokoro                 (en, live)
                                                      ├─ Chatterbox Multilingual (hi)
                                                      └─ Chatterbox Turbo       (cloned voice)
```

`src/voiceagent/`: `stt/ llm/ tts/ orchestration/ tools/ storage/ voice_clone/
text/ eval/ train/ web/ diagnostics/ models.py`.

Key design choices, all deliberate and documented in the README: not Pipecat (no
MLX LLM service), not SQLCipher (no arm64 wheel), no phonemizer (GPLv3, stubbed
out), one Indic model resident at a time (memory ceiling), and Hindi on
Chatterbox rather than IndicF5 (Phase 12 — licence tree, and it measured better).

## Limitations

**Blocking / unresolved**

- **Naturalness is still unmeasured**, and round trip cannot stand in for it.
  Measured against SpeechArenaBench's 654 human-rated clips, our round-trip
  scorer predicts a native speaker's intelligibility verdict at AUC 0.671, and
  **0.625 with the one broken system removed** — near the 0.500 coin flip. It
  separates broken speech from working speech and cannot rank within the working
  band, where its ordering actually inverts against the raters. So the intelligibility
  half of the quality question is answered (see below) and the naturalness half
  needs listeners. The blind harness stays, as the regression gate it is good at.

**Known and accepted**

- Hindi is **Hindi only**. Chatterbox Multilingual speaks 1 Indic language where
  IndicF5 spoke 11; other Indic scripts raise `UnsupportedLanguage`. Measured
  2026-08-16 (`eval_out/devanagari/FINDINGS.md`), and there are two different
  ceilings behind that one sentence:
  - **Other scripts are impossible, not merely unsupported.** The vocab holds 124
    Devanagari tokens and zero Bengali/Gujarati/Gurmukhi/Oriya/Tamil/Telugu/
    Kannada/Malayalam. A `[ta]` language token exists with no Tamil script behind
    it, which will mislead anyone who reads the vocab as a language list.
  - **Marathi is refused; Nepali is served with a warning.** Both share Hindi's
    script, so `detect` calls them `hi` and neither is caught by script.
    Marathi's `ळ` came back 0 of 4 seeds and the model swaps in Hindi words
    (शाळेत → शायद), so `/api/speak` and `/v1/speech` answer
    **`400 unsupported_language`** before synthesis — refusing beats billing for
    audio that is confidently not Marathi. Nepali still synthesizes (0.80
    round-trip, Hindi phonology) with an `X-Language-Warning` header, because its
    evidence is weaker (`र्` survived 2 of 4) and so is its detector. Sell
    neither as supported.
- The live loop hears Hindi only with `--language hi` or `auto`; the default is
  English. Moonshine is English-only and does not *fail* on Hindi, it invents
  English — a real Hindi recording came back as "I have given a documentary for
  many years". Whisper turbo costs ~2.3 GiB against Moonshine's 228 MiB, which is
  why it is a flag and not the default.
- `--language auto` inherits Whisper's short-clip language-ID problem: a one-word
  reply may be transcribed in the wrong script. Pinning is more reliable.
- Streamed Hindi sits at RTF ~0.86–1.12 warm — real time with no margin, because
  the stream path is deliberately ungrouped. Batch is 0.63.
- **Hindi needs an enrolled voice.** Chatterbox is a cloning model with no
  built-in speaker, unlike Kokoro, so Hindi is silent until a voice is enrolled
  in `voice-web`. No default speaker ships: it would be a real person's voice
  with no consent record.
- Hindi needs 3.0 GiB free to synthesize, up from IndicF5's 2.5: the 8-bit
  checkpoint is 1.33 GiB resident, 2.77 GiB peak.
- Sub-1 s turn latency is **not** met: ~1.8 s to first audio warm, ~2.4 s cold.
- No acoustic echo cancellation — headphones advised.
- The desktop app is **unsigned and unnotarised**. Gatekeeper will refuse it on
  another Mac until it is signed; `xattr -dr com.apple.quarantine` is the manual
  workaround and is not a shipping answer.
- Model weights are **not** in the bundle (~7.6 GB against the app's 1.3 GB), so
  first use needs network and a long wait.
- Synthetic voice runs ~19 % faster than the speaker (0.81 ratio). The old 0.75
  figure was an f5-tts duration artifact and did **not** transfer; Chatterbox
  exposes no speed control, so there is no knob to correct it with.
- Round-trip scoring ignores the *language label* below 2.5 s and judges on
  overlap alone. Whisper mislabelled a 1.7 s Hindi clip as Korean and scored it
  0 %; pinned to `hi` the same file scores 88 %. Fixed in
  `roundtrip.decode_for_scoring`, but it means genuine babble in a very short
  utterance is now caught by overlap only.
- Chatterbox generation is unseeded upstream; the engine seeds per call. Seeded
  T3 tokens are bit-identical, audio differs by ~1.2e-07 (Metal reduction order).
- Round-trip intelligibility ceiling is ~90 %, not 100 % — Whisper's spelling
  vs the transliterator's. Older README numbers were read against an implicit 100.
- Latin→Devanagari fallback is still a floor for *unlisted* words. The table is
  now 227 entries covering banking, health, education, logistics and the app's
  own vocabulary; measured +4.2 points of round-trip overlap on code-mixed
  sentences from those domains (91.2 % → 95.4 %). Spellings are
  assistant-authored and want a native speaker's read.
- Memory encryption is weaker than SQLCipher: schema, row counts, timestamps stay
  visible; recall decrypts and scores in Python (wrong at large scale).
- Qwen3's repetition loop is bounded, not eliminated: sampling now uses Qwen's
  published top_p/top_k (it was temperature-only), and `_find_repetition_cycle`
  stops generation after 4 repeats of a token cycle. `engine.degenerations`
  counts how often it fires; baseline was ~1 in 80.
- Machine reality: 18 GB Mac, often 2–5 GB free — models load and evict per
  language; LLM fine-tuning is off the table here.
- **A long-running `voice-web` does not pick up code changes**, and used to fail
  obscurely because of it: a server left up across the Chatterbox migration kept
  serving Hindi through the deleted `indic_engine`, whose `f5-tts` had been
  uninstalled beneath it. Every Hindi request failed while the checkout was
  correct. `voiceagent.version` now compares source mtimes against the moment of
  import and the UI says so, but the *fix* is still to restart. The check
  watches this project's source only — a `uv sync` that changes installed
  dependencies without touching `src/` is invisible to it, and that was half of
  what broke the server in the first place. Restart after `uv sync` regardless.

## Billing — built, unpaid

`web/billing.py` (accounts, plans, an append-only credit ledger) and
`web/razorpay.py` (the payment boundary). 731 tests.

Credits are **characters**, money is **paise** as integers. The ledger is
append-only and the balance is its sum — a refund is an appended reverse, never
an edit — so a billing dispute can be answered a month later. Payment references
carry a unique index, which is what makes a retried Razorpay webhook a no-op
rather than a second credit.

| Plan | ₹/month | Characters | ₹/10k | Accounts one machine holds @30% |
| --- | --- | --- | --- | --- |
| free | 0 | 5,000 | — | blocks when spent |
| creator | 499 | 500,000 | 9.98 | **38** |
| developer | 1,999 | 2,500,000 | 8.00 | **7** |

Both paid plans undercut Sarvam's ₹15/10k on the unit the market quotes;
overage is ₹25/10k, deliberately between Sarvam's ₹15 and ₹30 rather than
cheapest. Free blocks at zero; paid plans take overage instead, because a
narration that stops halfway costs more than the overage does.

**The number to keep in view: one Mac supports ~38 Creator accounts**, a hard
revenue ceiling near ₹19,000/month per machine. That falls straight out of the
cost work below and it is when the second machine has to exist.

**What is not done: taking money.** `POST /v1/checkout` answers 503 and the
webhook refuses to verify until `RAZORPAY_KEY_ID`, `RAZORPAY_KEY_SECRET` and
`RAZORPAY_WEBHOOK_SECRET` exist. Everything else runs today — allowances, the
402, the ledger. Signature verification is fully implemented and tested (both
signatures: webhook HMAC over the **raw body** with the webhook secret, checkout
HMAC over `order_id|payment_id` with the API secret — different secrets, easy
and expensive to confuse). There is deliberately **no sandbox fake**: a stub
returning a plausible order id would make the flow look finished and fail on
first contact with the real API, holding someone's money.

## What a generation costs

Measured 2026-08-16, `eval_out/cogs/FINDINGS.md`, module `voiceagent.eval.cogs`.

| | |
| --- | --- |
| Marginal synthesis cost | **~40 s per 1000 characters** (four sweeps, 37.7–41.9) |
| Sustained RTF | ~0.62 swept; **0.85 median in real traffic** |
| Capacity, 100% utilisation | ~2.0M characters/day, ~38 h of audio |
| Electricity per 10k chars | ~₹0.04 — **0.14–0.28% of Sarvam's price** |
| Fixed cost per request | **unresolved**, −0.03 s to 3.27 s across sweeps |

Two things follow. **Margin is not the question** — on hardware we own the
per-character cost is a fraction of a paisa, and the real budget is 86,400
machine-seconds a day, spent one generation at a time. **Charge per character
and do not surcharge short requests**: the slope is solid, the intercept is not,
and the batching penalty is bounded between 1.0x and 2.9x without being known.

Use the *production* RTF (0.85, one request at 4.10) for capacity planning, not
the swept 0.62. The sweep measures the machine; metering measures the service,
and the gap between them is the real overhead.

## Future plan

See `plan.md` for the strategy this now serves, and why it changed.

**Next, in order**

1. **Sign and notarise the `.app`** — the bundle is built and verified running
   outside any checkout; Gatekeeper is what is left. Needs Apple Developer
   credentials, so it cannot be done unattended.
2. **Record the offline demo**, network physically off, airplane mode in frame.
   The last unfinished Phase 2 item, and now the blocker on everything in
   `outreach/` — every template earns its reply on a claim a video proves and a
   paragraph cannot.
3. **Phase 3, the first customer conversation.** The material is written and
   unsent in `outreach/`; read `outreach/CLAIMS.md` before quoting any number
   from this project to anyone. What remains is not writing: build the ten
   names against the filter in `outreach/TARGETS.md`, send the AI4Bharat letter
   (no sales cycle, send it regardless), and apply to the IndiaAI Compute Portal.

The spec sheet (old item 1) shipped; see `eval_out/SPECSHEET.md`.

**Later**

- Acoustic echo cancellation, so speakers work without barge-in false positives.
- Have a native Hindi speaker review the 227 loanword spellings. A wrong entry
  silently overrides the fallback and is never revisited.
- More Indic languages. No longer free via script-detect → engine-route: it needs
  a checkpoint that speaks them. The shared-script shortcut was tested on
  2026-08-16 and closed — see Limitations. Indic Parler-TTS (Apache-2.0, **18**
  languages) is the only route to breadth, and it cannot clone, so breadth
  arrives as catalogue voices rather than cloned ones. Spiked 2026-08-16
  (`eval_out/parler_spike/FINDINGS.md`) and **not started**, for two reasons that
  both need a decision before code:
  - **The weights are gated.** Two 403s on the official checkpoints; needs an
    access request from the user's HF account. Third-party mirrors exist and are
    refused — unattributed re-uploads, no declared licence.
  - **The `parler-tts` package would fail `voice-doctor`**, via
    `librosa` → `soxr` (LGPL-2.1-or-later) — one of the four packages that made
    `f5-tts` untenable. Nothing was installed; the tree was resolved in a
    throwaway venv. The clean path is to vendor the Apache-2.0 modeling code and
    decode with `transformers.DacModel`, which we already have, but that is a
    port across a transformers major version rather than an afternoon.

## Working principles that keep paying off

- **Verify by round trip, not by ear or spectrum.** Babble measured identical to
  real speech on spectral flatness. But know what it is: an alarm, not a
  ranking. Every clip under 0.5 overlap was rejected by human raters too (11 of
  11), and above that it stops discriminating — a 0.88 is not better than a
  0.85. Never quote it against a competitor.
- **Ask whether a specific grapheme survived, not what the mean was.** Marathi
  averaged 0.77 — a pass by any aggregate reading — while failing on the one
  letter that makes it Marathi. The mean was the least informative number
  available; the disqualifying evidence came from grepping a transcript for `ळ`.
- **Sample before believing a bad score.** One generation scored 0.59 and read as
  a phonological failure; the same sentence scored 0.88 on two other seeds and
  the failure did not exist. Chatterbox samples, so a per-sentence number carries
  ~±0.15 of seed noise. Only a defect that reproduces on every seed is a defect.
- **Calibrate a metric before trusting its ordering.** Round trip looked like a
  quality measure for months and reads r=0.948 across seven systems, which is
  one broken outlier dragging a line through a cluster. The check that exposed
  it was removing that outlier and asking again.
- **Read the data, not the data card.** SpeechArenaBench documents six 1–5
  rating scales and ships six binary ones (only 1 and 5 occur), and its
  `preference_model` column holds model names where the card says "Model A".
  Both would have produced confident, wrong numbers in silence.
- **The starting loss, not the loss curve**, tells you a fine-tune actually loaded
  (0.908 loaded vs 8.24 random — falling loss looks the same either way).
- **Control conditions make results interpretable** — the vocoder-only condition is
  what separates "the clone is a tell" from "the channel is a tell".
- **A measurement that contradicts the person whose voice it is, loses.**
- **Diff a model's bundled package against the installed one** before chasing
  version numbers — that is what found rope-on-16-heads.
- **A permissive model card is not a clean dependency tree.** IndicF5's weights
  were MIT the whole time; `f5-tts` was what failed the audit.
- **Check what is already installed before planning a migration.** Chatterbox
  Multilingual had been sitting in `.venv` via `mlx-audio` for months.
- **MLX arrays are thread-affine, and MLX is lazy.** Building a graph works on
  any thread; only `mx.eval` touches the stream, so an off-thread bug surfaces
  20 layers away from its cause. Load and generate on one owned thread.
- Filters that select on condition (e.g. a flat attention threshold across clips
  of different lengths) bias the comparison they are meant to protect.

## Hindi is under real time

Profiled first: T3 (token generation) is ~58 % of synthesis and S3Gen ~42 %, so
quantizing the transformer was the lever. `voiceagent.tts.quantize` builds an
8-bit checkpoint from the MIT source in ~7 s, on first load, and that is now the
default. Community requants exist and all declare no licence, so they cannot go
in `models.py`.

Measured idle, 12 held-out sentences, medians of 3 runs:

| | aggregate RTF | resident | peak | mean overlap | code-mixed |
| --- | --- | --- | --- | --- | --- |
| fp32 | 1.17 | 3.04 GiB | 4.55 GiB | 93.5 % | 94.9 % |
| **8-bit** | **0.63** | **1.33 GiB** | **2.77 GiB** | **93.5 %** | 94.5 % |

1.9× faster on 44 % of the memory at identical quality, and every sentence is
under RTF 1.0.

A caution worth keeping: the first attempt at this measured RTF 6–12 and was
completely wrong. A virtualization process had the host at load average **537**
with 13 GiB of swap in use. `voice-doctor` warns about exactly this. Check the
load average before trusting any timing here.
