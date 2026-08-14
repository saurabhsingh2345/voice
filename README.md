# Local Voice Agent

A fully local, privacy-first AI voice agent for Apple Silicon. No cloud APIs, no
recurring cost, no data leaving the machine.

## Constraints

1. **Permissive licenses only** — Apache-2.0, MIT, BSD. Enforced in code by
   `src/voiceagent/models.py` and checked by the diagnostic below.
2. **Memory budget** — the resident pipeline stays under 12 GiB on an 18 GiB
   M3 Pro, leaving ≥4 GiB for macOS.
3. **Zero recurring cost** — local inference only.
4. **Modular** — STT, LLM, TTS, and tools each sit behind an interface in
   `<module>/base.py` and can be swapped independently.

## Setup

```bash
uv sync
uv run voice-doctor
```

`voice-doctor` reports RAM, Metal/MLX status, the license audit, and the memory
budget table. It exits non-zero if any constraint is violated.

## Phase 1 results — STT

Measured on M3 Pro / 18 GiB, 4 synthetic fixtures (1.4–14 s), median of 3 runs.
RTF = compute time ÷ audio duration; lower is better.

| Engine | Load | Model mem | Median RTF | Notes |
| --- | --- | --- | --- | --- |
| moonshine tiny-streaming | 9.8 s | 131 MiB | 0.063 | fastest, weakest accuracy |
| **moonshine small-streaming** | 15.8 s | 228 MiB | 0.120 | **chosen** |
| moonshine medium-streaming | 15.7 s | 481 MiB | 0.131 | best Moonshine accuracy |
| mlx-whisper large-v3-turbo | 4.6 s | 2362 MiB | 0.104 | most accurate, 10× memory, pulls torch |

Moonshine won the live-loop slot: it streams natively (Whisper has a fixed 30 s
window, so its "streaming" is a re-decode of a growing buffer), it costs ~10× less
memory, and it avoids a torch dependency. Whisper stays available for batch work
where accuracy matters more than latency.

Reproduce:

```bash
uv run python -m voiceagent.stt.benchmark                     # all four engines
uv run python -m voiceagent.stt.live                          # live mic
uv run python -m voiceagent.stt.live --file fixtures/medium.wav   # replay a WAV
```

## Phase 2 results — LLM

Qwen3-4B-Instruct-2507, 4-bit MLX. Load 5.3 s, **2.16 GiB** peak.

Time-to-first-token is what gates the voice loop, and it is dominated by prompt
processing, not generation:

| Prompt | Tokens | TTFT |
| --- | --- | --- |
| bare | 35 | 185 ms |
| + tool schemas | 234 | 505 ms |
| + 12-msg history | 209 | 465 ms |
| + both | 408 | 703 ms |

One dummy tool costs ~200 prompt tokens and ~300 ms. Since the system prompt and
tool schemas never change between turns, the engine reuses their KV cache:

| Turn | No cache | Prefix cache |
| --- | --- | --- |
| 1 | 491 ms | 523 ms |
| 2 | 469 ms | **165 ms** |
| 3 | 540 ms | **191 ms** |
| 4 | 603 ms | **146 ms** |

Uncached TTFT *grows* with the conversation; cached stays flat. That is the
difference between a loop that stays responsive and one that degrades as you
talk to it.

Reproduce:

```bash
uv run python -m voiceagent.llm.agent          # scripted demo incl. tool call
uv run python -m voiceagent.llm.agent --chat   # interactive
uv run python -m voiceagent.llm.benchmark      # latency tables above
uv run python -m pytest tests/ -q              # 92 tool-parser tests
```

## Phase 3 results — TTS

Kokoro-82M bf16 on MLX. Load 3.3 s, **0.68 GiB** peak, 24 kHz output.

| Approach | First audio |
| --- | --- |
| synthesize the whole reply, then speak | 822 ms |
| **sentence-streamed** | **280 ms** |

Streaming starts speaking 543 ms sooner — 66 % less silence — because the first
sentence is synthesized while the LLM is still writing the rest.

```bash
uv run python -m voiceagent.tts.demo                        # batch vs streamed
uv run python -m voiceagent.tts.demo --text "Hello there."  # speak a line
uv run python -m voiceagent.tts.demo --voices               # list voices
```

### A GPL dependency was avoided deliberately

Kokoro's text frontend (misaki) falls back to `phonemizer` for out-of-dictionary
words, and **phonemizer is GPLv3** — viral copyleft that would force the entire
packaged desktop app to be GPL. misaki makes that fallback optional; only
mlx-audio imports it unconditionally. `tts/kokoro_engine.py` installs a stub so
the fallback is refused and the GPL package is never installed.

The cost is that unpronounceable words are skipped rather than guessed. Measured
coverage on assistant-style text: technical jargon, brand names, numbers, and
everyday speech all resolve cleanly. Only bare symbols (`$`) and raw filenames
(`pyproject.toml`) fail — and the system prompt already tells the model to spell
symbols out. `KokoroEngine.check_coverage(text)` reports gaps for any string.

`voice-doctor` now audits every installed package's license, not just the models,
since copyleft arrives through the dependency tree.

**One accepted exception: `num2words` (LGPL-2.1).** Used unmodified by misaki to
turn digits into words. LGPL permits commercial use of an unmodified library, so
it does not restrict the product — but it carries an obligation the
Apache/MIT/BSD set does not: **recipients must be able to swap in their own
build of the library.** Shipping it as an ordinary importable package in
`site-packages` satisfies that. Freezing Python into one opaque binary would
not, so the desktop build must keep the dependency replaceable.

## Phase 4 — the live voice loop

```bash
uv run voice-chat                              # talk to it (headphones advised)
uv run voice-chat --replay fixtures/short.wav  # drive from a file, no mic
uv run voice-chat --no-barge-in                # disable interruption
```

Loads in ~8–14 s, **~5.5 GiB resident** for VAD + STT + LLM + TTS together.

Per-turn latency, measured from the moment you stop talking:

| Stage | Turn 1 | Later turns |
| --- | --- | --- |
| STT final | 163 ms | 370 ms |
| LLM first token | ~1400 ms | 585 ms |
| TTS first audio | 2383 ms | 1832 ms |

**The sub-1 s target is not met.** Turn one costs ~1.4 s at the LLM because the
prefix cache is cold; later turns are ~0.6 s. TTS first audio then adds the wait
for enough text to speak. Priming the cache at load was tried and **did not
help** (1382 ms vs 1457 ms — noise) while hanging the loader on executor
shutdown, so it was removed; `Agent.prime()` remains if someone wants to revisit.

Numbers above were taken with the machine relatively quiet. Under the memory
pressure this Mac is usually in (1.6 GiB free, 25 GiB swap), the same loop
measured 4.9 s to first token. Close other apps before judging it.

### Why not Pipecat

The brief specifies Pipecat, and this deviates. Reasons:

- Pipecat has **no MLX LLM service** — its only local LLM path is Ollama
  (llama.cpp), which contradicts the MLX-first constraint.
- Its local audio transport needs `pyaudio`, which needs a Homebrew
  `portaudio` that would then have to be bundled for the Tauri build.
  `sounddevice` already vendors PortAudio in its wheel.
- Using it meant writing custom services for all three engines plus a
  sounddevice transport — more adapter code than the loop itself.

The engines sit behind interfaces, so swapping this orchestrator for Pipecat
later touches only `orchestration/loop.py`.

### Echo, and why you want headphones

There is no acoustic echo cancellation, so on laptop speakers the mic hears the
agent and can interrupt it. Mitigated with a stricter VAD threshold during
playback (0.85 vs 0.5) plus a 350 ms grace window, but headphones remove the
problem properly. AEC is the real fix and is not built.

## Phase 5 — tools

The agent can act, not just talk. Verified end to end against the real model:

```
you>   What files are in my workspace?
         -> list_files({})   <- database.yml (52 bytes)
you>   Read database.yml and tell me the pool size.
         -> read_file({'path': 'database.yml'})
agent> The pool size in database.yml is 5.
you>   Write summary.txt containing: pool size is five.
         -> write_file(...)  [confirmation required]
```

| Tool | Confirmation | Notes |
| --- | --- | --- |
| `list_files`, `read_file` | no | read-only, sandboxed |
| `write_file` | **yes** | sandboxed, 256 KB cap |
| `run_command` | **yes** | allow-list only, no shell |
| `http_request` | **yes** | the only tool that leaves the machine |

Workspace defaults to `~/VoiceAgentWorkspace`.

### How the sandbox actually holds

Enforced on the **resolved** path, not the string, which defeats three separate
escapes that a naive check misses:

- `../` traversal — resolved before comparison.
- **symlinks** pointing outside the workspace — resolution follows them.
- **prefix collision** — `/x/workspace-evil` passes `startswith('/x/workspace')`
  but fails `is_relative_to`.

The shell tool parses with `shlex` and matches the *executable* against a fixed
allow-list, then runs via `create_subprocess_exec` — never a shell. So `;`,
`&&`, `|`, backticks and `$(…)` are inert bytes. `find` is allow-listed but
`-exec`/`-delete` are refused explicitly, since those would be arbitrary
execution through an allow-listed binary.

`http_request` refuses loopback and private addresses, so the agent cannot be
talked into probing your LAN or calling this project's own API on 127.0.0.1.

**The confirmation gate fails closed**: `Agent`'s default confirm hook returns
`False`, so a tool marked `requires_confirmation` cannot run if someone forgets
to wire up a prompt. Tested. Approval is typed, not spoken — "yes" is exactly
the word a speech recogniser mishears, and this gate protects the filesystem.

## Phase 6 — encrypted memory

Conversations persist across restarts, message bodies are encrypted at rest, and
everything can be wiped on command. Three tools expose it: `remember`, `recall`,
and `forget_everything` (confirmation-gated).

Retrieval runs before each turn: relevant past facts are injected as a *system*
message, not appended to the user's words, so the model can't mistake recalled
context for something just said.

### Not SQLCipher — and what that costs

`sqlcipher3-binary` publishes **no arm64 macOS wheel** (manylinux x86_64 only),
so using it needs `brew install sqlcipher` plus a source build — another system
library to bundle for Tauri, the same reason pyaudio was rejected in Phase 4.

Instead: plain SQLite with every message body encrypted via Fernet
(AES-128-CBC + HMAC), key in the macOS Keychain. Honestly stated:

- **Same as SQLCipher** — content is unreadable without the Keychain key, and
  destroying the key makes it unrecoverable. Both are tested.
- **Weaker than SQLCipher** — the schema, row counts and timestamps stay
  visible to anyone opening the file. SQLCipher encrypts the whole page store.
- **Consequence** — content can't be searched in SQL, so recall decrypts and
  scores in Python. Fine at personal scale, wrong at millions of rows.

Swapping in SQLCipher means replacing the connection factory in
`storage/db.py`, if the system dependency ever becomes acceptable.

## Voice cloning + web UI

```bash
uv sync --extra tts --extra clone
uv run voice-web          # then open http://127.0.0.1:8823
```

Record ~10 s of speech in the browser (or upload a clip), type the consent
phrase, and you can type any text and hear it back in that voice. Everything runs
on this machine; nothing is uploaded.

**Model:** Chatterbox Turbo, 350M params, **MIT** (Resemble AI), zero-shot from a
single reference clip, 24 kHz. Fish Speech — the brief's other suggestion — was
rejected: its weights are CC-BY-NC-SA-4.0, i.e. non-commercial.

### Consent is structural, not a checkbox

- `VoiceProfileStore.save()` takes a `ConsentRecord` as a required argument.
  There is no code path that stores a voice without one.
- The record requires typing `I consent to cloning my voice` exactly — a
  checkbox is too easy to click past for biometric data.
- The reference clip is encrypted at rest (Fernet) with a key in the **macOS
  Keychain**, never beside the data.
- "Delete all my data" removes the clips, the metadata, and **destroys the key**,
  so any stray ciphertext is unrecoverable rather than merely unreferenced.

15 tests in `tests/test_consent.py` assert these properties, including that a
consent-shaped duck-typed object is rejected.

### Performance

Measured on M3 Pro. Model load ~30 s (once per server start); after that:

| Text | Audio | Synthesis | RTF |
| --- | --- | --- | --- |
| "Hello there." | 1.0 s | 1.6 s | 1.64 |
| one sentence | 3.6 s | 5.9 s | 1.63 |
| three sentences | 6.8 s | 8.3 s | 1.22 |

Two fixes got it there, both worth knowing about:

- **Short text was pathological.** With Chatterbox's default 800-token budget,
  "Hello there." took **40.8 s** to produce 0.9 s of audio — the model often
  fails to emit EOS on a short phrase and grinds through the whole budget before
  the vocoder trims it. The token cap now scales with the text.
- **The reference clip was re-encoded every request.** Passing `ref_audio` to
  `generate()` re-runs the speaker encoder and S3 tokenizer over the full
  reference each call — a fixed ~3 s cost that dominates short utterances.
  `prepare_conditionals()` is now called once per voice: **6.5 s → 1.6 s**.

RTF above 1 means synthesis takes longer than the audio lasts, so this is
comfortable for type-and-listen but is **not** fast enough for the real-time
conversational loop. Kokoro (RTF ~0.1) remains the agent's live voice; cloning
is for composed playback.

## Phase 8 — desktop app

```bash
cd desktop/src-tauri && cargo tauri build
open "target/release/bundle/macos/Local Voice Agent.app"
```

2.9 MB arm64 bundle, starts the server in ~6 s, opens the UI in a native window.
Quitting reaps the server and frees the port — verified.

**It is a launcher, not a distributable bundle.** It still needs the project
checkout and its `.venv`. Embedding a Python runtime is separate work, and is
constrained by the LGPL obligation above: `num2words` must remain replaceable,
so Python cannot simply be frozen into one opaque archive.

Two bugs surfaced only when double-clicking the app, which no terminal test
would have caught:

- A GUI-launched app inherits a **minimal PATH** (`/usr/bin:/bin:/usr/sbin:/sbin`),
  so `uv` at `/opt/homebrew/bin` was not found. It now runs `.venv/bin/voice-web`
  directly, falling back to `uv` by absolute path.
- The server **outlived the app** and kept port 8823 bound: on macOS the window
  can be destroyed before the app-level exit event runs. The child is now reaped
  from both events.

Server output goes to `data/server.log` — silencing it had made a failed launch
look identical to a slow one.

## Phase 9 — Hindi

```bash
uv run voice-web                                      # type Hindi, hear it
uv run python -m voiceagent.eval.hindi_tts            # 22-sentence gate
uv run python -m voiceagent.stt.benchmark_hi          # Hindi ASR table
uv run python -m voiceagent.eval.hindi_llm --repeat 3 # LLM register
```

English goes to Kokoro, Devanagari to IndicF5 (MIT, 1.4 GB, MPS), one resident
at a time. Kokoro is not merely accented in Hindi — it **cannot speak it at all**
in this build, because its Hindi route needs espeak/phonemizer and that is GPLv3
and deliberately disabled. So the Indic path is a separate engine, not a setting.

### The bug that took the longest: rope on 16 heads instead of 1

IndicF5 produced confident babble. The same Hindi sentence came back from
Whisper as Indonesian, then Welsh, then Arabic. Everything structural checked
out: 364/364 weights load with no missing or unexpected keys, the bundled
vocoder is **bit-identical** to stock `charactr/vocos-mel-24khz` (mean |delta|
exactly 0.0), the vocab covers the text, CPU and MPS agree.

The cause was a behavioural change in `f5-tts`. IndicF5 shipped in March 2025
against a version whose attention applied the rotary embedding to the *flat*
`[b, n, 1024]` projection and only then split it into 16 heads of 64.
`apply_rotary_pos_emb` rotates `t[..., :freqs.shape[-1]]` — channels `0:64`,
which after that reshape are **exactly head 0**. The model was trained with rope
on one head. Current `f5-tts` splits heads first and rotates all sixteen.

No shape changes, so every weight loads and the failure is fluent nonsense
rather than a crash. That is why it survived every check that looked at
structure. Two flags restore the old semantics, and **both** are required:

| `pe_attn_head` | `text_mask_padding` | Round-trip overlap | Detected |
| --- | --- | --- | --- |
| None (all heads) | True | 0 % | Arabic |
| 1 | True | 55 % | Hindi, garbled |
| **1** | **False** | **92 %** | **Hindi, correct** |

An earlier pass concluded `text_mask_padding` made no difference — true in
isolation, because it does nothing without the rope fix.

**The method that found it:** IndicF5's Hugging Face repo bundles its own
`f5_tts/` package. Diffing that against the installed version localises this
class of bug in one step, and is worth doing before chasing PyPI version
numbers. (EPSS sampling is a red herring: 32 steps is not in its table, so it
falls back to `linspace` and is a no-op at the default.)

### Verify by round trip, never by ear or spectrum

Synthesize, then transcribe the result back with Whisper. Spectral measures are
worse than useless here — the babble measured 0.06–0.10 spectral flatness
against 0.088 for real speech, i.e. indistinguishable from genuine speech while
being nonsense.

Two things the harness has to get right or it lies to you:

- **Normalize both sides before scoring.** Whisper applies *inverse* text
  normalization, so a correct `एक हज़ार दो सौ निन्यानवे` comes back as `1299` and
  scores 39 % against the words it actually spoke.
- **Seed the sampler.** Flow matching starts from Gaussian noise and
  `infer_process` exposes no seed, so one sentence scored 100 % on one run and
  44 % on the next with no code change. The engine now seeds per call, which
  makes degeneration deterministic rather than absent.

Result: **22/22 intelligible**, 85–100 % overlap, all detected as Hindi, across
formal, colloquial, code-mixed and numeric registers.

### Code-mixed Hindi needed transliteration, not a better model

Real speech mixes English in constantly. Every code-mixed sentence lost its
English words — `meeting` vanished, `calendar` came back as `एउननर` — while the
surrounding Hindi scored 88–100 %. All 52 Latin letters *are* in IndicF5's
vocab, so nothing was out of vocabulary; the model simply has no acoustic
mapping for English orthography.

`text/translit_en.py` rewrites Latin into Devanagari: a loanword table first,
because Hindi's spellings for these are conventions no rule derives (`email` is
`ईमेल`, never `एमैल`), then a rule-based fallback so an unlisted word is
approximated rather than dropped. Romanized Hindi is handled too, since `theek`
read as English gives `थीक` rather than `ठीक`. Code-mixed went 0/5 → **5/5**.

The fallback is a floor, not a solution — English spelling is not phonetic, so
`though` and `tough` cannot both come out right. Add frequent words to the table.

### Hindi ASR — and why the planned download was dropped

Measured over 23 Hindi clips, scored by **character** error rate: Hindi matra
placement and word boundaries vary between correct spellings, so WER punishes
differences a listener would not notice.

| Engine | Model mem | Median CER | Median RTF |
| --- | --- | --- | --- |
| **whisper-large-v3-turbo** (`hi` pinned) | 2362 MiB | **4.8 %** | 0.236 |
| whisper-large-v3-turbo (auto-detect) | 2360 MiB | 4.8 % | 0.432 |
| moonshine small-streaming | 643 MiB | 121.2 % | 0.071 |

4.8 % from a model already installed makes `vasista22/whisper-hindi-medium`
(~3 GB) unjustifiable against the memory ceiling. Auto-detection costs roughly
double the compute for identical accuracy, so the language stays pinned whenever
it is known.

**Moonshine's 121 % is not a bad transcription, it is fabrication.** On a Hindi
clip it returned *"In Namaste, my name is Lekh. I am your Sahay Takeli…"* and
carried on inventing a passage about the Hajj. It is English-only here and does
not refuse non-English audio. Nothing errors and the output looks like a
successful transcript, which is why `Transcript` now carries a `language` field
and Moonshine reports `en` rather than `None`: it will not tell you when it is
wrong, so the caller has to be able to notice.

### Getting the best voice out of it: the reference clip is the lever

**Use a clip of 12 s or less, with an exact transcript.** f5-tts clips the
reference to 12 s regardless, so a longer recording is half-wasted — and worse,
the transcript then describes speech the model never hears. The transcript sets
the output length, since duration is estimated as
`(generated text length / reference text length) × reference duration`. Inflate
the denominator and the output comes out short, with syllables swallowed to fit.

Measured on a 21.1 s clip whose transcript described all 21.1 s, synthesizing the
same sentence (round-trip overlap, Hindi pinned):

| Reference handling | Overlap | Output | |
| --- | --- | --- | --- |
| hard-cut audio to 12 s, full transcript | 88 % | 2.89 s | rushed — `सुहावना` → `सहुना` |
| whole audio, full transcript | 82 % | 2.76 s | worse |
| **whole audio, transcript trimmed to match** | **95 %** | **5.00 s** | verbatim |

So `set_reference` now hands the audio over whole — letting f5-tts cut at a
silence boundary rather than mid-word — and trims the *transcript* instead.
Clips of 12 s or less are untouched. Over-long clips are reported rather than
silently accepted.

**Raising `nfe_step` from 32 to 48 changed nothing** once the transcript matched
(95 % either way, ~1.6× the compute), so the sampler was never the constraint.
That is worth stating because it is the obvious knob to reach for first.

### Hindi and Urdu are one language in two scripts

Whisper can transcribe correct Hindi into Perso-Arabic and label it `ur`. A
faithful rendering of `आज मौसम बहुत सुहावना है…` came back as
`آج موسم بہت سہاونا ہے اور آسمان بلکل صاف ہے` — the same sentence, word for word
— and scored **0 % overlap purely because the scripts differ**. Pinned to Hindi
the same audio scored 95 %.

The round-trip check therefore treats `ur` as an alias for `hi` and re-decodes
pinned before scoring. Without that it rejects good audio, which is the most
dangerous kind of test failure: it sends you looking for a bug that isn't there.

### Unresolved: the Indic path fails the license audit

`voice-doctor` exits non-zero right now, and it is right to. Installing
`f5-tts` for IndicF5 pulled in four non-permissive packages, all four
transitively from that one dependency:

| Package | License | Arrives via | Imported on our path? |
| --- | --- | --- | --- |
| `encodec` | **CC-BY-NC-4.0** | vocos ← f5-tts | yes |
| `Unidecode` | **GPL** | f5-tts (direct) | **no** |
| `frozendict` | LGPL v3 | einx ← f5-tts | yes |
| `soxr` | LGPL-2.1-or-later | librosa ← f5-tts | yes |

This is not an accepted exception like `num2words`; it is a **known violation of
this project's own constraint**, recorded rather than waived. Consequences,
stated plainly:

- **Personal use is fine.** Nothing here restricts running it on your own
  machine, which is what this project is.
- **Commercial distribution is not.** `encodec` is non-commercial — the same
  class as Fish Speech and XTTS v2, which were rejected on exactly these
  grounds. Shipping the Indic path commercially would be inconsistent.
- `Unidecode` is GPL, the viral copyleft the Kokoro work went out of its way to
  avoid. It is **not imported** on the synthesis path, so the phonemizer stub
  precedent applies directly and it looks removable.

The English pipeline is unaffected — none of these are needed to run it.

Resolving this means either replacing `vocos` (which is awkward: IndicF5's
bundled vocoder tensors are bit-identical to stock `vocos-mel-24khz`, so it is
the *right* vocoder), or dropping IndicF5 for **Indic Parler-TTS** (Apache-2.0,
no cloning, needs its own older `transformers`). That is a product decision, not
a cleanup, so it is left open.

### Hindi is type-and-listen, not conversational

IndicF5's median **RTF is 3.40** against Kokoro's ~0.1. Three seconds of compute
per second of speech is fine for typing a line and hearing it back, and unusable
for a live loop, so no bilingual voice loop was built rather than shipping one
that stalls every turn.

### The LLM did not need fine-tuning

The suspicion was that Qwen3 would produce stiff, translated-sounding Hindi.
Measured over 60 samples across four prompting strategies, it does not:

| Prompt | Devanagari | Everyday words | formal:everyday | Words/sentence |
| --- | --- | --- | --- | --- |
| baseline | 100 % | 100 % | 0:21 | 7.6 |
| + language rule | 100 % | 89 % | 3:15 | 7.3 |
| + colloquial rule | 100 % | 100 % | 0:18 | 8.2 |
| + few-shot | 100 % | 100 % | 0:21 | 7.4 |

Register was never the problem — it reaches for `मदद` and `ज़रूरत` over `सहायता`
and `आवश्यकता` on its own, with no English leakage and nothing the synthesizer
mangles. So fine-tuning is skipped, which is fortunate: a 0.9B full fine-tune
needs ~14 GB before activations and cannot run on this 18 GB machine.

The language rule went into the system prompt anyway, because reply language
*selects the TTS engine* and neither engine degrades into the other's language.
Emergent is not the same as guaranteed.

Two honest limits on that table. The register lexicon measures word choice, not
fluency — replies containing genuine nonsense (`जैसे गैस के साथ चले जा सकते हैं`)
and inconsistent तू/आप still scored 100 %. And Qwen3 occasionally degenerates
into a repetition loop (`ऊपर से ऊपर ऊपर से ऊपर …` until the token budget runs
out), seen once in ~80 generations and not attributable to any prompt. Judging
Hindi *quality* still needs a native speaker; these numbers only catch the
failures that are mechanical.

## Phase 10 — fine-tuning the voice

Zero-shot cloning has a ceiling: it transfers timbre from a 12-second prompt and
nothing else, so it cannot learn how a particular person lands on the end of a
sentence. Getting past that means fine-tuning IndicF5 on a dataset of one voice.

```bash
uv run voice-web                       # section 4: record clips, correct text
uv run voice-train-prep <profile_id>   # prints the two commands to run
```

Result on 96 clips / 9.72 min: **8 epochs, 648 updates, 13.5 min wall** at
batch_size_per_gpu=800, ~1.2 s/update, peak availability drop ~4.7 GiB.
Round-trip stayed at **81 % INTELLIGIBLE**, so nothing was traded for it. Judged
by the speaker: indistinguishable from his own voice, including on long narration.

### Four things stop the stock trainer, none of which announce themselves

1. **Architecture flags.** `F5TTS_v1_Base.yaml` sets `text_mask_padding: True` and
   `pe_attn_head: null`; IndicF5 needs `False` and `1`. Every tensor still loads, so
   it does not error — it spends hours unlearning the pretrained weights.
2. **Checkpoint key layout.** IndicF5 carries 447 keys as
   `ema_model._orig_mod.*`; the trainer strictly loads 366 as `ema_model.*`.
3. **Tokenizer and base vocab.** Needs `--tokenizer custom` against IndicF5's
   2545-entry vocab, and `prepare_csv_wavs` asserts on a base vocab at a path named
   `Emilia_ZH_EN_pinyin` — the name is just where it is hardcoded.
4. **Both the checkpoint and dataset directories resolve inside `.venv`**, so
   `uv sync` can delete a finished fine-tune.

`voice-train-prep` handles all four and refuses if the trainer would look for the
checkpoint somewhere the prepared one is not.

### The loss curve is not diagnostic. The starting value is.

A full 22-epoch run reached loss 1.73 and looked healthy. It was not a fine-tune:
`Trainer.load_checkpoint` had found nothing to load and trained from **random
weights**, silently. Falling loss looks identical either way.

| | initial loss |
| --- | --- |
| random weights | 8.24 |
| IndicF5 loaded | **0.908** |

Nine times lower, because the loaded model already speaks Hindi. That check now
runs at preparation time, because this class of failure is only cheap to catch
before the run.

### F0 and spectral centroid did not predict identity

Measured against 15 real clips (F0 134.3 Hz), the fine-tune came out *further* from
the speaker on pitch with the short reference (18.6 Hz vs 17.8) and the numbers
preferred a different reference clip entirely. The speaker's own judgement was the
opposite: the file the metrics ranked worst was the one that sounded like him.

So these proxies rank candidates at best. They do not settle identity, and a
measurement that contradicts the person whose voice it is loses.

### What made the dataset usable

Three defects, all found by auditing 86 real clips before spending compute:

- **29 % of the dataset was invisible.** 15 clips exceeded the 8.5 s a batch of 800
  frames holds (93.8 frames/sec), so the trainer skipped them. Re-split: total fell
  0.5 min while *usable* audio rose from ~7.2 to 9.7.
- **Transcription error was training the wrong mapping.** Whisper wrote क्रिपया for
  कृपया and इस पर्ष्ट for स्पष्ट — six of eight prompt-read clips wrong. Fixed at the
  root: when the speaker reads a displayed prompt, the prompt *is* the transcript.
  Asking a 4.8 %-CER model to guess text we already have can only add error.
- **Training text did not match inference text.** `/api/speak` runs
  `normalize_hi()` first, so the model never sees a digit at inference. Export now
  normalizes too.

### Using it

A fine-tune is looked up by profile id, so a trained voice uses its own weights
with nothing to configure. `/api/speak` reports which answered:

```
x-weights: fine-tuned      # or: stock
```

Switching between a fine-tuned and a stock voice costs a ~15 s reload — one Indic
model fits in memory, two do not.

## Phase 11 — a blind listening test

Phase 10 ended on "the speaker says it sounds like him." That is the right kind of
evidence and the wrong amount of it: it is one unblinded opinion from the person
with the strongest reason to hear a resemblance, and it cannot be checked by anyone
else. The two numeric proxies had already contradicted him — F0 and spectral
centroid ranked the sample he identified with *worst* — so the honest position was
that voice quality was unmeasured.

```bash
uv run voice-web                                          # /record-benchmark, then /listen
uv run python -m voiceagent.eval.build_benchmark <profile_id>
```

Two questions, deliberately separate, because they have different answers. A clone
can be indistinguishable from its speaker while still being less natural Hindi than
a competitor's stock voice, and one combined score would hide that.

| | asks | reports |
| --- | --- | --- |
| identity | "is this a real recording, or synthesised?" | fooled rate, Wilson interval |
| naturalness | "how natural is this Hindi, 1–5?" | mean opinion score |

Four conditions, and the fourth is what makes the rest interpretable:

| | |
| --- | --- |
| `real` | the speaker reading the held-out sentences (upper anchor) |
| `vocoded` | those same recordings through the mel-and-vocoder path, nothing else |
| `ours` | the fine-tune on those sentences |
| `stock` | stock IndicF5 on those sentences (baseline) |

Without the vocoder control, a listener can learn to spot the *channel* rather than
the voice, and "they could tell" becomes uninterpretable. If `vocoded` is called
synthetic too, the tell is the vocoder and no amount of fine-tuning fixes it; if
`vocoded` passes as real and `ours` does not, the tell is ours and is fixable.

Blinding is structural rather than policy: item ids are random, the mapping to
systems lives in a manifest the listening UI never reads, and audio is copied under
its opaque id. The natural implementation serves `ours_h1.wav` and leaks the answer
in the URL bar.

### The first run measured nothing, and finding that out was the point

Two listeners rated 24 items. The result looked like a result — ours fooled 25 % of
the time against 100 % for real. It was not one, and all three reasons are now
fixed rather than remembered.

**The real clips were different sentences.** `build_benchmark` falls back to
training clips when the speaker has not read the held-out set, printed a warning,
and scrolled it away. So `real` was four spontaneous 8.4 s clips and `ours` was four
read sentences of 3.5–7.1 s: a listener could sort them by content and length
without hearing a voice. The fallback now **refuses** unless `--allow-unmatched-real`
is passed, and `results()` carries `content_matched` per system so the caveat
travels with the score instead of living in someone's memory.

**The attention filter passed people clicking through.** `ms` runs from first play
to answer, so a rating shorter than the clip proves it was not heard out — but the
threshold was a flat 1.2 s, and one rater answered 8.4 s clips in 2.4 s. Under the
corrected rule, **13 of 24 identity ratings and 8 of 12 naturalness ratings were
rushed**. Worse than losing them: because the conditions differed in length, the
filter cut them unequally, leaving ours n=8 against real n=1 — a filter that
selects on condition biases the comparison it is supposed to protect. Same
sentences on both sides fixes this too, since it makes the durations comparable.

**The vocoder control had audible aliasing, and only it.** Resampling 48 kHz to
24 kHz by dropping every other sample folds everything above 12 kHz back into the
audible band: an 18 kHz tone came through at full energy as 6 kHz. Measured against
`resample_poly`, which attenuates it 57 dB. The control condition therefore carried
a distortion no other condition had. `real` was also the only condition still at
48 kHz — the one axis the design had left uncontrolled. Everything is now
band-limited to 24 kHz, so the vocoder is the only thing separating `real` from
`vocoded`.

The corrected benchmark is 12 held-out sentences × 4 conditions = 96 items. It has
not been rated yet, and there is still no quality number for this project. That is
the accurate statement.

### The model speaks a third faster than the speaker

The held-out recordings make a comparison possible that was not before: the same
sentences, by the person and by the model. Summed over all twelve, the speaker takes
**60.8 s and the model 45.4 s — a ratio of 0.75**, worst 0.55 on `h5`.

This is not something fine-tuning can fix, and `ours` and `stock` come out to
identical durations for the same reason: f5-tts sets output length arithmetically,
not acoustically.

    generated_seconds = ref_seconds × gen_bytes / ref_bytes / speed

The enrolment clip is the speaker talking quickly — 5.28 s for 69 characters, an
implied 33.9 bytes/s against the 27.9 bytes/s he actually reads at. Every sentence
inherits that rate. The formula predicts the measured durations within 4 % on the
pure-Devanagari sentences, which is what confirms the mechanism.

`IndicTTSEngine` already takes `speed`; the empirical correction for this profile is
0.75, and it is derivable per profile from clips the speaker has already recorded.
It is not applied yet, on purpose — rate is exactly the kind of change that should
be settled by the listening test rather than by the person proposing it, and
slowing synthesis has to be re-checked against the round trip before it ships.

### Recording the speaker gave the round-trip check a control, and it failed it

Twelve human readings of the sentences the model also speaks is a control the
intelligibility harness never had. Run it on the human and the number should be
close to perfect, because a person reading their own language is intelligible by
construction. It came back **54 %** on the code-mixed sentence — which does not
indict the speaker, it indicts the scorer. Two bugs, both of which had been quietly
depressing every Hindi score:

**`check()` compared the two sides in different scripts.** Whisper writes loanwords
in Devanagari (`डॉक्यूमेंटरी`) while the held-out text keeps them in Latin
(`documentary`), so every character of every English word counted as a miss. The
engine transliterates before synthesis, so Devanagari was always the correct side to
compare on. `hindi_tts` had done this from the start via `normalize_hi`; `check`,
the one the README points you to for a single file, never did. Same sentence after
the fix: **92 %**.

**Whisper was fed aliased audio.** `transcribe` resampled to 16 kHz by dropping
samples, folding everything above 8 kHz back onto the speech about to be judged.
The identical bug was in the vocoder control. It is now one function,
`eval/audio.py`, with the reason written down, because it was independently written
wrong twice.

With both fixed, the human anchor is the useful part:

| | mean overlap | intelligible |
| --- | --- | --- |
| real (the speaker) | **90.2 %** | 12/12 |
| stock IndicF5 | 88.8 % | 12/12 |
| ours (fine-tuned) | 87.3 % | 12/12 |

The ceiling is 90 %, not 100 %. Every previous Hindi number in this README was read
against an implicit 100 % that this metric cannot reach — the residue is Whisper's
spelling disagreeing with the transliterator's, not bad speech. Read that way, both
models sit within ~3 points of a human reading the same sentences.

It also puts the fine-tune slightly *below* stock, entirely on the three code-mixed
sentences (73/75/82 against 81/75/85). Fine-tuning on one speaker's Hindi-only clips
plausibly costs a little English handling. At twelve sentences that is a hypothesis,
not a finding, and it is one the listening test is better placed to settle than this
metric is.

## Phase 12 — off IndicF5, onto Chatterbox Multilingual

`voice-doctor` failed for months with `licenses=False`, and this is the change
that fixed it. It is recorded here because the reasoning matters more than the
diff, and because two of the things it turned up were surprises.

**The problem was never IndicF5's weights.** Those are MIT. It was the `f5-tts`
package needed to run them, which drags in `encodec` (CC-BY-NC), `Unidecode`
(GPL), `frozendict` (LGPL-3) and `soxr` (LGPL-2.1). A non-commercial licence
restricts *use*, not only redistribution, so "we are a service, we never ship
weights" does not clear CC-BY-NC. There used to be a LAN split here — a second
machine ran `--extra indic` and this one ran `--extra remote` — which kept those
packages out of *this* venv and turned the audit green locally. That isolated
the violation; it did not resolve it. Both the extra and the service are gone.

**The fix was already installed.** `mlx-audio` ships Resemble AI's Chatterbox
Multilingual alongside the Chatterbox Turbo used for cloning, and it speaks
Hindi:

```
mlx_audio/tts/models/chatterbox/chatterbox.py:43    "hi": "Hindi",
```

MIT, 23 languages, zero-shot cloning from the same reference clip. No new
dependency.

**Quality was measured before the switch, not assumed.** Same 12 held-out
sentences, same reference clip, the project's own round-trip scorer:

| | human (ceiling) | IndicF5 | Chatterbox |
| --- | --- | --- | --- |
| Mean round-trip overlap | 90.2 % | 88.7 % | **94.0 %** |
| Worst sentence | 81 % | 75 % | **84 %** |
| Code-mixed subset | 91.5 % | 86.0 % | **95.4 %** |
| Aggregate RTF | — | 3.40 | **1.24** |

Better on 9 of 12, tied on 2, worse on none. The biggest gains are digits
(75 → 95 %) and mid-sentence code-switching (81 → 94 %) — the two registers the
held-out set was built to stress. Full write-up and audio in
`eval_out/chatterbox_spike/FINDINGS.md`.

Read the 94 % correctly: it is **above** the human anchor, which means the engine
has reached this metric's ceiling and round trip can no longer discriminate. It
says Chatterbox is not worse and is 2.7× faster. It does not say it sounds
better — intelligibility is not naturalness, and that is what Phase 11 is for.

Sampling follows the recipe in Praxy Voice (arXiv 2604.25441), which tuned
Chatterbox for Indic text: `exaggeration=0.7, temperature=0.6, min_p=0.1`. Its
other finding is load-bearing too — the paper's LoRA helps Telugu and Tamil but
*regresses* Hindi, and its own model card says to use vanilla Chatterbox for
Hindi. That is what this does.

### What it cost

**Ten languages.** IndicF5 spoke 11 Indian languages; this checkpoint speaks
Hindi and no other Indic one. The router still claims every Indic script,
deliberately: `route_for` falls back to `routes[0]`, which *is* the Indic route,
so narrowing it would send Bengali to a Hindi voice silently. Claiming the script
and raising `UnsupportedLanguage` names the language in the error instead.

**Memory.** 3.04 GiB after load and 5.09 GiB MLX peak during generation, against
IndicF5's 1.4 GiB checkpoint. `MIN_FREE_GIB` went from 2.5 to 4.0, so Hindi that
used to synthesize on a very full machine will now be refused.

**Fine-tuning.** Chatterbox clones zero-shot, so there is one checkpoint for
every voice. `voice-train-prep`, the f5-tts trainer path and the per-profile
checkpoint lookup in `web/server.py` are all gone, and Phase 10's fine-tune with
them. The dataset builder stays — a clean transcribed corpus is the asset that
takes weeks to collect, and it is worth having whatever consumes it next.

### Two things the switch turned up

**The 0.75 speed correction does not transfer, and shipping it would have been
wrong.** It was an f5-tts artifact: that model set duration arithmetically from
the enrolment clip. Chatterbox runs at **0.81×** the speaker's duration and
exposes no `speed` parameter at all. Applying the derived 0.75 would have made
output ~8 % too slow.

**The round-trip scorer has a short-clip bug.** h8 ("बिल्कुल, हो जाएगा।", 1.7 s)
auto-detected as **Korean** and scored 0 %. The transcript `밀쿨 호자에가`
romanises to *milkul hojaega* — it is the target sentence. Pinned to `hi` the same
file scores 88 %. This is the Hindi/Urdu case already handled in
`EQUIVALENT_LANGUAGES`, in a script nobody thought to alias, and it fired on 2 of
10 repeats at two different temperatures. Short clips need a pinned re-decode
before round trip is trusted as a gate.

Generation is also unseeded upstream, so the engine seeds per call as
`indic_engine` did. Verified: with a seed the T3 speech tokens are bit-identical
across runs, while the audio differs by ~1.2e-07 peak — Metal reduction order,
not sampling. Compare seeded output with a tolerance, never with `array_equal`.

## Layout

| Path | Role |
| --- | --- |
| `stt/` | Speech-to-text backends (Phase 1) |
| `llm/` | Local LLM + tool calling (Phase 2) |
| `tts/` | Speech synthesis (Phase 3) |
| `orchestration/` | Pipecat pipeline, VAD, barge-in (Phase 4) |
| `tools/` | Tool registry and implementations (Phase 5) |
| `storage/` | Encrypted history and memory (Phase 6) |
| `voice_clone/` | Consent-gated voice cloning (Phase 7) |
| `text/` | Script detection, Hindi normalization, Latin→Devanagari (Phase 9) |
| `eval/` | Hindi round-trip and register harnesses (Phase 9), blind A/B benchmark (Phase 11) |
| `train/` | Reading prompts for voice enrolment (Phase 10; the fine-tune prep went in Phase 12) |
| `diagnostics/` | Environment and budget checks (Phase 0) |
| `models.py` | Model registry, licenses, memory budget |

## Phase status

- [x] **Phase 0** — environment and memory audit
- [x] **Phase 1** — STT standalone (Moonshine small-streaming chosen)
- [x] **Phase 2** — LLM brain standalone (Qwen3-4B 4-bit, tool calling, prefix cache)
- [x] **Phase 3** — TTS standalone (Kokoro-82M, sentence-streamed)
- [x] **Phase 4** — full voice loop with VAD + barge-in (not Pipecat, see below)
- [x] **Phase 5** — tool layer: files, shell, HTTP, with a confirmation gate
- [x] **Phase 6** — encrypted history + memory (not SQLCipher, see below)
- [x] **Phase 7** — voice cloning, consent-gated (Chatterbox Turbo) *(brought forward)*
- [x] **Web UI** — enrol a voice, type text, hear it *(Tauri shell still pending)*
- [x] **Phase 8** — Tauri `.app` bundle (launcher, not yet self-contained)
- [x] **Phase 9** — Hindi: intelligible TTS (22/22), ASR at 4.8 % CER, no *LLM* fine-tune needed
- [x] **Phase 10** — voice fine-tuning: dataset builder, IndicF5 fine-tune, judged
      indistinguishable by the speaker. *The fine-tune itself is retired — see
      Phase 12; the dataset builder is not.*
- [x] **Phase 11** — blind listening test: harness, held-out sentences recorded by
      the speaker. **Built, not yet rated** — see below
- [x] **Phase 12** — Hindi moved from IndicF5 to Chatterbox Multilingual (MIT).
      `voice-doctor` exits 0 with every extra installed

Known gaps, stated rather than buried:

- Hindi covers **Hindi only**. IndicF5 spoke 11 Indian languages; Chatterbox
  Multilingual speaks 1 of them. Other Indic scripts now raise
  `UnsupportedLanguage` rather than being spoken badly (Phase 12).
- The live loop speaks Hindi (RTF 0.63 batch, 0.97 streamed) but cannot **hear**
  it: its STT is Moonshine, which is English-only. Bilingual output, English
  input.
- Hindi needs an **enrolled voice**. Chatterbox clones and has no built-in
  speaker, so Hindi is silent until one is enrolled in `voice-web`.
- No acoustic echo cancellation — use headphones (Phase 4).
- The desktop app is a launcher and still needs the checkout and its `.venv`.
- Hindi *quality* beyond intelligibility still has no result. The blind harness
  exists (Phase 11) and `results()` refuses a verdict below 20 ratings per
  system. F0 and spectral centroid were tried and contradicted the speaker's own
  ear, so they are not it. Note that AI4Bharat has since published
  **SpeechArenaBench** (arXiv 2604.21481) — 120K pairwise comparisons from 1,900
  native raters, sentences and preference data released — which is a far better
  instrument than recruiting listeners here. It also ranks IndicF5 last of seven
  systems, which is part of why Phase 12 happened.
- The synthetic voice runs ~19 % faster than the speaker (0.81×, down from
  IndicF5's 0.75×). Chatterbox exposes no speed control, so there is currently no
  knob to correct it with.

## Rejected models

These are non-commercial and must not be reintroduced (see `DENYLIST` in
`models.py`):

- **XTTS v2** — CPML, non-commercial.
- **F5-TTS** — trained on Emilia (CC-BY-NC-4.0).
- **Fish Speech** — weights are CC-BY-NC-SA-4.0.

**IndicF5** is a different case and worth stating precisely, because the entry
above is easy to over-read. Its weights are MIT and it is not denylisted. What
made it unusable was the `f5-tts` runtime it needs — `encodec` (CC-BY-NC),
`Unidecode` (GPL), `frozendict` (LGPL-3), `soxr` (LGPL-2.1). A permissive model
card is not a clean dependency tree, and the audit is on the tree. See Phase 12.
