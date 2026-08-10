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

## Rejected models

These are non-commercial and must not be reintroduced (see `DENYLIST` in
`models.py`):

- **XTTS v2** — CPML, non-commercial.
- **F5-TTS** — trained on Emilia (CC-BY-NC-4.0).
- **Fish Speech** — weights are CC-BY-NC-SA-4.0.
