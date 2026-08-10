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
since copyleft arrives through the dependency tree. One accepted exception is
recorded: `num2words` (LGPL-2.1), used unmodified by misaki — **needs a decision
before Phase 8 packaging.**

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
- [x] **Phase 7** — voice cloning, consent-gated (Chatterbox Turbo) *(brought forward)*
- [x] **Web UI** — enrol a voice, type text, hear it *(Tauri shell still pending)*
- [ ] Phase 4 — full voice loop (Pipecat)
- [ ] Phase 5 — tool layer
- [ ] Phase 6 — local memory and storage
- [ ] Phase 7 — voice cloning (optional, consent-gated)
- [ ] Phase 8 — Tauri packaging

## Rejected models

These are non-commercial and must not be reintroduced (see `DENYLIST` in
`models.py`):

- **XTTS v2** — CPML, non-commercial.
- **F5-TTS** — trained on Emilia (CC-BY-NC-4.0).
- **Fish Speech** — weights are CC-BY-NC-SA-4.0.
