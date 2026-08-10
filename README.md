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
- [ ] Phase 3 — TTS standalone
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
