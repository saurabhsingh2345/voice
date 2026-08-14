# What we have

Living summary of the project. Amend this after each change; keep it concise —
the README holds the long-form evidence, this holds the current state.

Last updated: 2026-08-14 (Phase 12: Hindi off IndicF5, licence audit green)

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
| TTS Hindi | working | Chatterbox Multilingual (MIT), RTF 1.24, 94.0 % round-trip |
| Live voice loop | working (English) | ~5.5 GiB resident, first audio 1.8–2.4 s |
| Tools | working | files / shell / HTTP, sandboxed, confirmation-gated |
| Memory | working | SQLite + Fernet, key in macOS Keychain, wipeable |
| Voice cloning (zero-shot) | working | Chatterbox Turbo (MIT), RTF ~1.6, consent required |
| Voice dataset builder | working | 96 clips / 9.7 min; the IndicF5 fine-tune it fed is retired |
| Web UI | working | `voice-web` → 127.0.0.1:8823 |
| Desktop app | launcher only | 2.9 MB Tauri arm64, still needs the checkout + `.venv` |
| Blind A/B listening test | **built, unrated** | 12 held-out sentences, real / vocoded / ours |
| Licence audit | **green** | `voice-doctor` exits 0 with every extra installed |

Entry points: `voice-doctor`, `voice-web`, `voice-chat`. 395 tests.

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

- **No Hindi quality verdict.** The blind harness exists but has never been rated
  by enough listeners; `results()` refuses a verdict below 20 ratings/system. F0
  and spectral centroid were tried and contradicted the speaker's own ear.
  AI4Bharat's **SpeechArenaBench** (120K pairwise comparisons, 1,900 native
  raters, data released) is now a better instrument than recruiting our own.

**Known and accepted**

- Hindi is **Hindi only**. Chatterbox Multilingual speaks 1 Indic language where
  IndicF5 spoke 11; other Indic scripts raise `UnsupportedLanguage`.
- Hindi is type-and-listen (RTF 1.24, down from 3.40) — the live loop stays
  English, but this is now an engineering problem, not a research one.
- Hindi needs 4.0 GiB free to synthesize, up from 2.5: the checkpoint is 3.04 GiB
  resident against IndicF5's 1.4 GiB.
- Sub-1 s turn latency is **not** met: ~1.8 s to first audio warm, ~2.4 s cold.
- No acoustic echo cancellation — headphones advised.
- Desktop app is a launcher, not a self-contained bundle (blocked partly by the
  `num2words` LGPL obligation: Python can't be frozen opaquely).
- Synthetic voice runs ~19 % faster than the speaker (0.81 ratio). The old 0.75
  figure was an f5-tts duration artifact and did **not** transfer; Chatterbox
  exposes no speed control, so there is no knob to correct it with.
- Round-trip scorer mislabels clips under ~2 s (a 1.7 s Hindi clip auto-detected
  as Korean, 0 %; pinned to `hi`, 88 %). Same class as the documented Urdu case.
  Needs a pinned re-decode guard before round trip is trusted as a gate.
- Chatterbox generation is unseeded upstream; the engine seeds per call. Seeded
  T3 tokens are bit-identical, audio differs by ~1.2e-07 (Metal reduction order).
- Round-trip intelligibility ceiling is ~90 %, not 100 % — Whisper's spelling
  vs the transliterator's. Older README numbers were read against an implicit 100.
- Latin→Devanagari fallback is a floor, not a solution; unlisted English words are
  approximated. Add frequent words to the table.
- Memory encryption is weaker than SQLCipher: schema, row counts, timestamps stay
  visible; recall decrypts and scores in Python (wrong at large scale).
- Qwen3 occasionally degenerates into a repetition loop (~1 in 80 generations).
- Machine reality: 18 GB Mac, often 2–5 GB free — models load and evict per
  language; LLM fine-tuning is off the table here.

## Future plan

See `plan.md` for the strategy this now serves, and why it changed.

**Next, in order**

1. **Fix the short-clip scorer bug** — re-decode pinned under ~2.5 s. Cheap, and
   everything else that uses round trip as a gate depends on it.
2. **Fix the Qwen3 repetition loop** (~1 in 80). Fatal in a live demo, invisible
   to every quality metric.
3. **Bilingual live loop.** RTF 1.24 is close; streaming/chunked synthesis, a
   warm model and no per-turn reload should get Hindi under 1.0. A Hindi agent
   you can *talk to* is the product; type-and-listen is a demo of a component.
4. **Self-contained desktop bundle** — replace `num2words` with a Hindi/English
   number routine we own (needed for lakh/crore anyway) and drop the last
   recorded licence exception.

**Later**

- Acoustic echo cancellation, so speakers work without barge-in false positives.
- Grow the loanword transliteration table from real code-mixed usage —
  `translit_en` still measurably helps even though Chatterbox takes raw Latin.
- More Indic languages. No longer free via script-detect → engine-route: it needs
  a checkpoint that speaks them.

## Working principles that keep paying off

- **Verify by round trip, not by ear or spectrum.** Babble measured identical to
  real speech on spectral flatness.
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
