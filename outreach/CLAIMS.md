# The claims register

Every number that may leave this building, where it came from, and what it does
not mean. **Read this before writing any sentence a customer will see.** If a
claim is not on this page, it is not cleared — measure it or drop it.

Written 2026-08-14 against `eval_out/SPECSHEET.md` and `eval_out/arena/FINDINGS.md`.
Regenerate the spec sheet before quoting it in a new quarter; this field moves in
weeks.

---

## 1. The forbidden claim, first, because it is the tempting one

> ❌ **Never quote round-trip overlap against a competitor. Never say or imply we
> beat ElevenLabs, Gemini, Sarvam or Cartesia on quality.**

On our own round-trip metric this engine sits at the **top** of a table
containing Gemini 2.5 Pro and ElevenLabs v3. That is not a result, and
publishing it would be a self-inflicted wound.

Why: the metric was calibrated against 654 arena clips carrying real ratings
from native speakers. It predicts a human intelligibility verdict at **AUC
0.671**, and **0.625 once the one broken system is removed**, against 0.500 for
a coin flip. Inside the band of systems that work, its ordering does not merely
weaken — **it inverts**. It ranks Speech 2.8 HD first; the raters put it fifth
of six.

The instrument that produces our number cannot rank the band it places us in.
`overlap < 0.5` is a trustworthy alarm. `overlap = 0.88` is not a trophy.

`plan.md` §1.1 is a case study of a lab that shipped a ranking claim its own
benchmark contradicted. We would be repeating it **with our own counter-evidence
sitting in the same repo**. Anyone technical enough to buy this is technical
enough to ask how the metric was validated.

What to say instead is §2.

---

## 2. Cleared claims

Each row is safe to say as written. The qualifier is not decoration — it is the
part that makes the claim survive scrutiny.

### Quality position

> ✅ "On held-out code-mixed sentences from SpeechArenaBench — a benchmark of
> 120,000 pairwise comparisons by 1,900 native raters — our fully offline engine
> lands **inside the intelligibility band of the six working cloud systems**, and
> separates decisively from the one that fails (92% paired wins over IndicF5,
> which those raters approved 13% of the time)."

Source: `eval_out/arena/FINDINGS.md` §3. n=40 sentences against their 43–173.

The shape of this claim is *"we are in the band"*, never *"we are above X in the
band."* Inside-the-band is defensible and is enough — the pitch is not that we
sound better, it is that this quality runs with the network off.

> ✅ "Cloud-competitive intelligibility, entirely offline."

> ❌ "Cloud-beating quality." ❌ "Better than ElevenLabs at Hindi."
> ❌ Any ranking claim, ours or theirs, from our own metric.

### Naturalness — say nothing

> ✅ "Naturalness we have not measured. Round-trip scoring is blind to prosody —
> a monotone reading with every phoneme correct scores at ceiling. That needs
> listeners and we would rather tell you than guess."

Volunteering this converts our biggest gap into the credibility move. Preference
in that arena is driven by expressiveness and voice quality, and we have no
instrument for either. Do not let a prospect discover it after the demo.

### Licences

> ✅ "Every dependency in the installed tree is Apache, MIT or BSD. Zero recorded
> exceptions. It is enforced in code, not policy: `voice-doctor` exits non-zero
> on a violation anywhere in the tree, so the build fails."

Source: `eval_out/SPECSHEET.md`. This is a diligence artifact almost nobody in
Indic voice can produce, and it is the claim least likely to be challenged and
most likely to matter to procurement. Screenshot the green run.

Sharpen it with the history — it lands better than the clean state alone:
IndicF5's *weights* were MIT the whole time; the `f5-tts` dependency tree
underneath carried CC-BY-NC, GPL and LGPL. A permissive model card is not a
clean tree, and a licence audit that fails the build is what catches the
difference.

### Speed and footprint

> ✅ "Hindi speech synthesis at **RTF 0.68 median, 0.83 worst case** — synthesis
> keeps up with playback — in 1.33 GiB resident, on an 18 GiB consumer Mac."

> ✅ "**1.8 s to first audio warm, 2.4 s cold.**"

Both from `eval_out/SPECSHEET.md`. Two mandatory riders:

- RTF is **strongly load-sensitive**: the same engine measures 0.63 idle and
  1.18 with a VM running beside it. Quote the number with the host state, or
  quote the range.
- **Sub-1-second turn latency is not met.** If a prospect needs sub-second
  barge-in telephony, say so on the first call. That is not our envelope today.

### Recognition accuracy

> ✅ "Hindi speech recognition at **CER 9.2% on a deliberately hard held-out set**
> — code-switching, digits, retroflex clusters, nuqta consonants, chosen because
> they break things — and 4.8% on ordinary Hindi."

Quote **9.2% first**. Leading with the hard-set number and explaining the 4.8%
underneath reads as rigour; leading with 4.8% and conceding 9.2% under
questioning reads as spin. Same two numbers, opposite effect.

### The intelligibility ceiling

> ✅ "Our round-trip ceiling is ~90%, not 100% — a flawless human recording by the
> speaker scores 90.2%, because the scorer compares Whisper's spelling against
> the transliterator's. We publish the ceiling so our own numbers can be read
> against something."

Publishing your own ceiling is the credibility move. It also pre-empts the
obvious attack on any number above it.

### The product claim — the one the whole pitch rests on

> ✅ "A complete Hindi + English voice agent — speech in, reasoning with tool
> calling, speech out, encrypted memory — running **fully offline on a consumer
> Mac**, with every dependency under a permissive licence, verified by an audit
> that fails the build. Nothing leaves the device. Here are the measurements."

### Competitive position — factual, checkable, no disparagement

> ✅ "Sarvam Edge is the only other on-device Indic player we know of: 74M
> params, ~294 MB, ASR + TTS + translation for 10 languages. It is speech I/O
> only — no LLM, no tool calling, no memory, no agent — and as of now has no
> announced availability date or OEM partnership. Everyone else is a cloud API."

Source: `plan.md` §2.3, from Sarvam's own product page. State it flatly as a
capability difference. Never characterise a competitor as bad — say what each
system *does*, and let the buyer's constraint do the work.

Note the honest asymmetry aloud when it comes up: **they cover 10 Indic
languages and we cover Hindi.** See §3.

---

## 3. What we must volunteer

These are true, they are discoverable, and each one is cheaper to say first than
to concede later. The offline pitch is bought on trust; a single omission
discovered by the buyer costs more than all of them disclosed together.

| Limitation | How to say it |
| --- | --- |
| **Hindi only among Indic languages** | "Hindi and English today. Chatterbox Multilingual speaks 23 languages including Hindi; other Indic *scripts* raise `UnsupportedLanguage` rather than guessing. More Indic languages need a checkpoint that speaks them — it is roadmap, not configuration." |
| **Marathi and Nepali are not refused, and are not supported** | Say this before a buyer discovers it. They share Hindi's script, so they do not hit the error above — they synthesize, intelligibly (0.77 / 0.80 round-trip, nothing under the 0.50 alarm) and with Hindi phonology. Marathi's `ळ` is **not produced at all** — 0 of 4 seeds — and the model substitutes Hindi words for Marathi ones. "It will make a sound; it is not a Marathi voice and we will not sell it as one." The API now warns rather than guessing silently. Evidence: `eval_out/devanagari/FINDINGS.md`. |
| **Weights are not in the bundle** | "The app is 1.3 GB; the models are ~7.6 GB and download once. It runs offline *after* provisioning, not out of the box. For an air-gapped site we pre-stage the weights — ask early, it changes the install." |
| **The `.app` is unsigned** | "Not yet signed or notarised, so Gatekeeper will refuse it on your Mac until we do. In progress. Do not hand a prospect an `xattr` workaround as though it were a shipping answer." |
| **Naturalness unmeasured** | §2 above. |
| **Sub-second latency not met** | 1.8 s warm. Disqualify fast if that is the requirement. |
| **No echo cancellation** | "Headphones for now, or a speakerphone will hear itself." |
| **Hindi needs an enrolled voice** | "It is a cloning model with no built-in speaker, so Hindi is silent until a voice is enrolled — deliberately: shipping a default would mean shipping a real person's voice with no consent record." That framing turns a setup step into evidence of how we treat consent. |
| **The loanword table is unreviewed** | 227 entries, assistant-authored, never read by a native speaker. Do not present it as a curated linguistic asset. |
| **One machine, one speaker** | Every number is from a single 18 GiB M3 Pro, and the arena run was conditioned on one amateur reference recording. The cloud systems we sit beside were not. |

---

## 4. Numbers not to quote without re-checking

- **Competitor pricing** (Sarvam ₹15/₹30 per 10k chars, Bhashini ~₹250/month).
  Checked 2026-08-14. Bulbul v4 was shown 30 July 2026 and pricing will move.
  Re-check before putting a rupee figure in front of a buyer.
- **The Hindi-only Bradley-Terry table** in `eval_out/arena/FINDINGS.md` §1 is
  **our own fit**, not published by AI4Bharat — 43 of 70 shards, ±13 intervals.
  It is honest and it is ours. Attribute it as a fit, never as "AI4Bharat
  published." The Gemini row landing within 4 points of its published pooled
  score is the check that the pipeline reads their data correctly; cite that if
  challenged.
- **Anything about a customer deployment.** There are none. Do not imply one.

---

## 5. The test to apply before sending anything

1. Is every number in this document, or in the spec sheet it points at?
2. Does any sentence rank us against a named system on our own metric?
3. Are the limitations in §3 that a buyer would hit in week one present in the
   message, not just in my head?
4. Would this sentence survive the prospect forwarding it to a sceptical
   engineer with the repo open?

If (2) is yes, or any of (1), (3), (4) is no — rewrite it.

---

*Companion documents: `ONE-PAGER.md` (the leave-behind), `EMAILS.md` (first
contact), `CALL-GUIDE.md` (the fifteen minutes), `PRICING.md`, `TARGETS.md`.
All of them draw their numbers from this page.*
