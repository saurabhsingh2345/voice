# Phase 3 — outreach

Everything needed to have the first customer conversation. Written 2026-08-14,
after Phase 2 produced the three evidence artifacts this material rests on:
the measured spec sheet, the licence audit as a green build gate, and a quality
position narrow enough to defend.

The strategy is `plan.md` §7. This directory is that strategy made sendable.

| File | What it is | When |
| --- | --- | --- |
| **[CLAIMS.md](CLAIMS.md)** | Every number cleared to leave the building, and the one that must never — plus what we volunteer unprompted | **Read first, always** |
| [ONE-PAGER.md](ONE-PAGER.md) | The leave-behind. Wedge-neutral; survives being forwarded to a sceptical engineer | After a reply, never attached to a first mail |
| [EMAILS.md](EMAILS.md) | First contact for three wedges, the AI4Bharat letter, the IndiaAI framing, one follow-up | Sending |
| [CALL-GUIDE.md](CALL-GUIDE.md) | Six qualifying questions, the demo order, the eight objections | The fifteen minutes |
| [PRICING.md](PRICING.md) | Deployment pricing, and why per-minute loses | When asked, not before |
| [TARGETS.md](TARGETS.md) | The filter, the sourcing method, an empty tracker | Two hours before any of the above |

---

## The four things this material is built around

**1. The claim is narrow on purpose.** Not "better Hindi voice" — that race has
120,000 pairwise comparisons saying it is over. It is:

> A complete Hindi + English voice agent — speech in, reasoning with tools,
> speech out, encrypted memory — running fully offline on a consumer Mac, with
> every dependency under a permissive licence, verified by an audit that fails
> the build. Nothing leaves the device. Here are the measurements.

**2. The limitations are in the material, not behind it.** Every document
volunteers what does not work: naturalness unmeasured, 1.8 s not sub-second,
Hindi only, weights outside the bundle, no customers yet. An offline product is
bought on trust, and each of those is discoverable in week one — so saying it
first costs a paragraph and buys the rest of the page.

**3. We refuse our own best number.** Round-trip overlap puts this engine above
Gemini and ElevenLabs. Phase 2 calibrated that metric against 1,900 native
raters and found it cannot rank inside the working band. `CLAIMS.md` opens with
the refusal rather than burying it, and `CALL-GUIDE.md` turns it into an answer:
declining to use a favourable number, and explaining why, buys more than the
number would.

**4. Qualifying beats pitching.** The thesis needs buyers for whom India data
residency is genuinely not enough. Most are not. `TARGETS.md` scores that
first and `CALL-GUIDE.md` asks it in question 3 — a fast, clear no is a good
outcome and makes the next call better targeted.

---

## Order of operations

1. **Record the offline demo.** Airplane mode in frame. Still outstanding from
   Phase 2, and every template earns its reply on a claim a thirty-second video
   proves and a paragraph cannot. Nothing below is as valuable without it.
2. **Send the AI4Bharat letter.** No sales cycle, and its value does not depend
   on a reply. Do it this week.
3. **Apply to the IndiaAI Compute Portal.** ₹65–92/GPU-hour with up to 40%
   subsidy, and being accepted is a credibility signal in its own right.
4. **Build the ten** in `TARGETS.md`. All ten rows before sending any.
5. **Send wedges 1 and 2 in parallel.** Enterprise procurement is slow; the
   clock starts the day you send.
6. **Hold wedge 3 until the `.app` is signed.** A prosumer who cannot get past
   Gatekeeper is a lost first impression, and there is no engineer on that call.

---

## Before quoting anything

- `uv run python -m voiceagent.eval.specsheet --check` — exits non-zero on a
  loaded host. Every measurement in this directory is invalid on a busy machine,
  and the demo is slow on one.
- Competitor pricing was checked 2026-08-14. Bulbul v4 was shown 30 July 2026.
  Re-check before any rupee figure reaches a buyer.
- If a claim is not in `CLAIMS.md`, it is not cleared. Measure it or drop it.
