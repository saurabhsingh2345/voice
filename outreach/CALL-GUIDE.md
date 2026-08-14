# The fifteen minutes

What to ask, what to show, and the eight objections that will actually come.
Numbers from `CLAIMS.md`; do not improvise one.

The goal of a first call is **not** to sell. It is to find out whether their
constraint is real, because the whole thesis rests on buyers for whom "India
data residency" is genuinely not enough. Most will not be. Finding that out in
twelve minutes is a good outcome.

---

## Before the call

- `uv run python -m voiceagent.eval.specsheet --check` — it exits non-zero on a
  loaded host. **A demo on a loaded machine is a slow demo**, and this is the
  same failure that once measured RTF 6–12 at load average 537.
- Voice enrolled, models warm, headphones on. Hindi is silent without an
  enrolled voice, and there is no echo cancellation.
- Network genuinely off, visibly. Not "I promise it's local."
- Know which of the three wedges they are, and do not pitch the other two.

---

## The first six minutes are questions

Do not demo yet. A demo before you know their constraint answers a question
nobody asked.

1. **"Where does your voice data go today, and who signed off on that?"**
   Opens the whole conversation. The name of the signer tells you who the actual
   buyer is.
2. **"What's your DPDP position on voice recordings specifically — has anyone
   ruled on whether they're biometric for your purposes?"**
   If they have, they are ahead and this is a real opportunity. If the question
   is new, you are educating, and that is a longer, lower-probability sale.
3. **"Is there any category of call you already *cannot* send to a cloud
   vendor?"** ⭐
   **This is the qualifying question.** A yes — RBI-regulated data, patient
   records, a specific state's rules, an internal policy predating DPDP — means
   they have a bounded problem with a budget. A no, or "our vendor gives us India
   residency," means they are §11's High-likelihood risk: politely find out if
   there is a sub-case, and if not, end early and leave the one-pager.
4. **"Hindi, English, or both — and how much code-mixing in real calls?"**
   Scopes it, and surfaces the Hindi-only limit before they discover it.
5. **"What would this have to *do*, beyond speak?"**
   Look up a policy number, check a claim status, write to a CRM. Their answer is
   the tool-calling demo, and tool calling is what nothing else on-device has.
6. **"Who else has to say yes?"**

Then: *"Let me show you the thing that's hard to believe. Then I'll tell you
what it doesn't do."*

---

## The demo, in this order

**1. Network off, in frame.** Airplane mode, or pull the cable. Do this first and
do it visibly. Everything after it is read differently.

**2. A Hindi turn, end to end.** Speak, get an answer, hear it. Say the latency
out loud before they notice it — *"about 1.8 seconds to first audio; that's
warm, and it is not sub-second."* Naming it first turns a weakness into
accuracy.

**3. A tool call.** Whatever they answered in question 5, as close as you can
get. **This is the moment that separates the product from Sarvam Edge**, and it
is worth more than the speech quality.

**4. Memory.** Reference something from earlier in the session, then show that it
is encrypted at rest and wipeable.

**5. `voice-doctor`, live.** It exits 0. Then say what it does when it doesn't:
it failed this build until the last CC-BY-NC, GPL and LGPL dependency was gone.
For a compliance buyer this is often the most persuasive thirty seconds in the
call — a build gate is a control, and controls are the language they think in.

**6. The spec sheet, and the ceilings.** Hand over `eval_out/SPECSHEET.md`. Point
at the human-recording row: *"a flawless recording by a person scores 90.2% on
our own scorer, so that's the ceiling every other number is read against — we
publish it so you know what our numbers mean."*

Then stop and go to limits. **Do not run the demo long.** A short demo with an
honest limitations list beats a long one every time.

---

## The eight objections

### 1. "Our vendor already gives us India data residency." — *expect this most*

The one to answer well; `plan.md` §11 rates it High and it is the thesis's real
risk.

> "Residency answers where the data is stored. It doesn't answer who can access
> it, which subpoenas reach it, or what happens on a cross-border support
> escalation — the data is still in someone else's process, on their operator's
> credentials. For most workloads that genuinely is enough, and I'm not going to
> pretend otherwise. What I'd ask is whether there's a category where it isn't —
> where the answer needs to be 'it never left our building,' not 'it was stored
> in Mumbai.' If there isn't one, you don't need me."

Then stop talking. Offering the exit is what makes the distinction credible, and
the ones who have such a category will name it in the silence.

### 2. "Is it as good as ElevenLabs?"

> "On naturalness I don't know, and I'm not going to guess — we haven't measured
> prosody and I'd rather tell you than estimate. On intelligibility, we're inside
> the band of the six working cloud systems on held-out code-mixed sentences from
> a benchmark with 1,900 native raters. Our own metric puts us above ElevenLabs
> and I won't quote that, because we calibrated it against those raters and it
> can't rank inside that band — it's an alarm for broken speech, not a ranking.
> So: cloud-competitive intelligibility, unmeasured naturalness, running with the
> network off."

Refusing to use a number in our own favour, and explaining why, buys more
credibility than the number would. **Do not soften this into a claim.**

### 3. "Why not Sarvam? They're Indian, funded, and they have an edge model."

> "For rendered speech, use Sarvam — Bulbul is strong, and on Hindi code-mixed
> it's second in the arena, above ElevenLabs. I'd tell you that on a sales call.
> But Bulbul is API-only with no published on-prem option, and Sarvam Edge is
> speech I/O only — 74M parameters, no LLM, no tool calling, no memory, and no
> announced availability date. If you want speech in a box that works offline,
> wait for them. If you want an agent that reasons and calls your systems
> offline, that isn't what they're building."

Praising the competitor accurately is what makes the distinction land. Never
disparage.

### 4. "Ten Indic languages vs your one."

> "Correct, and it's a real limitation. We're Hindi and English; other Indic
> scripts raise an error rather than guessing at them, which I'd argue is the
> right failure mode. Extending needs a checkpoint that speaks the language — it
> is roadmap, not a config flag. If you need Tamil and Telugu on day one, I'm not
> your answer today. If Hindi and English cover the volume, the agent layer is
> the part nobody else has."

### 5. "Who else is using this?"

> "Nobody yet — you'd be first, and I'd rather say that than imply otherwise.
> What I have instead of references is measurements you can reproduce and a
> licence audit your legal team can run themselves. The pilot is priced so that
> being first is not an expensive bet."

Never invent or imply a deployment. There are none.

### 6. "It's one person. What happens if you disappear?"

Legitimate, and the honest answer is strong:

> "Fair. Two things that are structural rather than promises: every dependency is
> permissively licensed, so there's no component you'd lose the right to run; and
> the deployment is on your hardware, so nothing switches off if I stop invoicing
> you. Worst case you have a working system and a source escrow conversation, not
> a service that goes dark. That's a materially better failure mode than a cloud
> vendor's."

### 7. "Can it run on Windows / Linux / a server?"

Apple Silicon today; the stack is MLX. Say so plainly, and take it as a genuine
requirement to note rather than a thing to hand-wave. Do not promise a port on a
call.

### 8. "How much?"

See `PRICING.md`. Never quote per-minute — the moment the frame becomes minutes,
the comparison is ₹2.70 and the conversation is lost. The comparison is the cost
and risk of the compliance path they would otherwise take.

---

## Closing

One ask, and make it small and specific:

- **Best:** "Can I install it on one machine at your site and let your team try
  it for two weeks?" A scoped paid pilot is the goal, and doing the install by
  hand is deliberate — it is how the deployment workflow gets learned.
- **Good:** a second call with the person from question 6.
- **Acceptable:** "Send me the spec sheet and the licence audit output."
- **Also a win:** "This isn't a problem we have." Take it and move on. The wedge
  depends on finding the few for whom residency is not enough, and every fast no
  makes the next call better targeted.

Never leave without either a next date or a clear no.

---

## After

- Same-day note: what they said, in their words, and which of the eight
  objections came up. **The pattern across ten calls is the actual output of
  Phase 3** — more than any single deal.
- If a limitation lost the deal, write it down. If the same one loses three, it
  has stopped being a limitation and become the roadmap.
- Then send the one-pager. Not before.
