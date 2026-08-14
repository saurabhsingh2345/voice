# First contact

Templates for the three wedges in `plan.md` §7, plus the two letters that are
not sales. Every number here traces to `CLAIMS.md`.

**Rules that apply to all of them.** Short enough to read on a phone. One ask,
and the ask is fifteen minutes, never a deck. Nothing that requires clicking to
understand. No attachment on the first mail — attachments trigger enterprise
filters and signal a mass send; the one-pager goes out *after* a reply. Send from
a personal address, not a no-reply. Send Tuesday–Thursday morning IST.

Anything in `[brackets]` must be replaced with something specific to the
recipient. If you cannot fill a bracket, you have not researched them enough to
send it — pick a different name from the list.

---

## Wedge 1 ⭐ — Regulated enterprise (BFSI, insurance, healthcare)

Buyer: whoever owns the constraint, not whoever owns voice. CISO, Head of Data
Privacy, DPO, Head of Compliance, sometimes the CTO. The person with the May
2027 date on a slide.

**Subject:** Voice AI that never leaves your perimeter — 15 minutes?

> Hi [Name],
>
> Under the DPDP Rules you have until 13 May 2027 for full compliance, and voice
> recordings sit close enough to biometric data that the consent bar is the
> strict one. If [Company] is running or planning voice — [collections calls /
> claims intake / patient triage / branch support] — the audio is almost
> certainly going to a cloud API today.
>
> I've built a complete Hindi and English voice agent that runs entirely on your
> own hardware. Speech in, reasoning with tool calling, speech out, encrypted
> memory. Not a speech box — the whole agent. No audio leaves the machine,
> because there is nowhere for it to go.
>
> Three things I can show you with measurements rather than adjectives:
>
> - It runs on an 18 GB Mac, not a GPU cluster. Hindi synthesis at RTF 0.68,
>   1.8 s to first audio.
> - Every dependency is Apache, MIT or BSD — enforced by an audit that fails the
>   build, zero exceptions. Your legal team can have the output.
> - On AI4Bharat's SpeechArenaBench (120,000 comparisons, 1,900 native raters),
>   it lands inside the intelligibility band of the six working cloud systems on
>   held-out code-mixed sentences.
>
> I'd also tell you what it does not do yet — naturalness is unmeasured and turn
> latency is 1.8 s, not sub-second — before you spend any time on it.
>
> Fifteen minutes? I can show it running with the network physically off.
>
> Saurabh Singh
> saurabh.singh@enfec.com

*Why it is shaped this way:* the deadline is their problem before it is our
pitch, the disqualifier in paragraph five is what makes the three claims above
it credible, and "nowhere for the data to go" is the architectural point that
data residency cannot match.

---

## Wedge 2 — OEM / ISV embedding

Buyer: VP Engineering or Head of Product at device makers — kiosks, POS,
in-vehicle, medical devices, rural-connectivity products. They already know they
need offline; they have been told it is not available.

**Subject:** Offline Hindi voice for [product] — working today, MIT/Apache only

> Hi [Name],
>
> [Specific observation: your kiosks run where connectivity is unreliable / your
> device ships into rural deployments / your in-vehicle unit can't depend on a
> data link.] If you have looked at Indic voice for it, you have probably found
> that everything with an agent layer is a cloud API, and everything on-device is
> speech-only.
>
> I have a working stack in the gap: Hindi and English speech in and out, plus an
> LLM with tool calling and persistent memory, entirely local. Running today, not
> announced.
>
> The details that decide whether it fits your BOM:
>
> - 18 GB Apple Silicon today; the memory envelope is published per component
>   (1.33 GiB TTS, 2.31 GiB STT, 2.16 GiB LLM).
> - RTF 0.68 for Hindi synthesis, 1.8 s to first audio.
> - Apache/MIT/BSD throughout, zero exceptions, audited by a build gate — so
>   there is nothing here that complicates redistribution in your product.
> - Hindi and English only right now. More Indic languages need a different
>   checkpoint; I would rather say that up front than after an evaluation.
>
> Licensing would be per device or per deployment. Worth fifteen minutes to see
> whether the envelope fits what you ship?
>
> Saurabh Singh
> saurabh.singh@enfec.com

*Why:* an OEM cares about BOM, redistribution rights and memory envelope, in
that order — quality is a threshold, not a ranking. The licence audit is a
stronger argument here than anywhere else, because they ship our dependency tree
inside their product. Naming the Hindi-only limit early avoids a dead evaluation.

---

## Wedge 3 — Privacy-first prosumer

Buyer: doctors, lawyers, journalists, therapists. No IT department, a
confidentiality duty, and no procurement cycle. Fastest route to a first
invoice. This one works through communities and word of mouth, not cold mail —
the template below is for a warm introduction or an association newsletter.

**Subject:** A voice assistant that runs on your laptop and sends nothing anywhere

> Hi [Name],
>
> You dictate [case notes / patient histories / interviews] into something. If
> that something is a cloud service, [privileged client material / patient
> records / a source's identity] is sitting on someone else's servers under
> whatever their retention policy says today.
>
> I've built a voice assistant that runs entirely on your Mac. It listens,
> understands Hindi and English, answers, remembers across sessions — all
> offline. You can unplug the network and it keeps working. The memory is
> encrypted with a key in your Keychain and you can wipe it.
>
> Honest about the state: it's a working tool I use daily, not polished
> consumer software, and it needs an 18 GB Mac. If that's fine, I'd like a few
> early users at [₹2,000–5,000 one-time / ₹500 a month] who will tell me what
> breaks.
>
> Interested in a look?
>
> Saurabh Singh

*Why:* the threat model is concrete and personal, "unplug the network" is the
whole demo in four words, and "not polished consumer software" sets the
expectation that makes an early user forgiving. Charge from the first one —
never free.

---

## The AI4Bharat letter — not sales

Their benchmark redirected this entire project. They are a supplier, a
credibility source, and the people best placed to tell us where the AUC finding
is wrong. Send it because it is true, not to open a door.

**Subject:** SpeechArenaBench changed what we were building — and a calibration result

> Hi [Name / AI4Bharat team],
>
> I want to tell you that SpeechArenaBench changed the direction of a project,
> and share a result you may find useful.
>
> I was fine-tuning IndicF5 for Hindi and code-mixed speech. Your benchmark told
> me, with 120,000 comparisons and 1,900 raters, that the wedge I was aiming at
> did not exist — rankings were stable across code-mixed input, so "we win on
> Hinglish" was already tested and closed. I stopped, and moved the work to
> where nobody is: a fully offline Indic voice *agent* for deployments where
> data cannot leave the premises. That redirection is worth more than the
> fine-tune would have been, so: thank you.
>
> Two things from working with the data that might be worth your time:
>
> **1. The six rating axes are documented as 1–5 scales but only ever take the
> values 1 and 5.** Across 654 clips, no intermediate value occurs. So a
> per-system "mean of 4.48" is an 87% good-rate in disguise, and treating
> differences between such means as quality gradations — or running Pearson on
> them — measures a unit that is not there. Worth a note in the dataset card;
> it would have produced confident, wrong numbers for us in silence.
>
> **2. Your released ratings make the benchmark usable as a calibration set for
> other people's metrics, which may be an under-advertised use.** We ran our own
> round-trip intelligibility scorer over 654 of your rated clips and measured it
> against the human verdicts by AUC. It scores 0.671 over all seven systems and
> **0.625 with IndicF5 removed**, against 0.500 for chance. In other words it
> detects broken speech and is close to guessing among systems that work — its
> ordering actually inverts inside that band. We had trusted that metric for
> months. Your data is what falsified it, and I don't know of another public
> dataset in Indic speech that could have.
>
> Also, since Table 4 pools all ten languages: fitting Hindi-only code-mixed
> Bradley-Terry on your votes puts Bulbul V3 Beta second, above ElevenLabs and
> Sonic, and IndicF5 at 714.7 — a hundred points below its own pooled score.
> Gemini lands within 4 points of its published pooled figure, which is how we
> checked the pipeline. Happy to send the code; it reads your parquet over HTTP
> range requests rather than materialising 32.5 GiB.
>
> No ask. If a per-language table is something you would publish, I would rather
> cite yours than mine.
>
> Saurabh Singh
> saurabh.singh@enfec.com

*Why:* it leads with what their work did for us, gives two findings they can act
on, and closes with "no ask." A researcher's inbox is full of people wanting
something. This one is worth answering, and if a relationship comes of it, it
came honestly.

---

## IndiaAI Compute Portal — application note

Not an email; the framing to reuse in the application form. GPU hours at
₹65–92/hour with up to 40% subsidy, and being an accepted applicant is itself a
credibility signal.

**Frame it as:** an Indian-language voice agent for regulated sectors that cannot
send data to a cloud, aligned with DPDP compliance and digital sovereignty.
Emphasise (a) it runs on consumer hardware, so the compute is for *building*,
not serving — modest and specific; (b) permissive licences throughout, so
outputs can be released; (c) measured evidence already exists, and cite the
spec sheet.

**Ask for:** what is needed to train an Indic TTS checkpoint beyond Hindi, since
that is the real roadmap gap and it is a genuine use of subsidised compute.

---

## Follow-ups

One follow-up, once, after five working days. Never more, and never "just
bumping this."

> Hi [Name] — following up once, then I'll stop.
>
> Since I wrote, [one specific new thing: the app is now signed / the offline
> demo is recorded, here it is / a spec sheet regenerated with X].
>
> If voice data leaving your perimeter isn't a problem you own, tell me and I'll
> close the loop. If it's someone else there, I'd appreciate the name.
>
> Saurabh

The "tell me and I'll stop" line gets replies from people who would otherwise
ignore it, and the referral ask is the only thing in this document with a
better-than-even hit rate.

---

## Sequencing — do this in the right order

Send **wedge 1 and wedge 2 first**, in parallel, ten names total. Enterprise
procurement is slow, so the clock starts the day you send, and OEM conversations
qualify or kill themselves fast on the memory envelope.

Send the **AI4Bharat letter this week** regardless of everything else. It has no
sales cycle and it is the only one where the value is not contingent on a reply.

Hold **wedge 3** until the `.app` is signed — a prosumer who cannot get past
Gatekeeper is a lost first impression, and unlike an enterprise pilot there is
no engineer on the call to walk them through it.

**Before any of it: record the offline demo.** Airplane mode in frame. Every
template above earns its reply on a claim that a thirty-second video proves and
a paragraph cannot.
