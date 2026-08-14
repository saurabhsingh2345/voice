# The list of ten

`plan.md` §12 says: find ten people in regulated Indian enterprises or at
Indic-hardware OEMs and ask for fifteen minutes. This is how to build that list.

**The names are not in this file, deliberately.** A target list assembled from
memory would be plausible companies with guessed job titles, and every
`[bracket]` in `EMAILS.md` would then get filled with something invented. That
is the failure mode those brackets exist to prevent. Below is the filter, the
sourcing method and the tracker; spend the two hours and fill it with real
people.

---

## The filter

Score each candidate 0–2 on all four. **Send to 6 and above. Never send to
anything scoring 0 on the first criterion**, however attractive the logo.

### 1. Does data genuinely fail to be allowed out? ⭐

The whole thesis. Not "prefers local" — *cannot*.

- **2** — A named regulator or internal policy forbids it: RBI-regulated data,
  hospital records under a state rule, government-adjacent work, an existing
  no-cloud-for-X policy.
- **1** — DPDP compliance work is under way and voice has not been ruled on.
- **0** — "We like the idea of privacy." This is `plan.md` §11's
  High-likelihood risk. They will take the call and buy residency.

### 2. Do they already run voice at volume?

Existing voice means an existing budget line, a known pain and a person who owns
it. Collections, claims intake, patient triage, branch support, IVR
deflection, field-agent tooling.

- **2** — running voice AI today, in Hindi or code-mixed.
- **1** — human call centre, evaluating automation.
- **0** — no voice anywhere. Educating them is a different, longer business.

### 3. Is Hindi + English enough?

Our sharpest limit. A national footprint needing Tamil and Telugu on day one is
a dead evaluation — see objection 4 in `CALL-GUIDE.md`.

- **2** — Hindi-belt concentrated, or Hindi/English covers the volume.
- **1** — Hindi is the largest slice among several.
- **0** — multi-lingual South Indian footprint is the requirement.

### 4. Can it be sold to one person?

- **2** — the constraint owner (CISO, DPO, Head of Compliance) can authorise a
  two-lakh pilot.
- **1** — needs one more approver.
- **0** — committee, RFP, empanelment. Real, but not a first customer.

---

## Where the names come from

Two hours of actual research, not recall:

- **DPDP and data-localisation conference speaker lists**, 2026. Anyone who spoke
  on voice, biometrics or consent has publicly self-identified as owning this
  constraint. Best single source, and it gives you a specific reference for the
  first line of the email.
- **RBI-regulated entities** — NBFCs, small finance banks, insurers, TPAs — that
  have announced voice automation. Announcement plus regulator is criteria 1 and
  2 at once.
- **Hospital chains and diagnostics networks** with published patient-data
  policies.
- **LinkedIn by title, not company**: "Data Protection Officer" / "Head of
  Information Security" + India + BFSI or healthcare. Title-first finds the
  constraint owner directly, and question 1 in `CALL-GUIDE.md` exists because
  that person is usually not the one running voice.
- **For OEMs:** Indian kiosk, POS, in-vehicle and medical-device makers; anyone
  shipping into rural deployments where connectivity is unreliable. Trade shows
  and tender documents beat search here.
- **Warm paths first.** One introduction outperforms ten cold mails, and the
  prosumer wedge only works this way at all.

**Anti-targets, worth recognising:** the consumer names already signed to cloud
voice vendors — the Meesho / Kuku FM / PocketFM / Cars24 / hoichoi / TVS Motor
tier from ElevenLabs' own India page. They have chosen cloud, they have no
regulator forbidding it, and they score 0 on criterion 1. Media and consumer apps
in general are wedge 4: opportunistic, never strategic.

---

## The tracker

Keep it here, in the repo, next to the evidence it draws on. Ten rows before
sending any of them — batching forces comparison and stops the first plausible
name from consuming the week.

| # | Org | Person + title | Wedge | 1 | 2 | 3 | 4 | Σ | Why them (one line, specific) | Sent | Reply | Outcome |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | | | | | | | | | | | | |
| 2 | | | | | | | | | | | | |
| 3 | | | | | | | | | | | | |
| 4 | | | | | | | | | | | | |
| 5 | | | | | | | | | | | | |
| 6 | | | | | | | | | | | | |
| 7 | | | | | | | | | | | | |
| 8 | | | | | | | | | | | | |
| 9 | | | | | | | | | | | | |
| 10 | | | | | | | | | | | | |

If the "why them" column cannot be filled with something specific and true, the
row is not researched enough to send. That column becomes the first line of the
email.

---

## What ten calls are actually for

Not ten deals. One pilot from ten well-filtered sends would be a good outcome;
zero pilots with ten clear reasons why is also a result, and a cheap one.

The output to protect is **the pattern**: which of the eight objections in
`CALL-GUIDE.md` came up, how many scored 0 on criterion 1, and which limitation
lost which deal. If the same limitation loses three, it is no longer a
limitation — it is the roadmap, and it outranks whatever was planned next.

Ten sends is also the smallest number that can distinguish "the wedge is wrong"
from "the list was wrong." Fewer than ten and a null result tells you nothing.
