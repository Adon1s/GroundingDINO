# Pass 2d — offline shortcut analysis and bounded Phase B proposal (2026-09-09)

Inputs: the v3 record, Steven's twelve photo answers (`pass2d_photo_review_answers_20260909.md`), and two offline enumerations run this session with no model or sidecar calls (`b2_shortcut_analysis.py`, `b2_rule_d.py`, outputs `b2_*.json`). Nothing here is authorized to run; it is a proposal for Steven's go.

## 1. What the twelve reviewed rows established (human grade, observation axis)

| Outcome | Rows |
|---|---|
| Sentence wrong or unsupported by the photo (upstream of Pass 2d) | C2 (window-light artefact), C4 (staining not visible), C1 in part ("old" for a missing section) |
| A listed candidate fit the photo better than the model's pick | C1, C2, C5 |
| Model's pick endorsed | C3, D4 |
| Shortcut fired on a location word and named the wrong subject | C5 (floor → fixtures via "tub"), D1 (ceiling → fixtures via "shower") |
| Decline lost a true, listed condition | C6, C7, C8 |
| Decline correct (control) | D2: swirl plaster, not popcorn; would not cost it |
| Decline correct given the list; true item never offered | D3: scene classification correct (laundry), `cabinets_dated_style` eligible in kitchens only |
| Bears on a preserved disposition, evidence only | C1 → CAP-013 (missing base trim, third property); D3 → D6 (cabinets, no change); C3/C4/D4 → CAP-014; D2 → CAP-016 |

Caveat carried forward: C6-C8 were chosen as the rows a reader thought had a fitting candidate. D2 and D3 are the only "correct decline" rows reviewed, and both held. Terra was `supported` on every reviewed shortcut row including the two wrong-subject ones, confirming it cannot see this failure.

## 2. Offline shortcut-rule enumeration (640 stored shortcut rows)

| Rule | What it requires before the shortcut may fire | Rows flipped to the model | Catches C5 / D1 / row 6 | Reading |
|---|---|---|---|---|
| B (CAP-005's suggestion) | a condition-bearing term among the firing terms | 385 (60%) | yes / yes / yes | Rejected: flips mostly correct "dated X" modernization shortcuts (windows 96, roof 46) |
| A / D (subject prefix or head only) | a firing term inside the sentence subject | 121 / 131 | partial | Rejected: most firing terms are condition words that legitimately sit in the predicate ("Walls have holes") |
| C (fails both A and B) | either a subject-head hit or a condition term | 21 | yes / no / no | Misses D1 |
| **E (proposed)** | **if every firing term is a bare subject noun, at least one must match the sentence head (cut at the first verb or preposition)** | **25 (3.9%)** | **yes / yes / yes** | About 11 of the 25 are genuine wrong-subject firings (floor/ceiling/caulk/cabinets/wall/countertops → tub, shower, sink; built-in → windows); the rest are harmless and would go to the model, which can still pick the same item |

Also enumerated and not proposed: removing the generic gate (377 extra firings, 178 onto the scuffs item, 22 new billings where the model declined). Terra refutation does not separate any rule's flipped set from its kept set (5-6% throughout); the rule is justified by the human answers, not by Terra.

## 3. D3 follow-up: cabinet scene eligibility (offline count)

52 stored observations mention cabinets outside kitchen scenes; 17 declined. Of the declines, four are kitchen cabinets seen from an adjacent living room, three are laundry or basement cabinets (D3 among them); the rest are bathroom vanities that resolved correctly to vanity items. `cabinets_dated_style` and `cabinets_worn_finish` are both eligible in kitchen scenes only. **This is a catalog scene-eligibility limitation with the scene classification itself correct.** D6 (cabinets: no change) is preserved; this is recorded as evidence toward its trigger and no change is proposed here.

## 4. Bounded experiment (one manifest; local Qwen; text held fixed; Terra only where a selection changes)

| Arm | Change under test | Anchor rows (human-answered) | Controls | Success | Failure / stop |
|---|---|---|---|---|---|
| **S** shortcut Rule E | the 25 flipped rows go to the model instead of shortcutting | C5, D1, row 6 must no longer resolve to the fixtures item | the other ~22 flipped rows | anchors move to a floor/ceiling/wall item or null; ≤ 2 of the harmless rows change item | > 2 harmless rows change; or any anchor still lands on fixtures |
| **N** null rule | one added prompt line: when the observation's subject is a specific instance of a candidate's subject (a faucet is a fixture, a range is an appliance), select that candidate | C6, C7, C8 must select the named item; **D2 and D3 must stay null** | 40 declines the triage called correct (≤ 4 may flip) and 40 non-null rows (≤ 2 may change) | 3 of 3 anchors select, both controls stay null, control limits hold | any control limit breached |
| **R** rendering | scene shown; `atomic_claim` subject and state rendered alongside description | C1 (→ bathroom finishes), C2 (→ staining/contamination item or null), C4 (→ null or a moisture item is unreachable, so null) | same 80 controls | ≥ 2 of 3 anchors move as stated, control limits hold | control limits breached |

Common rules. One replica per row (self-agreement 98-99% with text fixed; this design does not vary text, and results are stated as holding with text held fixed). Any changed selection must clear the retrieval noise floor (2 of 1,969 rows change rank 1 between same-catalog snapshots). Every changed selection on a control row goes to Steven if it is one of ≤ 10, otherwise to Terra. Arms run in the order S, N, R; an arm that breaches a control limit is stopped and reported, not tuned.

Cost. Local Qwen: about 25 + 85 + 83 calls. Terra: only new billable landings, worst case ≈ 25 + 40 + 40 units at ~7k tokens ≈ 735k, expected well under 300k. Within Steven's 2.5M authorization; ledgered per arm; nothing spans a day.

Preconditions before any call. Sidecar probe with a real POST; LM Studio URL verified against the served model; `KIND_ONTOLOGY_VERSION=observation_kind_v2`; catalog-sha-keyed cache; the manifest_v2 budget block applied with Steven's authorization recorded.

## 5. Explicitly out of scope

The D4-residual replay (needs its own authorization under the decision record §5). Any catalog edit: cabinet scene eligibility (D6, evidence only), CAP-013 third property (evidence only), CAP-016 (preserved). A Terra-tier model swap (only if S, N and R all fail on their anchors). Pass 2a light-artefact hallucinations (C2, C4): upstream, recorded for the 2a thread. The seventeen remaining null cards.

## 6. The decision this enables

Whether any of the three changes earns a code proposal. Each arm has human-answered anchor rows, a correct-null control that must not move, and control limits that stop it. If none succeeds, the record closes Phase B with the mechanism census and these twelve human answers; if one does, it becomes a bounded code change proposal with its own validation, not a prompt-tuning cycle.
