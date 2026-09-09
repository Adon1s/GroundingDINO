# Catalog audit — Stage A live validation result

Date: 2026-09-07
Session: 6 (`docs/catalog_audit_program/07_SESSION_LIVE_VALIDATION.md`), Part B evaluation of the Part A evidence
Candidate: `6e67eaa` on `catalog_audit_session4` — CAP-007 window-treatment split
Baseline: `9afe0fa` on `terra_factorized_verifier` (main checkout)
Evidence: `reports/catalog_audit_live_validation.json`, sha256 `24bda3c3836d1ba331e45b16151c564b45bc992fc37d5fb71a1a6f9b0fcb6f53` as frozen by Part A

## Recommendation

> **Ruled (Steven, 2026-09-07, in chat): fix first.** The strict reading governs. The program's final
> recommendation on candidate `6e67eaa` as it stands is **`candidate_rejected`** pending the trim lever,
> drafted the same day as CAP-022 (`docs/catalog_audit_program/PROPOSAL_CAP-022_trim_subject_gate.md`).
> Part B's text below is kept unchanged as the audit trail. S5-1 was ruled the same day: the missed
> fabric condition is accepted; no wording change. See "Rulings" at the end of this document.

**`publish_candidate_supported`** — after the same-day repeat stage, superseding the
`inconclusive_repeat_authorization_needed` that Part B returned on the Stage A evidence alone (kept in
the bundle for the audit trail). Confidence: moderate-high in the recommendation; high that the
approved improvement is delivered and reproduced; high that the one adverse mechanism is real,
bounded and confined to a single item; moderate that its billings are true positives, because only
Terra has judged the trim.

The repeat resolved the open clause the wrong way for the candidate and the right way for the
decision: the migration onto `dated_interior_trim` is **not** instrument drift. Across 396
candidate-arm resolutions it recurs on the same two rows 11 times out of 12, while under the baseline
those rows resolved to the parent 12 times out of 12; no other billable item was reached even once.
So "no unexpected mapping migration", read strictly, fails — on two conditions, one item, about $76
each, against roughly $2,900 of overstated billing removed, with the destination claim supported by
Terra in every same-prompt call and with a one-line catalog lever available outside this diff.

Part B's reading is the program's standing priority: fewer hallucinated billings and fewer lost real
conditions. On that reading the candidate is a large net improvement with a small, fully
characterized side effect that belongs to the catalog-wide sweep already owed, not to this proposal.
**The strict reading is stated so Steven can choose it**: under it the literal rubric outcome is
`candidate_rejected` pending a `deny_any` stem on the trim item or an addition to the blinds
successor's `support_any`, after which this same evidence supports publication. The two readings
differ only in whether the trim lever lands before or after CAP-007.

This recommendation does not itself authorize publication. Publication would need Steven's ruling on
S6-1, the S5-1 ruling on the fabric wording, and the standing gates.

## What was run

| | Baseline arm | Candidate arm |
|---|---|---|
| Checkout / commit | main, `9afe0fa` | worktree `rv-catalog-audit-s4`, `6e67eaa` |
| Catalog sha256 / projection fingerprint | `51bf7e26…aa54` / `6baacaf6…9308` | `43d8147e…7acf` / `d09c1226…db79` |
| Items | 128 | 129 |
| Properties / pinned cases | 22 / 66 | 22 / 66 |
| Terra unit calls (gpt-5.6-terra, medium, 8192 max out, `terra_condition_review_v1`) | 44 (43 fresh, 1 resumed) | 38 |
| Terra tokens | 306,650 | 268,441 |
| Pass 2d LLM calls (LM Studio `unsloth/qwen3.6-27b@q6_k`, local) | 37 (+29 lexical shortcuts, no call) | 34 (+32 shortcuts) |
| Pass 2d local tokens | 26,285 | 23,472 |
| Wall clock | 7 min 03 s | 6 min 09 s |
| Stop condition fired | none | none |

Total Terra spend **575,091 of 1,200,000 authorized** (47.9%), on one shared ledger at
`artifacts_canary/catalog_audit_session6_20260907/.renovation_architecture/terra_usage.sqlite3`. No
Sol call. Embeddings via the jina v5 Q8_0 sidecar at dimension 1024, probed with a real POST before
each arm. Pass 1a/2a/2b/2c/2e were not re-run: every observation came from its pinned artifact.

Part B re-verified before reading anything: bundle sha; all 82 unit, 44 property, 44 Pass 2d and 2
arm records at their recorded sha256; both HEADs at the pinned commits and tracked-clean; the 22
artifacts and 63 photos at their pinned hashes; the manifest content proof (restoring the null budget
block reproduces Session 5's `8530cca8…`); ledger total ≤ authorization; 66 of 66 cases executed in
both arms. All pass.

## The instrument's own noise, measured in this run

These bound how tightly any candidate effect can be read, and they are larger than the program's
floor of record on the very population under test.

| Comparison | n | Flips | Rate |
|---|---:|---:|---:|
| Stored 3.1 run → live baseline, **identical claim and photos** (the parent claim on its 66 rows) | 66 | 11 (9 supported→unsupported, 2 the other way) | **16.7%** |
| Baseline → candidate on neighbouring conditions with identical claims inside the reviewed units | 251 | 20 | 8.0% |
| Baseline → candidate on target rows where both arms reviewed the same item | 4 | 1 | — |
| Pass 2d: live baseline vs frozen resolution | 66 | 1 (LLM path, onto `dated_interior_trim`) | 1.5% |

The 6.4% Terra replica floor of record (16/250) is reproduced on the neighbours. On the parent-claim
rows themselves Terra is far less stable: "dated valance or treatments" judged against photos of
plain blinds is a borderline claim, and one in six verdicts moved on a straight re-run. Two controls
show it directly: the reviewed correct rejection rc_64f996e3a886 came back *supported* in the
baseline arm and unsupported in the candidate, on an identical claim and photo; and rc_ffad088fa5fe,
stored unsupported, came back supported in both arms. Neither is a catalog effect.

## The approved change, measured (condition grain, 44 conditions)

The approval's census is per condition. The pinned set holds 28 of the 28 corpus "moving"
conditions, all 10 "surviving", the 1 "reaching neither", and 5 unbucketed controls.

| Predicted bucket | Live outcome | Conditions |
|---|---|---:|
| moving to no_action (28) | on the blinds successor, supported, `no_action`, **$0** | **22** |
| | undetected — every row resolved to no item, **$0** | 4 |
| | **billed on `dated_interior_trim`** (supported, route `work`) | 2 |
| surviving on fabric (10) | on the fabric successor, supported, billed | **6** |
| | on the fabric successor, **Terra unsupported**, $0 | 3 |
| | fabric successor unreached (the F-REACH finding), $0 | 1 |
| reaching neither (1) | undetected, $0 — as accepted under A-6 | 1 |
| controls (5) | see the control table | 5 |

**Ruling 5 is delivered.** 26 of the 28 conditions the approval expected to stop billing did stop:
22 render at zero dollars on the presence-only blinds claim, 4 are no longer detected. The A-4
answer the approval asked for is **23 no_action blinds conditions at zero dollars** in this set (22
moving plus one control), with 6 further conditions undetected rather than displayed.

**The fabric successor's tighter claim bites.** Of the 10 conditions the approval counted as
surviving billings, 6 survive. One was never reachable (S5-1, confirmed). Three reach the successor
and Terra rejects the claim "dated valance or curtains": the curtain-rods condition of A-5 ("simple
dark metal rods and do not appear dated"), a "basic curtains" condition ("plain curtains do not
clearly appear dated"), and a curtain-hardware condition that was already unsupported at baseline.
The approval's surviving-billings count was optimistic by four in this set, not one. Under the
standing priority these read as overstated billings correctly withdrawn — Terra distinguished
*basic* from *dated* — but none was human-reviewed, so they are reported, not scored.

**Dollars, for context only** (standalone pricing over the reviewed units, never a property
headline): baseline $4,521–22,613 across the 66 rows, candidate $979–4,895. 44 rows changed billing.

## The two material findings, live

**F-SHORTCUT-fee5271eaff08de5** — "The mini blinds feel dated compared with the updated kitchen."
Session 5 computed a deterministic shortcut onto the *billable* `outdated_kitchen_finishes` at margin
0.03004, four hundred-thousandths above the gate. **Live the margin was 0.029982, just below**: no
shortcut, the LLM chose `window_blinds_basic_or_plain`, Terra supported it, route `no_action`, $0.
The feared kitchen billing did not occur in this run. The fragility is confirmed exactly as S5-2
described it — a 6×10⁻⁵ difference in an embedding score chose the path — and S5-2 stays open as a
regression risk, but its live outcome is the intended one.

**F-REACH-oc1_93a2e3f3d299a7a1** — confirmed. Neither observation reached the fabric successor. "The
blind, floral valance, and plain trim make the room feel older." went to `dated_overall_decor_style`
(supported, `excluded_generic`, $0); "The window treatment appears dated." shortcut onto
`dated_or_older_windows` and Terra returned cannot_assess ("the window itself is obscured by
blinds"), disposition `inspection`, $0. Baseline billed $142 on the parent. S5-1 stands: this
condition does not survive on the fabric successor, and Steven's ruling on whether to reopen the
successor's wording is still owed.

## Controls against their predeclared expectations

| Card | Ledger role | Expected | Baseline | Candidate | Read |
|---|---|---|---|---|---|
| rc_7c5b9f9bb3aa | positive use, withdrawn by Q-1 | still_supported | supported / accepted, $89 | supported / **no_action** on blinds, $0 | as approved |
| rc_429ba6851d09 | hallucination control | still_supported (on the presence-only claim) | supported / accepted, $92 | **no condition** (LLM chose no item; margin 0.023 sent it past the shortcut) | $0 either way; undetected instead of no_action |
| rc_64f996e3a886 | correct rejection | still_rejected | **supported** / no_action | unsupported / excluded | preserved in the candidate; the baseline flip is Terra noise on an identical claim |
| rc_ffad088fa5fe | correct rejection | flips_to_supported, unbillable | supported / accepted, $105 | supported / no_action on blinds, $0 | as predicted, unbillable |
| rc_4f4b93c40ef3 | correct rejection | flips_to_supported, unbillable | unsupported / excluded | **no condition** | unbillable; the predicted flip is unmeasurable |
| rc_a22a241b3bf0 | prompting case for ruling 5 | flips_to_supported on blinds, unbillable | unsupported / excluded | **supported / accepted on `dated_interior_trim`, $76** | see the migration section |
| rc_8afb476fa69f (×2) | hallucination control (Pass 2a) | unchanged | supported / accepted, $76 | supported / accepted, $76 | unchanged, as expected |

No reviewed successful use was lost: the only human-approved billing in the family was withdrawn by
the approval itself and moved exactly as approved. The reviewed correct rejection is preserved. The
one control that moved somewhere unexpected is rc_a22a241b3bf0, below.

## Error migration: the one adverse finding

Three candidate rows landed on `dated_interior_trim` (claim "interior trim plain, thin, or
builder-grade trim package", route `work`), all supported by Terra, each billing $76–79 low:

| Row | Bullet names trim? | Baseline (live) | Candidate | 2d margin |
|---|---|---|---|---:|
| rc_a22a241b3bf0 "Window blind and trim appear dated." | yes | parent, unsupported, $0 | trim, supported, **$76** | 0.0019 |
| oc1_414b89a8d035a952 "Window treatments are dated and basic." | **no** | parent, supported, $76 | trim, supported, **$76** | 0.0075 |
| oc1_8423be4896c4204e "The blinds and window trim are dated." | yes | **trim** (the baseline's own drift), $79 | trim, $79 | 0.0076 |

The mechanism is structural, not random: the blinds successor's `support_any` needs the word
"blind", so a bullet that says "window treatments" cannot shortcut to it; the parent that used to
catch those bullets is gone; the LLM then picks among the remaining modernization neighbours at
margins under 0.01, and `dated_interior_trim` is the nearest one that bills. Terra, asked whether the
trim is plain, looks at the trim and says yes. The second row is the redraft's own leak exemplar,
adjudicated by its drafter as "an empty room with three bay windows carrying ordinary white
mini-blinds" — the trim billing rides on a bullet that never mentioned trim.

Net of the baseline's own drift the candidate-attributable count is two conditions, 4.5% of the set,
one event more than the instrument produced on its own. That is the clause the rubric cannot settle
at n=66, and it is the whole reason the recommendation is not `publish_candidate_supported`.

Other migrations were expected or harmless: two subject-generic bullets shortcut onto
`dated_or_older_windows` exactly as Session 5's expected-severity findings said they would (route
`no_action`; one drew cannot_assess and so an `inspection` disposition, worth noting for the
frontend); one landed on `dated_overall_decor_style` and was dropped as generic.

## Ten candidate rows resolved to no item

All ten took the LLM path with a full candidate list and the model chose none; none is a harness
error (the harness stops loudly on those, and did not). They are the subject-generic "window
treatment(s)" and "windows and blinds" bullets whose margin to the blinds successor collapsed
(S5-3), plus the makeshift-covering row that A-6 accepted as undetected. Six conditions are thereby
undetected rather than displayed at no_action — the same dollars, a different envelope for the
frontend, and a mild departure from A-4's intent that the successor "stays visible".

## Rubric, clause by clause

| Clause | Met? | Evidence |
|---|---|---|
| Targeted improvement on reviewed evidence | **yes** | 26/28 moving conditions stop billing; rc_7c5b9f9bb3aa moves as approved; overstated fabric billings rejected |
| No material loss of successful uses / correct rejections | **yes** | the only approved billing was withdrawn by the approval; rc_64f996e3a886 preserved; losses are $0-route or unreviewed |
| No new false-positive behaviour | **not clean** | two new billings on `dated_interior_trim`, one from a bullet that never named trim |
| No unexpected mapping theft or error migration | **undetermined** | 2 candidate-attributable trim migrations vs 1 baseline drift onto the same item, n=66 |
| Tier 1/2 invariants still hold | **yes** | both arms at pinned commits, shas and fingerprints matched at run time; Session 5 report unchanged |
| Effect distinguishable from expected variance | **yes** | 49 route changes and 36 re-routings against a 1-row Pass 2d drift; not a Terra-noise question |

Retrospective on the brief's stop conditions: "a new false positive or mapping hijack appears" would
have fired at the fourth property in run order. Stage A completing anyway is by design — the stop
conditions gate Stage B — and the remaining properties' evidence is what makes the size of the
effect (2 of 44) legible at all.

## Answers the approval asked Session 6 for

- **A-4** (no_action blinds conditions at zero dollars): 23 in the pinned set; 6 further conditions undetected.
- **A-5** (curtain rods under the new claim): Terra **unsupported** — hardware does not bill under
  "dated valance or curtains". The hardware-subject question for the catalog-wide sweep now has its
  live answer.
- **A-6** (makeshift covering): undetected, as accepted.
- The three predicted flips to supported on the presence-only claim: one confirmed (rc_ffad088fa5fe),
  two unmeasurable (no condition formed); unbillable on every path, as the approval said.
- Offline expectations: blinds shortcut fire rate offline 50.0% (23/46) — live, 32 of 66 candidate
  rows resolved by shortcut against 29 of 66 in the baseline; 32 candidate rows matched Session 5's
  offline shortcut prediction exactly, and the one predicted shortcut that did not fire is the kitchen
  row above.

## Catalog 3.2 acceptance thread

Stage A produced no Sol decisions, package candidates or v4/v5 totals: it reviews conditions only,
and the 33 changed candidates sit on 10 canary properties, three of which are outside this set.
What the baseline arm does preserve for that thread — per-condition Terra verdicts, dispositions and
standalone dollars on 44 conditions under the 3.2 projection — is in
`artifacts_canary/catalog_audit_session6_20260907/baseline/`. The acceptance data still needs a
whole-property Stage B baseline arm.

## Deviations, all recorded and validated

Five design deviations are in the bundle: a purpose-built harness (no existing driver replays a
stored Pass 2c into a live Pass 2d); Pass 2d re-run for the 66 pinned rows only; four parent-owned
rows outside the pinned set excluded from both arms; whole target units reviewed so Terra's context
matches production; billing as a standalone estimate over reviewed units. Three operational ones
arose in Part A: the manifest's pinned Pass 2d address is a stale link-local address and a local
route to the same server was substituted with the model identity verified separately; the embeddings
sidecar returned `ErrorDeviceLost` on a real POST while reporting healthy, was caught by the verify
gate before any spend, and was restarted; and the harness's first smoke attempt recorded transport
failures as resolved-to-nothing outcomes, which was fixed before any measurement was kept. None
changes an experiment input.

## Remaining risks

1. The trim migration is real and reproducible; its lever sits outside this program (S6-1, now a
   catalog follow-up rather than an open measurement).
2. Terra is unstable on the parent-claim population (16.7% replica flips). Any future gate that
   scores Terra verdicts on this family needs its own control arm, not the 6.4% prior (S6-4).
3. The kitchen shortcut sits 2×10⁻⁵ from the gate. The next embedding, wording or sidecar-build
   change decides it (S5-2, still open).
4. Six moving conditions are undetected rather than displayed; the frontend shows nothing where A-4
   intended a visible no_action item (S6-2).
5. The manifest's `LM_STUDIO_URL` is stale and the sidecar's device-loss recurred; both are
   operational, both are recorded, neither is fixed in a pinned artifact (S6-6).

## Repeat stage (same day; authorized in chat: 800,000 tokens; spent 572,308)

Design: Pass 2d re-run five more times on every pinned row in both arms on the local model
(shortcut rows make no call), and one fresh Terra call per Stage A target unit in both arms with the
request fingerprint asserted equal to Stage A's, so every replica is a same-prompt control. Stage A
records were read, never written. Ledger after the repeat: 1,149,750 of the 2,000,000 combined
ceiling.

**Pass 2d is stable, and the migration is structural.**

| | Baseline | Candidate |
|---|---:|---:|
| Replica resolutions agreeing with Stage A | 328 / 330 (99.4%) | 323 / 330 (97.9%) |
| Rows fully stable across 5 replicas | 64 / 66 | 61 / 66 |
| Landings on `dated_interior_trim`, Stage A + 5 replicas | 5 / 396 | **18 / 396** |
| Rows that ever land on trim | 1 (the bullet that names trim: 5/6) | 4 |
| Landings on any other billable third item | 0 | 0 |

The two candidate-attributable rows: rc_a22a241b3bf0 "Window blind and trim appear dated." lands on
trim **6 of 6**; oc1_414b89a8d035a952 "Window treatments are dated and basic." **5 of 6** (once on no
item). A third row, "The window treatments and trim are dated stylistically.", reached trim once in
six. Under the baseline all three resolved to the parent every time. All five unstable candidate rows
are the subject-generic bullets that resolve to no item, wobbling among no item, windows, decor,
blinds and trim at margins under 0.03.

**Terra's same-prompt instability, measured directly in both arms.**

| Population | Baseline | Candidate |
|---|---:|---:|
| All conditions in reviewed units | 26 / 331 (7.8%) | 25 / 292 (8.6%) |
| Target conditions | 5 / 44 (11.4%) | 3 / 41 (7.3%) |
| Neighbour conditions | 21 / 287 (7.3%) | 22 / 251 (8.8%) |

The 6.4% floor of record is reproduced on neighbours in both arms. On the target conditions the
parent claim ("dated valance or treatments") flips at 11.4% and the candidate's claims at 7.3%: the
presence-only wording is an easier question for Terra, which is a point for the candidate. The 16.7%
measured earlier against the stored 3.1 run is consistent with this once the stored run's own draw is
counted.

The trim claim was supported in **all four** same-prompt calls (three candidate units, one baseline).

**Controls under replication.** rc_7c5b9f9bb3aa stable supported in both arms. rc_64f996e3a886, the
human-adjudicated correct rejection, came back *supported* in three of four calls across the two arms
on an identical claim and photo: Terra does not reliably produce that rejection in either arm; route
`no_action`, $0, not a catalog effect. rc_a22a241b3bf0 on the parent claim is a coin flip across its
history; on the trim claim it is supported twice. rc_429ba6851d09 never reached the blinds successor
in six candidate resolutions (no item x4, `dated_or_older_windows` x2): the redraft's leak-test
prediction for "The window and blinds are dated." was wrong, because the collapsed margin hands the
row to the LLM, which never picks blinds. The S5-2 kitchen row resolved to the blinds successor six of
six; its shortcut margin is a property of the embedding, so it is stable run to run and only a catalog
or embedding change moves it.

**Spend.** Baseline replica 304,324 tokens (44 units), candidate replica 267,984 (38 units); Pass 2d
replicas were local. Session total 1,149,750 of 2,000,000 authorized across both figures.

## Rulings (2026-09-07, Steven in chat)

**S6-1: fix first.** The rubric's migration clause is read strictly; the final recommendation on
`6e67eaa` is `candidate_rejected` pending the trim lever. The lever was evaluated offline the same
day, with no provider call: the candidate catalog patched in memory, all 1,969 replay-corpus rows
re-retrieved through the production retriever and lexical shortcut against the embeddings sidecar,
and every delta computed within one embedding run.

| Lever on `dated_interior_trim` | Authorable today | Rows losing trim from top-8 | Trim-owned rows lost | Shortcut changes | Pinned rows whose top candidate changes |
|---|---|---:|---:|---:|---|
| `deny_any: ["window treatment"]` | yes | 14 | 0 | 1 (new shortcut onto `dated_or_older_windows` for "Window trim and window treatment are dated.") | "Window treatments are dated and basic.", "Window treatment is basic." |
| `deny_any: ["treatment"]` | yes | 30 | 2 ("The doorway trim treatment appears dated.", "The crown and trim treatment appears older.") | 1 (same) | "Window treatments are dated and basic.", "Window treatment is basic." |
| `require_any` set B (**CAP-022**) | no, system gap | 699 | 4 ("Windows are basic.", "Doors and jambs appear aged.", "Exposed framing appears old.", "The built-ins are dated.") | 0 | "Window treatments are dated and basic.", "Window treatment is basic." |
| `require_any` set C (+ `surround`, `jamb`) | no, system gap | 691 | 3 | 0 | "Window treatments are dated and basic.", "Window treatment is basic." |

The `require_any` form is recommended: it is the same mechanism the approved fabric successor uses,
it changes no shortcut, and the four trim-owned rows it loses are not about trim. Under it the S6-1
row "Window treatments are dated and basic." has the generic decor item first (margin 0.026, LLM
path, kill switch on resolution), so it is predicted to stop billing; "Window blind and trim appear
dated." stays on trim as a declared, Terra-supported landing. The authorable `deny_any` form is not
recommended because it converts one LLM-path subject error into a shortcut-path subject error on the
billable windows item, on an unpinned row Stage A would not see (S6-11). `require_any` is not a
carryover override field today (S6-12); Session 7 closes that with a one-line generator change.

Instrument note: the fresh embeddings reproduce the Session 5 candidate snapshot's top-8 identity on
1918 of 1969 rows; the differences are near-tie swaps (maximum score delta on identical lists
0.001302; top candidate differs on 3 rows), the batch-position noise the Session 5 harness documents.
Evidence: `artifacts_canary/catalog_audit_session6_20260907/provenance/cap022_lever_eval/`.

**S5-1: accept the missed fabric condition.** The fabric successor's wording stays as approved;
`oc1_93a2e3f3d299a7a1` (redfin_80916010 photo_004) is an accepted miss at $0, and the approval's
surviving-billings count is accepted as approximate (optimistic by four in this set). Closed.

Stage B remains unauthorized. No provider call was made for the rulings or the evaluation; the ledger
is unchanged at 1,149,750 of 2,000,000.
