# Pass 2d diagnosis — Phase A record (v3, consolidated, 2026-09-09)

**This is the single diagnosis.** It supersedes v1 (investigated), v2 (rewritten after verification) and the reconciled v1 JSON. Companion data: `pass2d_diagnosis_20260909.v3.json` (36 rows with final status, condition-level Terra re-score with a replication arm, Workflow 2 surviving cores and fatal corrections, 148 triaged nulls, 129-item divergence scan, budget reconciliation, the eight review cards). Provenance: `pass2d_phaseA_workflow1_result.json`, `wf2_cores.json`, `b0_rescore.json`, `pass2d_corpus_stats_script.py`, `b0_rescore.py`, `build_v3.py`.

Lineage: Workflow 1 `wf_7a3702b3-779` (7 Opus investigators, 1.26M agent tokens) → v1; Workflow 2 `wf_330c2f01-3b7` (36 refuters, 3 lenses each, plus 4 critics, 4.79M agent tokens) refuted all 12 v1 findings as stated → v2; offline correction pass B0 (this session, Fable 5.1) → v3. **Provider, Terra and sidecar calls across the whole of Phase A: zero.** The Session 6 Terra ledger is unchanged since 2026-09-07 17:17 and no newer ledger exists. Catalog `78796401…` and decisions `4cd0bc07…` equal the checkpoint pins. Nothing has been written to the repository; no recorded decision has been reopened.

## 1. The diagnosis in one paragraph

Pass 2d as shipped is a disciplined, near-deterministic head-word matcher with a strict null: it never emits an id outside its rendered list, it declines on a third of the bullets it is asked about, and its behaviour is reproducible from stored candidates with zero mismatches. Its inputs are not: upstream observation text reproduces on 1.5 percent of photos between two runs of the same canary, and the only corpus-scale verifier, Terra, checks the truth of whatever claim Pass 2d selected and is therefore blind to the one failure the worklist alleges, a wrong item among reachable candidates. **No count of Pass 2d selection failures can be asserted at any evidence grade.** Not the worklist's seventeen, not v1's five. What Phase A establishes is a deterministic mechanism census, a set of corrections to the worklist's annotations, a two-row shortlist for human adjudication, and eight photo cards whose answers decide whether any live spend on Pass 2d selection is justified.

## 2. Mechanism census (deterministic; survived hostile verification; no stage verdict attached)

| Id | Fact | Numbers |
|---|---|---|
| M-1 | Shortcut gate order: negation guard, then generic gate (`defaultHidden` or `drop_if_generic`, 27 of 129 items, `tools/scene_classifier_passes.py:510-511` at `:1095`), then score ≥ 0.72, margin ≥ 0.03, then one `support_any` prefix hit | `wall_scuffs_marks_or_dents` is default-hidden: rank-1 on 403 stored rows, shortcut fired on 0; all 12 CAP-014 rows took the LLM path. Negation blocks 0 of 1,969 snapshot rows; the generic gate 454. On a sole-blocker decomposition the generic gate is one of four comparable filters (348 rows) and outcome-neutral on 291 of them |
| M-2 | Kind routing precedes retrieval: exactly one Pass 2c kind, filtered before cosine top-K (`tools/catalog_embeddings.py:486-521`) | The defect-kind items nominated in worklist rows 17/19/20/22 could never enter a degradation bullet's pool. This restates `approvals_v2` CAP-014, which already records two of those rows as Pass 2c kind exclusion |
| M-3 | Guardrails run after top-8 with no backfill (`:520-527`); every reached (scene, kind) cell has ≥ 8 items, so a short list is always a removal | 2,248 of 4,107 stored rows (54.7%) and 479 of 1,969 product-lane rows (24.3%) got fewer than eight; ten of twelve CAP-014 rows got six. Today's catalog: 1,012 removals on 761 rows, 890 `require_any`, 300 at pre-guardrail rank 1-3, 364 from one item. **Whether a lost slot ever changed a selection is not established** |
| M-4 | Output discipline | 3,467 LLM-path rows: every `raw_response` parsed, one key, in-list id on all 2,348 non-null rows, explicit `null` on all 1,119 nulls. Zero off-list selections |
| M-5 | Null rates | 1,119 of 4,107 (27.2%); **32.3% of LLM-path rows**; product lane 445 of 1,969 (22.6%). Scene-conditioned: `other` 48.9%, utility 36.4%, bathroom 21.1% (kind-mix confound checked) |
| M-6 | Bare-noun shortcut census | 384 of 638 `support_phrase_hit` rows fired only on subject/material nouns (373 under a narrower lexicon). The worklist's four shortcut rows fired on exactly `vinyl`, `tub$`, `tub$`, `window` |
| M-7 | Threshold fragility | 95 rows within 0.002 of the 0.03 margin gate on both checkpoint snapshots (retrieval churn, not a D1/D4 effect); 186 within 0.01 of the 0.72 score gate, 106 below |
| M-8 | Rendering | The model sees id, name, trade, kind, description, first six support terms, two flags. Never scene, scores, `atomic_claim`, `embed_text`, work-item code. Empty system prompt; "choose 0 or 1 … if none fit, return null" |
| M-9 | Stability with text held fixed | Pass 2d self-agreement 99.4% / 97.9% over 330 replicas per arm (Session 6); frozen-shortcut self-test 4,107 of 4,107 |

## 3. Terra, re-scored on conditions with a replication arm (B0)

Every Terra verdict in the corpus is **canary** (17 artifacts with a v5 block); the 8 production artifacts are v4-era and carry none. v1 scored replicated rows; conditions carry 2.15 issues each on average. Pinned run_1 = 939 conditions; run_2 (18 artifacts, 3,588 rows, never previously opened, zero overlap with the pinned corpus) = 1,046.

| Population | Pinned run_1 unsupported | run_2 unsupported | Reads as |
|---|---|---|---|
| All conditions | 14.1% (132/939) | 14.0% (146/1046) | replicates |
| Shortcut-only / LLM-only / mixed | 13.6% / 16.1% / 3.3% | 11.8% / 16.1% / 5.1% | path gap 2.5-4 pts, inside Terra's ~8% same-prompt noise: **no path effect** |
| Kind: defect / degradation / modernization | 28.9% / 15.3% / 8.7% | 28.6% / 14.5% / 8.4% | **replicates strongly; kind is the Terra story** |
| Rank band, single-issue LLM conditions: 1 / 2-3 / 4+ | 22.3% / 22.3% / 12.1% | 22.5% / 15.7% / 20.0% | **no rank trend** (v1's "refutation rises with rank" withdrawn) |
| Divergence: seven strong items / rest | 45.2% (42/93) / 10.6% | 40.4% (38/94) / 11.3% | replicates; band membership is one reader's hand grading |
| Per item, n ≥ 30 | `paint_refresh_recommended` 52%→50%; `peeling_or_discolored_paint` 22%→26%; `older_flooring_style` 6%→20%; `dated_or_older_windows` 6%→0% | | per-item rates move up to ±14 pts even at n 30-47; **no per-item claim below n≈50 is stable** |

Read with the structural caveat in §1: Terra cannot see a wrong-but-also-true selection, so none of these rates measures selection correctness.

## 4. The 36 rows

| Status | Rows | Meaning |
|---|---|---|
| **Shortlist, clean** | 26 | "The shower trim is old." Rank-3 millwork item chosen over rank-1 bathroom finishes; the chosen item's rendered scope excludes shower hardware. No human label. Economic cost untested (work items dedup on max envelope). → card C1 |
| **Shortlist, weak** | 3 | Rank-1 worn/stained flooring passed over for rank 3, twice. Your retag on that card reads "I see stains not scuffs" (claim-level, not a candidate comparison). → card C2 |
| **Withdrawn** | 4, 5, 10 | 4: your retag withdraws the wear claim ("appears fine from the photos"); text favours the selection; better item kind-gated. 5: the "live wrong billing" is `suppressed`/`dedup_collision`. 10: the only human verdict (`terra_claim_supported`) ratifies the OSB selection; the contrary reading is a model |
| **Not a selection error on the evidence** | 1, 2, 6, 7, 8, 9, 11-25 | Model followed the rendered text, or the row is upstream wording, catalog wording/coverage, or Pass 2c kind gating. Row 8: your recorded answer (`mechanism_only`, later claim `misnamed`) **supports** the worklist's naming complaint; stage undetermined. Row 9: human ratified the selected claim's truth on a non-comparative axis; loss was Terra's `cannot_assess` |
| **Accepted limitation (D3)** | 101-110 | Four had no CAP-007 successor in the list (102, 104, 105 retrieval; 108 removed by its own deny terms, bucket `reaching_neither`); six had the blinds item present and the model declined. Its rendered description says "blinds are not billed as renovation work" (N-3), an untested cause |

On the text the model saw, its own pick was the better match on 18 rows, the nominee on 9. That tally is one reader's judgment, concentrated (ten of the eighteen come from one investigator on one item pair), and is reported as a lead, not a finding.

## 5. Corrections to the worklist's annotations (reviewer interpretation; correctable without touching a decision)

Rows 17, 19, 20, 22 assert "kind-open"/"gate-open" for defect items against degradation bullets: the items were kind-gated (M-2); "gate-open" is literally true of guardrails and irrelevant. Rows 16 and 18 carry the `shortcut_or_rank1_selection` axis with no shortcut and ranks 2 and 3. Rows 10, 12, 26 argue from fields the model never sees. Row 3's note misstates the matcher (`scuff` fires) and omits the generic gate. Row 7's `better_item` is the only field in the file naming an id absent from the catalog (R9 itself is satisfied: the successor id appears ten times). The D3 shared note is false on four of ten rows. Twelve CAP-014 rows are ten conditions. **CAP-016's recorded referral ("the bare `ceiling texture` shortcut token goes to the Pass 2d worklist") never landed: zero occurrences in the shipped file.**

## 6. Recorded decisions: all preserved; evidence that bears on them

- **CAP-014 (deferred; trigger = second human property plus demonstrated catalog mechanism).** The record already assigns "eight scuffs-vs-paint rows are Pass 2d selection and two are Pass 2c kind exclusion" (`approvals_v2`, D5). v1 overrode that silently; v3 does not. Evidence: all twelve rows Terra-supported and accepted for work; the item advertises "stains" its claim does not assert; refutation 3.2% (n=188 rows) / 3-4% (conditions, both runs). Cards C3-C4 put the two lower-rank rows to you on photos. Trigger judgment is yours.
- **D3.** Stands; the mechanism written under it is wrong on four rows; N-3 is a lever it did not consider; its reopening trigger reaches only the six rows where the successor was present.
- **D6.** `paint_refresh_recommended` (no change; trigger "Q-8 re-adjudication showing plain neutrals billing") is the worst-verifying item at n ≥ 15 in both runs (52% / 50%). Recorded; trigger judgment is yours.
- **D1, D4, D5, D7, D8, CAP-022 declined, S5-1, all other deferrals:** untouched. Row 23 is now an instance of D4's disclosed residual.

## 7. New facts surfaced by verification

N-1 `run_2` (18 artifacts, 3,588 rows) sat unopened; it is folded into §3 as the replication arm. N-2 Upstream text reproduces on 8 of 523 shared photos (1.5%); 63 of 3,294 observations recur verbatim (1.9%). Any end-to-end experiment needs a text-variation arm; "one replica suffices" holds only with text held fixed. N-3 The blinds item's rendered description tells the model it is not renovation work. N-4 Two recorded referrals into the worklist did not land (R9's field, CAP-016). N-5 Retrieval noise floor: 2 of 1,969 rows change rank 1 between same-catalog snapshots.

## 8. Budget, reconciled against your instruction

Your instruction of 2026-09-09, verbatim: *"2.5 million gpt 5.6 terra tokens which i am currently seeing 0 tokens spent. so we are still clear to spend 2.5 million terra tokens if necessary. there is no cap other than that."* That is the human authorization the manifest's rule requires, and it governs. `docs/analysis/session8_terra_budget_denominator.md` notes that 2.5M/day is also the provider's free daily allowance; that is operational context only: a run that would exceed 2.5M in one day must be split across days. Spend to date: **0 Terra tokens**. The proposed `manifest_v2` budget block (authorized_by Steven, 2026-09-09, 2,500,000 Stage A, ledgered per run) is in the JSON and should be applied to the repo together with this record at Stop 2.

## 9. What the photo review enables (see `pass2d_photo_review_cards_20260909.md`)

Eight cards (rules v2), three axes each answered separately: photo truth (multi-select, "cannot tell" allowed, including claims the model was never offered), sentence fidelity, and match-to-sentence with the photo set aside. Existing human dispositions are shown on each card and preserved; answers are recorded alongside them. No single answer assigns a stage, and no card or pair closes or opens the selection lane. Read together, the eight inform four questions, each a judgment made after the review: whether any row shows a human-confirmed gap between the sentence's best listed match and the model's pick, and whether it mattered for the photo (C1, C2); where the CAP-014 rows' trouble sits, as evidence toward the recorded trigger with the disposition unchanged (C3, C4); whether the "tub" shortcut asserted an unsupported subject (C5); whether declines left true, listed conditions unrecorded (C6-C8). Stage attribution and any change to the Phase B menu are proposed separately afterwards and are yours to accept. The remaining seventeen null rows are not required.

## 10. Phase B menu with authorization state

| Id | Experiment | Provider cost | State |
|---|---|---|---|
| B1 | The eight cards | 0 | **ready now** |
| B2 | Offline enumeration of a subject-aware shortcut rule and of the generic gate (must report the 377-firing side effect of lifting the gate) | 0 | ready; gated on C5 |
| B3 | Rendering/prompt variants on local Qwen (scene in prompt; `atomic_claim` shown; N-3 blinds description; null rule), with a text-variation arm and the N-5 noise floor; Terra only where a selection changes | ≤ ~300k Terra | **gated on C1/C2 or C6-C8 returning a human-confirmed error**; the D4-residual replay arm stays out unless you authorize it separately (decision record §5) |
| B4 | Terra-tier model as the 2d resolver on the same rows | ~100k | only if B3 leaves a confirmed error unmoved |
| B5/B6 | Window-family kind coverage; guardrail backfill | design packets | premature: B5's population shrank with the filter correction; B6 has no demonstrated consequence |

## 11. Owed and not done

The 17 remaining null cards (only if C6-C8 split); the write of this record, the JSON, the provenance scripts and the manifest budget block into the repo (your Stop 2 call); the CAP-016 and R9-field re-pointing in the worklist file (a one-line edit each, yours to authorize); a Pass 2c kind-boundary measurement, which this record names as a cause five times and no instrument has ever measured.
