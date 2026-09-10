# Pass 2d — bounded closeout (2026-09-10)

Implementation on branch `pass2d_prompt_v3` (off `master` @ `fb945b0`), **validated and recommended against**. Supersedes the 2026-09-10 decision document, which recommended N on temperature-0.2 evidence without generalization cases. Nothing merged, nothing published, no catalog change, no production artifact touched.

## 1. What was implemented

Two lines added to `PASS_2D_USER_PROMPT_TEMPLATE` in `tools/scene_classifier_passes.py`:

```
- If the observation names a specific instance of a candidate's subject, select that candidate
  (a faucet is a bathroom fixture; a range is an appliance; a cabinet door is a cabinet).
```

`PASS_2D_PROMPT_VERSION` bumped `pass_2d_exact_kind_v2` → `v3`; the prompt sha moves from `c779b2ac…` to `5caccea1…`. `tests/test_scene_classifier_passes.py` version pin updated. Four lines across two files.

## 2. What validation demonstrated

Unchanged prompt vs candidate prompt, **both at Qwen temperature 0.1**, 3 replicas, 93 rows, stored candidate lists and observation text held fixed, 558 calls, 0 failures.

**Baseline fidelity is exact.** At temperature 0.1 the unchanged prompt reproduced the stored selection on 40 of 40 non-null control rows and declined on 42 of 42 stored declines. Temperature 0.1 is also markedly steadier than 0.2: 7 unstable rows of 93 against 1 for the candidate.

**Across all 93 rows the candidate prompt changes exactly three:**

| Row | Base | Candidate | Reading |
|---|---|---|---|
| "The range appears older." (C6) | null 4/4 | `appliances_dated_or_basic` 3/3 | recovery — **and the prompt's own example** |
| "The faucet has an older style." (C7) | null 3/4 | `outdated_bathroom_finishes` 3/3 | recovery — **and the prompt's own example** |
| "The pavement is stained and aged." | `patio_or_porch_surface_wear` 3/3 | null 3/3 | **stable regression** |

**Generalization: the rule fired on 0 of 8 cases outside its own examples.** Backsplash to kitchen finishes, rear door to entry door, screens, exposed joists, porch, exterior trim, window-trim paint — none moved. On the backsplash the candidate is *worse*: the base selected it once in three, the candidate never. The one clean instance relation that does resolve, "Tile surround is dated" to `outdated_bathroom_finishes`, already worked under the base prompt (2/3) and the candidate only stabilised it (3/3).

**Correct nulls held.** D2 (swirl plaster) and D3 (laundry cabinets, right item absent) stayed null at 3/3 under the candidate. Controls: 0 of 40 correct declines moved; 1 of 40 non-null moved, the paved-surface regression.

**An incidental finding that matters more than the change.** At temperature 0.1 the *unchanged* prompt already selects `dated_overall_decor_style` for C8 ("Kitchen is functional and very dated", 3 of 4), which Steven judged acceptable. At 0.2 it returned null. The temperature setting moved that row; the prompt change did not.

## 3. Material tradeoff, and the recommendation it forces

The change buys two rows, and those two rows are the two subjects written into its own text. It generalises to nothing, and it costs one stable regression. That is a hardcoded special case wearing the clothes of a rule, which is the pattern this codebase is explicitly moving away from.

**Recommendation: implement neither N nor S. The branch is preserved as evidence and should not be merged.**

This reverses my 2026-09-10 recommendation of N. That recommendation rested on temperature-0.2 runs whose anchors were C6 and C7 — the examples themselves — with no generalization cases. The generalization test Steven asked for is what refuted it, and it should have been in the first design.

## 4. The wrong-subject shortcut: deferred with a trigger

The defect is real and human-confirmed twice (C5 floor stained not fixtures; D1 ceiling stained not fixtures). Rule E was rejected for being too broad, not for being wrong about the defect. A narrower two-sided hypothesis is recorded in `docs/PROPOSAL_pass2d_wrong_subject_followup_20260910.md`: flag only rows where every firing term is a bare subject noun that is *another item's* claim subject **and** an item matching the sentence's own subject is present in the same list. Its first step is a free offline enumeration with a kill condition.

Deferred rather than run, for two stated reasons: it needs a catalog subject index that does not exist yet, which is design work excluded from this closeout; and the measured billing consequence of both confirmed cases is currently zero (C5's billing was suppressed by dedup collision). **Trigger:** a wrong-subject shortcut that produces a real billing, or a subject index arriving for another reason.

## 5. Worklist reconciliation

`reports/pass2d_worklist_reconciliation_20260910.json`. Naming errors, missing observations and billing consequences are tracked separately.

| Status | Rows | |
|---|---|---|
| Fixed by an implemented change | **0** | N addresses missed selections; the worklist catalogues wrong selections. Disjoint populations |
| Unresolved 2d issue | 5 | rows 3, 5, 16, 18, 26 |
| Accepted limitation | 13 | rows 6, 7, 23 and all ten D3 blinds rows |
| Outside 2d | 18 | upstream wording, catalog wording or coverage, or the Pass 2c kind boundary |

Defect class: 18 naming, 10 missing observation, 8 none. Billing consequence: 29 none, 3 lost, 3 wrong line, **1 unmeasured** (row 26, where work items dedup on a max envelope so a correct selection might have merged rather than added).

The five unresolved rows, briefly. Row 3: the sentence is partly false (window light) and the pick is unsupported. Row 5: human-confirmed wrong subject, billing already suppressed. Row 16: Steven judged the *selected* item better than the worklist's nominee; the item his answer implies is kind-gated. Row 18: the sentence is unsupported and no listed candidate fits. Row 26: the pick named a real but minor thing and missed the primary subject; also a CAP-013 third property, recorded as evidence only.

## 6. Preserved

Every recorded catalog decision (D1–D8, CAP-022 declined, all deferrals with their triggers) is untouched. Steven's twelve photo judgments are recorded verbatim in `docs/REVIEW_pass2d_photos_steven_20260909.md` with the coordinator's axis mapping marked as such, and claim truth is kept separate from whether a pick matched the supplied observation throughout.

## 7. Tokens

| | Calls | Tokens |
|---|---|---|
| Local Qwen (S, N, resolver comparison, combined check, validation) | 970 | 672,474 |
| GPT-5.6 Terra (resolver comparison only) | 238 | 162,893 |

Terra: **162,893 of 2,500,000 authorized (6.5%), 2,337,107 remaining.** No Terra spend in this closeout; validation was local only. No Terra verification calls at any point, because Terra verifies the claim of whatever item was selected and cannot adjudicate a wrong-but-plausible selection.

## 8. Test state

115 tests covering the prompt, the benchmark and both catalog checkpoints pass on the branch. Full suite: 2852 passed, 9 failed. All nine are catalog-audit program tests failing on stale catalog ids and evidence-bundle drift from CAP-007 and the checkpoint; none references the 2d prompt, and the same family reproduces on `master`. The benchmark slices were not re-scored: that needs the embeddings sidecar, which was down, and with the change not recommended there is nothing to score.

## 9. Files in the repository

Decisions and results in `docs/`: `RESULT_pass2d_diagnosis_20260909.md`, `RESULT_pass2d_experiments_SN_20260910.md`, `DECISION_pass2d_20260910.md`, this closeout, `PROPOSAL_pass2d_wrong_subject_followup_20260910.md`, `PROPOSAL_pass2d_phase_b_20260909.md`, and the three photo-review documents.

Reports in `reports/`: `pass2d_diagnosis_20260909.json` (the consolidated record), `pass2d_experiments_20260910.json` (every resolver call), `pass2d_worklist_reconciliation_20260910.json`, `pass2d_terra_rescore_20260909.json`, the three shortcut-rule enumerations, and `pass2d_verification_cores_20260909.json`.

Provenance scripts in `scripts/pass2d_diagnosis/`. Raw run directories are not versioned, per the repository tracking policy; per-row results are folded into the JSON reports so a new session needs no temporary files.
