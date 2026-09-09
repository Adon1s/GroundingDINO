# Catalog audit — Tier 1 and Tier 2 validation result

Date: 2026-09-07
Session: 5 (`docs/catalog_audit_program/06_SESSION_DETERMINISTIC_VALIDATION.md`)
Candidate: `6e67eaa` on `catalog_audit_session4` — CAP-007 window-treatment split
Baseline: `9afe0fa` on `terra_factorized_verifier` (main checkout)

## Verdict

**Tier 1 pass (36/36 checks). Tier 2 pass (14/14 rules).** No provider call was made. The live
experiment manifest is written and awaits a human cost authorization.

The implementation is exactly what was approved: the generated diff equals
`expected_generated_changes` field for field, the in-memory dry run reproduces
`dry_run_at_approval` exactly, double generation is byte-identical, both validators return zero
errors, and no tracked file outside the approved eight differs in content between the arms.

Two findings are carried forward. Neither is an implementation defect; both are consequences of
the approved wording that only a retrieval replay could surface, and both are named as mandatory
Stage A cases in the manifest.

| Artifact | sha256 |
|---|---|
| `reports/catalog_audit_tier1.json` | `cbe2bc59b35775c5bf52145436a6d2a4b6fe19720a0057bbd8a45c56100a2b07` |
| `reports/catalog_audit_replay_corpus.json` | `159053217cf434793642667646cf79d92198d9900e061fbe360cd25149d0c965` |
| `reports/catalog_audit_replay_snapshot_baseline.json` | `07482b02c4d3cd389d281f5dc670ad0639ce12aed6e1bcb427f0ceec3d950181` |
| `reports/catalog_audit_replay_snapshot_candidate.json` | `040cc41b79d569a0b39a5df3dde0837bf8d03aa4517120245c9439596c1bdbfb` |
| `reports/catalog_audit_validation_tier12.json` | `674a9fccea8a4bb1ddcfcab10b198f3621b47a7553614ce48f817233994aee4a` |
| `reports/catalog_audit_live_experiment_manifest.json` | `e1f3b7f1f967266e64c71dedb6f93cef2e329cc02dca78799cfec4a678a4a42f` |
| `scripts/catalog_audit_validation.py` | `dbacf929b5ddba177e555456ee51d1685ee9b7b79261889c973fc1a1d8127ce6` |
| `tests/test_catalog_audit_validation.py` | `ba3bf5a4a059f31f04f56ca82b64cfbef0727789fe00bbeef023e2da1d708684` |

## Tier 1 — deterministic

All 36 checks pass. The load-bearing ones:

| Check | Result |
|---|---|
| Both arms at the expected commits, tracked-clean | pass |
| Every `approved_against` pin in the approvals manifest re-verified | pass |
| Double regeneration in the candidate worktree, byte-identical, tree still clean | pass |
| `apply_ops(baseline, approved ops)` reproduces the candidate decisions byte for byte (CRLF) | pass |
| `generate(v1, patched)` reproduces the candidate catalog and migration manifest | pass |
| Generated diff equals `expected_generated_changes` | pass |
| `dry_run` reproduces `dry_run_at_approval` exactly | pass |
| Both validators, both arms, zero errors; identical warning sets | pass |
| Economic fields identical on all 127 shared items; successors carry the approved inherited economics | pass |
| Terminal routes change only for the parent and its two successors | pass |
| 128 → 129 items, successors at the parent's index, all other order preserved | pass |

Route counts move exactly as approved:

| Route | Baseline | Candidate |
|---|---:|---:|
| work | 98 | 98 |
| no_action | 9 | 10 |
| excluded_quarantine | 12 | 12 |
| inspection | 5 | 5 |
| excluded_generic | 4 | 4 |

### Test suites

Both suites were run with a clean environment and a scratchpad base temp.

| Arm | Result |
|---|---|
| Baseline (main checkout) | 2756 passed, 6 skipped, 0 failed |
| Candidate (worktree) | 2759 passed, 6 failed, 21 skipped |

The six candidate failures are exactly the set Session 4 handed over, each confirmed by its own
message: three renderer pin-drift failures that are correct by design and pass at baseline, one
pre-existing environment failure for want of untracked inputs in the worktree, and the two known
ones addressed below. No other test changed state.

### Two arm-isolation facts worth recording

**92 tracked files differ on disk between the arms by line endings only.** Every one has an
identical git blob. The main checkout holds them as LF, the worktree checked them out as CRLF.
None is in the retrieval path, and the catalog artifacts themselves are CRLF in both arms and
reproduce their pinned hashes. Code identity is therefore compared by git blob; on-disk hashes are
still used for the four catalog artifacts, which is what the program pins.

**Both arms stamp catalog version `3.2`.** The version string cannot distinguish them. Every live
run must be identified by `catalog_sha256` and the projection fingerprint, both of which are in
the manifest.

## Tier 2 — provider-free retrieval replay

The embeddings sidecar was probed with a real POST before any replay: 1024 dimensions returned,
served model and build recorded. A 200 from `/health` was not treated as readiness.

Both arms replayed 1969 observations drawn from the 25 pinned evidence-era artifacts, all
re-hashed against the bundle before reading. The two arms shared a vector cache keyed on exact
text, so identical strings embed to identical vectors and the only variable is the catalog. The
candidate run needed exactly two new embeddings, one per successor.

### Two checks that establish the replay is comparable at all

**The frozen shortcut self-test passes on all 4107 stored rows.** Today's
`_resolve_candidate_via_lexical_shortcut`, run over each artifact's own stored candidate list,
reproduces the recorded resolution path, shortcut reason and resolved id with zero mismatches.
The thresholds and negation patterns have not drifted since the evidence era.

**The approved census reproduces from the artifacts.** 48 parent conditions found, against 48 in
the redraft, and the moving/surviving/neither split reproduces at 28/10/1 under the production
gate functions in the runtime's own order (kind, scene, deny, require).

Residual score drift between the arms peaks at 3.6e-07, four orders of magnitude below the 0.03
margin gate, and no shortcut decision that differs between the arms sits inside that band.

### The four measurements the approval required

**1. Blinds lexical-shortcut fire rate.** Denominator 46 rows — modernization, scene-eligible,
passing the blinds gate, naming a blind or shade — which matches the redraft's own count.

| | Baseline (onto the parent) | Candidate (onto the blinds successor) |
|---|---:|---:|
| Fires | 26 | 23 |
| Rate | 56.5% | 50.0% |

The mechanism matters more than the rate. The presence-only successor deliberately carries no
datedness language, so on "the blinds are dated" bullets it sits closer to `dated_or_older_windows`
than the parent did and the top-two margin collapses toward the 0.03 gate. Nine of the 46 rows fall
below the gate and now reach the LLM instead of resolving deterministically, including two reviewed
controls:

| Observation | Baseline margin | Candidate margin |
|---|---:|---:|
| The blinds are old. | 0.0788 | 0.0215 |
| The window and blinds are dated. | 0.0427 | 0.0224 |
| The blinds appear dated. | 0.0663 | 0.0123 |
| The window blinds are dated relative to the renovation. | 0.0553 | 0.0029 |

This is not a reachability loss: the blinds successor is rank 1 on all of them. It is a change of
resolution path, from deterministic to model-decided, on the observations the split most targets.
A further 91 rows across the corpus now decide within 0.002 of the margin threshold.

**2. Post-split ranks against `dated_or_older_windows`.** 913 rows involve that item. It loses no
ranked position it held in the baseline arm on any row it owned or ranked in the top three, its
reviewed correct rejection (`rc_64f996e3a886`) is bit-identical across arms, and no row moved from
the windows item to the blinds successor by shortcut. The blinds successor outranks it on 62 rows,
all of them blinds-subject bullets. At the item level the successors sit closer to that item than
to each other:

| Pair | Cosine |
|---|---:|
| parent to `dated_or_older_windows` (baseline) | 0.786 |
| fabric successor to `dated_or_older_windows` | 0.738 |
| fabric successor to blinds successor | 0.737 |
| blinds successor to `dated_or_older_windows` | 0.691 |

**3. Fabric successor's bare-stem surface.** 24 rows contain `curtain`, `drape` or `fabric`. Thirteen
pass the fabric gate, seven are denied (the A-3 stems `shower` and `tub` doing the work the
two-word phrase could not), and four are excluded by the kind gate. Six resolve onto the fabric
successor by shortcut. **No row owned by another item is taken by the fabric successor** — the
neighbour-hijacking risk the approval flagged did not materialise. The approved leak test
reproduces field for field across all 27 recorded bullets.

**4. Embedding neighbourhoods.** Reported for the parent and both successors, restricted to the
same kind and shared scene groups. The only flagged neighbours are the pair above and the blinds
successor's own sibling; no presentation-only item sits within 0.03 of either successor.

### Reachability against the approval's accounting

45 of 48 parent conditions reach their predicted successor inside the production top-8.

| Predicted bucket | Reachable | Total |
|---|---:|---:|
| moving to no_action | 35 | 37 |
| surviving on fabric | 9 | 10 |
| reaching neither | 1 | 1 |

The single "reaching neither" condition is `oc1_c839d6e1118a7006`, the makeshift-covering case,
confirming A-6 exactly as accepted: denied by `curtain` at the blinds successor and by `makeshift`
at the fabric successor.

The reviewed positive use (`rc_7c5b9f9bb3aa`, "The blinds are old.") reaches the blinds successor
at rank 1, so the Q-1 withdrawal proceeds as approved rather than the condition becoming
undetectable.

## Findings carried to Session 6

**F-REACH-`oc1_93a2e3f3d299a7a1` (material).** The approval counts this condition among the ten
surviving on the fabric successor, so it keeps billing. In the replay the fabric successor is not
in the candidate top-8 for either of its observations. On "The blind, floral valance, and plain
trim make the room feel older." the bullet names a valance and passes the gate, but generic decor
language dominates the embedding and the fabric successor never becomes a candidate; the blinds
successor does reach the list and is then denied by `valance`. On its other observation the fabric
successor reaches rank 2 and is denied by `require_any`. **The approval's count of surviving
billings is therefore optimistic by one condition.** This is a consequence of the approved wording,
not of the implementation, and it is a decision for the human reviewer, not for this session.

**F-SHORTCUT-`fee5271eaff08de5` (material).** "The mini blinds feel dated compared with the updated
kitchen." Removing the parent lifted `outdated_kitchen_finishes` to rank 1 and opened a lexical
shortcut onto it at a margin of 0.03004, four hundred-thousandths above the gate. In the baseline
the parent held rank 1 with a margin of 0.0153, below the gate, so the row reached the LLM. The row
now resolves deterministically to a billable kitchen item rather than to the no_action blinds
successor at rank 4. It is both a change of owner and the most threshold-fragile decision in the
corpus.

**Expected-severity findings**, recorded but not blocking: three subject-generic bullets
("Window treatments are dated.") no longer reach either successor, which is what the approval
intended, though two of them now shortcut onto `dated_or_older_windows`, whose route is also
`no_action` — the same economic outcome under a different item id in the frontend envelope.

**Informational.** On 261 rows a successor entered the raw top-8 and was then denied by a
guardrail, spending the slot so a lower-scoring item never became a candidate. This is existing
retriever behaviour, since guardrails run after top-K, amplified because two items now compete
where one stood.

## The two failures Session 4 handed over

**Benchmark gold coverage (S4-1) — drafted, not applied.** The slices under
`benchmarks/catalog-resolution-v2/` are frozen and fingerprinted, committed results pin those
fingerprints, and the benchmark's own gate mode refuses a non-frozen slice. Authoring cases into
them is evaluation content and outside a validation session's authority. Two schema-valid drafts
are recorded in the manifest, each grounded in an observation that actually resolves to the
intended successor at rank 1 in the candidate replay, with its property, run and artifact hash:

- `res-mod-cap007-blinds` — "The mini blinds have a basic style." (modernization, bedroom)
- `res-mod-cap007-fabric` — "The curtains and valances are dated." (modernization, living_areas)

Both satisfy the schema and the shipped-case checks. Session 6 adds them and re-freezes alongside
its live benchmark run. `test_every_split_successor_is_gold_somewhere` stays red until then.

**CCF-8 stored-artifact replay (S4-2) — confirmed as designed, inventoried.** The failure is the
approved behaviour: a stale id must fail hard. No alias was added and no stored artifact was
edited. The manifest carries a full inventory: **97 stored canary artifacts across six roots still
name the deprecated parent**, including all 25 pinned evidence-era runs. Each fails offline
recompute against the candidate catalog. The fix is a replay from frozen Pass 2c, which the
manifest requires as a stage and enforces as a stop condition.

## Live experiment manifest

`reports/catalog_audit_live_experiment_manifest.json` pins both arm identities with catalog hashes
and projection fingerprints, the approved proposal and its conditions, 66 targeted cases (8
controls, 3 findings, 55 affected) each carrying its frozen Pass 2c observation, kind, scene group,
issue id, stored artifact path and hash, photo reference with image hash, and the offline
prediction for both arms. All 66 photos resolved and hashed.

It also pins the model map and environment model names, prompt versions and shas, git-blob code
hashes, the embeddings configuration and the sidecar probe, the stages to rerun and the stages that
stay frozen, the metric and comparison schema, targeted-first ordering with full-canary
authorization held separately, stop conditions, and an invariant list to verify before any call.

**Both budget fields are null.** Session 6 may not make a provider call until a human fills them
in. Reference costs recorded: about 3.98M tokens for a full 18-property replica, about 3.2k tokens
per single-condition Terra call.
