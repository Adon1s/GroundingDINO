# Handoff — catalog audit session 1

Date: 2026-09-01

Session/task title: Evidence foundation and worklist (`01_SESSION_EVIDENCE_FOUNDATION.md`)

## Outcome

Complete. Every exit criterion in the brief is met:

- All required source hashes are recorded and reconciled (11 frozen-tier files, 6 queue-embedded pins, 25 run artifacts; zero mismatches, zero drift from the `e9a7dc3` baseline).
- Evidence and proposal baselines are explicit and mechanically derived: evidence era = catalog 3.1 at `07ee112` (CRLF checkout hash `d1cef743…`, recorded by every pinned run artifact), proposal baseline = catalog 3.2 at `a9ed7ad` (on-disk `51bf7e26…`, identical to HEAD's blob).
- The 3.1-to-3.2 comparison is stored: 128 → 128 items, same ids and order, the only field that changed is `package_affinity` on 10 items (the 24 `repair_support_when_driven` markers); `atomic_claim` and the derived Terra claim text are unchanged, computed rather than assumed.
- All 162 seed records reconcile exactly once into 140 evidence units plus 19 unattached records.
- Positive uses (80), agreements (7), correct rejections (31), and reviewer notes (27) are joined and counted per item and per family.
- No semantic proposal or production artifact was changed; `git diff --stat` is empty.
- The evidence bundle is deterministic (two builds byte-identical, `--check` passes) and tested (18 tests).

## Repository state

- Starting commit: `e9a7dc3a54439fb341c1ae8cd665e5e572fcfaf6`
- Ending commit: three commits on the branch — `a7eed9b` (the five session files), `48be33a` (builder: git dirty-state scoped to consumed inputs; working copies CRLF-normalized, no blob change), and the commit that adds the bundle regenerated from `48be33a` plus this handoff revision (`git log -1 -- reports/catalog_audit_evidence.json`). The bundle's `git.head` is `48be33a`, the builder it was built from.
- Branch/worktree: `terra_factorized_verifier`, main checkout
- Dirty state at start: no modified tracked files; untracked user files only (root-level `catalog_audit_*.json` from the legacy `tools/catalog_auditor.py`, `artifacts/`, `artifacts_canary/`, `.claude/`, `.gitattributes`, `reports/labels_v1_1.json`, others).
- Dirty state at end: the same untracked set plus this session's five files.
- Pre-existing changes preserved: yes. No tracked file was modified; no untracked file other than the five listed below was created, and the builder's scratch outputs were removed.

## Inputs verified

| Input | Expected identity/hash | Actual identity/hash | Result |
|---|---|---|---|
| HEAD / branch | `e9a7dc3` on `terra_factorized_verifier` | `e9a7dc3a54439fb341c1ae8cd665e5e572fcfaf6`, same branch | match |
| `reports/error_attribution_queue.json` | `b58dcca7…40df` | `b58dcca7c3779f2b0539f988077f865c6c1ca040d23a44646d89b04a398940df` | match (104 cases, 15 gold photos / 112 findings, lane counts identical to header) |
| `reports/error_attribution_verdicts.jsonl` | `166e8bec…894e` | same | match (107 lines → 95 effective: 55 `rc_` + 40 `ga_`; all 12 collapses are `revision_of` re-reviews, zero undo records) |
| `reports/error_attribution_gold_cases.json` | `77a7b515…954a` | same | match (40 cases, 127 matching rows) |
| `reports/labels_v1_1.json` (untracked) | `7f9b0301…3bd` | same | match (125 labels, 80 `supported_billed`) |
| `reports/review_queue.json` | `8512b87c…318a` | same | match (212 cards) |
| `reports/review_analysis.json` | `3652ba3c…259c` | same | match (27 qualitative notes) |
| `reports/factorized_review_scorecard.json` | `60300ec5…d103` | same | match (50 misnamed leads) |
| `reports/review_verdicts.jsonl` | `0c7ca6a8…d70f` | same | match |
| `benchmarks/pass2a-prompt/gold/reference.json` | `25307513…4543` | same | match |
| `tools/issue_catalog_kind_v2.json` | `51bf7e26…aa54` | same | match; equals CRLF conversion of HEAD blob `53a7b8ea…` |
| `tools/catalog_migrations/kind_v2_decisions.json` | `47614d82…03e8` | same | match; decision counts 32/50/4/19/2 recounted |
| Queue-embedded input pins (6) | queue `inputs[*].sha256` | current files | 6/6 match |
| Run artifacts named by the queue (25) | `run_ref.artifact_sha256` | re-hashed on disk | 25/25 match; 17 canary + 8 production. The 26th run (`production/redfin_10965375/20260821_233007_c5b989ee`, 11 orphan-lane cases) has no path or hash in the queue and was not read. |
| Artifact `catalog_sha256` | `d1cef743…8347` on every run | identical on all 25 pinned artifacts, `catalog_version` 3.1 | match; resolved mechanically to commit `07ee112` (LF blob `20adb6e3…`, CRLF `d1cef743…`) by hashing all 6 historical blobs of the generated catalog both ways |
| Recorded-tier inputs (hash only) | — | generator `b732022f…`, validator `041637a3…`, manifest `3db144c9…`, v1 catalog `4ba046a1…`, FINDINGS doc `23d336fa…`, factorized manifest `89f8fccd…` (untracked) | recorded in `sources` |

The context's prose lane names ("hallucination-label", "miss-v1-only") map to the real ids `halluc_label`, `halluc_v1only`, `miss_label`, `miss_v1only`; counts agree. `docs/CONTEXT_catalog_audit_inputs_20260901.md` says 10 production runs; the queue references 9 (8 pinned + 1 orphan), as the overall context already states.

## Files created or changed

| Path | Status | Purpose |
|---|---|---|
| `scripts/build_catalog_audit_evidence.py` | new (sha256 `8874bc34…af78`) | the one evidence builder: provenance freeze, catalog identity, 3.1↔3.2 comparison, joins, seeds, units, families, worklist, Markdown rendering; `--check` rebuilds in memory and compares fingerprints |
| `tests/test_catalog_audit_evidence.py` | new (sha256 `f690ce38…5440`) | 18 tests on synthetic fixtures plus one guarded parity test against the committed bundle |
| `reports/catalog_audit_evidence.json` | new (sha256 `43354104…29b7`, fingerprint `e4c0a2b8…0c08`) | principal machine-readable output, schema version 1 |
| `reports/catalog_audit_evidence.md` | new (sha256 `f48c713d…70fa`) | compact rendering of the same dict (provenance, identity, comparison, seed table, worklist, unavailable data, validation) |
| `docs/catalog_audit_program/HANDOFF_SESSION_1.md` | new | this handoff |

No other path was touched. Prohibited paths (decisions, both catalogs, migration outputs, prompts, runtime, frozen queue/ledger) were read only.

## Work completed

- **Workstream A (provenance):** two pin tiers. Frozen-tier files are pinned as constants in the builder and fail the run on any mismatch before anything is built (`verify_pins` / `fail_closed`); recorded-tier files are hashed only. The queue's embedded pins and all 25 `run_ref.artifact_sha256` values are re-verified on every build. `git` block holds HEAD, branch, tracked changes, and which consumed inputs are untracked (`reports/labels_v1_1.json`, the factorized manifest, the builder itself).
- **Workstream B (baseline reconciliation):** `resolve_catalog_identity` walks `git log -- tools/issue_catalog_kind_v2.json`, hashes every blob as stored (LF) and as a CRLF checkout, and matches the artifact hash set; exactly one state matches. `compare_catalogs` diffs 27 fields per item (semantic, retrieval, route, every `ECONOMIC_FIELDS` entry, plus derived `claim_text` and `embed_source_text`). Result: only `package_affinity` differs, on `peeling_or_discolored_paint`, `worn_or_stained_carpet`, `vinyl_linoleum_worn_or_stained`, `cabinets_worn_finish`, `appliances_worn_or_neglected`, `baseboard_wear_scuffs`, `wall_scuffs_marks_or_dents`, `vanity_countertop_worn`, `vanity_countertop_dated`, `worn_or_stained_flooring`. Product quarantine is `electrical` in both eras.
- **Workstream C (indexes):** cases by id with latest attribution, run identity, `condition_item` and per-issue `p2d.resolved_item_id` (never `p2c_surviving[].catalog_item_id`), gold cases, `by_item`, `by_run`, `by_photo`, `by_condition` (keyed `runtime:source:property:run:condition`); migration parent/successors/siblings from the manifest; all 125 labels joined to their exact cards (0 item mismatches, 0 missing cards); 27 notes joined by card id; leads joined by `(canary, property, run_1 run id, condition_id)`; gold matching rows carry run identity from the queue's gold photo.
- **Workstream D (frozen candidate enrichment):** performed for 93 pending cases, 37 unjoined pinned leads, and 15 gold photos from the exact hash-verified artifacts (`photos[k].debug.resolved_items[].candidates[]`, verbatim order, rank = position, scores untouched). Artifact resolutions agree with the queue's `p2d.resolved_item_id` on every issue.
- **Workstream E (seeding):** seven mechanical rules, each recorded on the record with its reason: R1 latest 2d attribution (12 `rc_` + 8 `ga_`), R2 `appendix_misnamed` lane (8, incl. the 2 attributed `pass_2a`), R3 miss lanes attributed Terra (21), R4 factorized leads (50), R5 gold miss candidates (40), R6 gold `out_of_catalog` rows (27), R7 the 11 deferred-finding sections (backlog only; item refs extracted from backticks and intersected with catalog ids). Quarantined trades divert to `product_policy` (0 hits). The 10 `pass_2a` hallucination cases annotate implicated items and never seed.
- **Workstream F (dedup):** 162 records → 140 units (73 runtime, 67 gold), 177 rule/record pairs, 3 `method_corroboration` records (leads on seeded conditions), 19 unattached (11 backlog references, 8 unpinned leads). 77 units carry an implicated item; 63 gold units are coverage questions with no item. Same-photo and same-property correlation flags are on every unit; no gold unit's covering condition is itself a seeded unit, so `corroborating_units` is 0.
- **Workstream G (families):** for each of the 29 worklist items: migration parent/successors/siblings, same kind + trade with overlapping scene groups, shared `work_item_code` / `estimate.group` / `package_type`, frozen candidate neighbors aggregated from enrichment, `current_retrieval_neighbors = unavailable` (no offline embedding store), blank `reviewer_neighbors`, normalized observations, and structured dimensions (claim subject/state, kind, trade, scenes seen, claim text per era).

## Decisions and invariants

- Evidence-era semantics are reconstructed as `item_semantics.items` (proposal baseline, all 128 items) overlaid with `baseline_comparison.changed_items[].changes[field].evidence_era`; only differing fields are stored twice.
- Independence keys: runtime `(source, property_key, run_id, condition_id)`, gold `(property_key, photo_key, gold_id)`. Within a unit, precedence is human-verdict case > gold case > factorized lead > gold row; the primary is `independent`, a different method is `method_corroboration`, the same method is `annotation`. A gold unit whose `covering_rejected_condition_id` names a seeded runtime unit is corroboration of it, not a second case.
- A gold case's implicated item comes from its covering condition: through the queue case when one exists, else through the hash-verified artifact (`covering_item_source` says which; 4 cases resolved via artifact: `wall_scuffs_marks_or_dents`, `dated_or_older_windows` ×2, `unfinished_interior_wall_osb_exposed`).
- `fingerprint = sha256_canonical(bundle minus git and fingerprint)`. The builder's own hash is inside `sources`, so any builder edit changes the fingerprint and the parity test demands regeneration.
- Reconciling a frozen-tier drift means editing `FROZEN_SHA256` in the builder and recording why in the handoff; there is no override flag.
- Worklist rows carry counts only. The evidence bar, clustering, and diagnosis are Session 2's.
- The 8 leads on `redfin_126224899` stay unattached: that canary property has no queue case, and its `run_1/candidate` folder holds two run directories, so no frozen file pins which artifact the factorized run used.

## Deviations from the session brief

| Deviation | Reason | Impact |
|---|---|---|
| Optional Workstream D enrichment performed | all 25 named artifacts verified byte-identical | Session 2 has original 2d shortlists without any rerun |
| Terra-attributed misses seeded in full, without a mechanical "specificity commitment" filter | any such filter is diagnosis-shaped; the design review recommended leaving it to Session 2 | 21 R3 records rather than a subset |
| Current-baseline retrieval neighbors recorded as unavailable | no offline embedding vectors exist; obtaining them needs the live sidecar, which is Session 5 Tier 2 work | `families[*].current_retrieval_neighbors.status = unavailable` |
| Evidence-era catalog materialized with read-only `git show` inside the builder instead of committing a second copy | keeps one catalog file in the tree; history is append-only for the resolved commit | builder fails closed if git is unavailable |
| `git` block records tracked changes and per-input tracked flags but not an untracked-file count | an untracked count changes between runs (the builder's own outputs), breaking byte-identity | determinism proven by fingerprint and byte-identical repeat runs |
| `scripts.build_error_attribution_queue.load_inputs` not reused; files loaded directly | the builder needs the qualitative notes and matching table that loader does not expose; only `latest_verdicts`, `review_cards`, `_claim_text`, `_catalog_text`, and quarantine accessors are reused | none |
| Compact Markdown summary produced | permitted by the brief; renders from the same dict | second output file |
| Bundle committed in two steps (session files, then the bundle regenerated from the committed builder) | the git block records the builder's commit and dirty state; building from an uncommitted builder would have recorded a modified builder | three commits instead of one |

## Commands and verification

| Command/check | Result | Notes |
|---|---|---|
| `.venv\Scripts\python.exe -m pytest tests/test_catalog_audit_evidence.py -q` | 18 passed | includes the parity test against the committed bundle; `--basetemp` under the session scratchpad |
| `.venv\Scripts\python.exe -m pytest tests/test_error_attribution.py tests/test_catalog_kind_v2.py tests/test_catalog_validation.py tests/test_catalog_embeddings.py tests/test_review_analysis.py -q` | 244 passed, 1 skipped | pre-existing skip; none of these import the new builder |
| `.venv\Scripts\python.exe scripts\build_catalog_audit_evidence.py` (twice) | JSON and MD byte-identical (`cmp`) | ~8 s per build |
| `.venv\Scripts\python.exe scripts\build_catalog_audit_evidence.py --check` | `fingerprint matches: e4c0a2b816395782cd84bc4abcdf6207d131a34a0e2d63eaebb39bec4c190c08` | |
| `git status --porcelain \| grep -v '^??'` / `git diff --stat` | empty | no tracked file modified |
| Bundle `validation.ok` | true, 13/13 checks | informational checks (`issue_items_equal_condition_item`, `label_item_matches_card`, `labels_without_card`, `artifact_resolution_agrees_with_queue`) all report empty exception lists |

## Outputs for the next session

| Artifact | Hash/commit | Contract |
|---|---|---|
| `reports/catalog_audit_evidence.json` | sha256 `43354104cc109c768efbe2d2f8b9a825f33e0882210b81e9f12b847ca67e29b7`; fingerprint `e4c0a2b816395782cd84bc4abcdf6207d131a34a0e2d63eaebb39bec4c190c08` | schema 1; sections `sources`, `queue_input_pins`, `run_artifacts`, `catalog_identity`, `baseline_comparison`, `migration`, `item_semantics`, `indexes`, `labels`/`positive_uses`/`agreements`/`correct_rejections`, `review_notes`, `factorized_leads`, `gold`, `candidate_enrichment`, `seeds`, `evidence_units`, `unattached`, `worklist`, `families`, `lanes`, `reconciliation`, `validation`, `unavailable`, `notes`, `fingerprint` |
| `reports/catalog_audit_evidence.md` | sha256 `f48c713d94f6afea3b1ba7aa06167d3ed24352ee67547420ca6f8bc68a6e70fa` | rendering of the JSON; start here for orientation |
| `scripts/build_catalog_audit_evidence.py` | sha256 `8874bc340b93c48ef0014bbf2529d750951709d70afb2bd6016c10d0e6deaf78` | re-run with `--check` to confirm the committed bundle still matches the inputs |
| `tests/test_catalog_audit_evidence.py` | sha256 `f690ce389c85669a8c63bbce0b55c44bf2ded1833a05210e45a4e67ea5c25440` | parity test skips when the production artifact root or frozen inputs are absent |

## Unresolved questions and risks

- Two worklist items are carried entirely by factorized model leads (R4): `older_flooring_style` (12 units, 7 properties) and `wall_scuffs_marks_or_dents` (12 units, 9 properties). Leads are model outputs from a verifier that failed its own gates (`docs/RESULT_factorized_replay_20260831.md`); Session 2 must not count them as human evidence.
- The 2d-versus-Terra stage boundary sits inside the review noise floor (`docs/RESULT_error_attribution_20260831.md` §7). R1 and R3 partition on that boundary; a cluster should not be built on the stage label alone.
- Gold units share photos densely (67 units on 15 photos), so `same_photo_units` is a weak correlation signal for gold; `same_property_units` and covering links matter more.
- The 4 gold cases whose covering condition was resolved through the artifact name conditions no human reviewed; their items are implicated by gold + Terra rejection only.
- The orphan production run (`redfin_10965375`) and the 8 `redfin_126224899` leads remain outside the evidence; pinning them would need a separately approved baseline change.
- The bundle stores the queue's absolute `artifact_path` values verbatim; the parity test therefore depends on this machine's `C:/Users/Steven/IntelliJProjects/renointel-prod/artifacts` root.
- `reports/labels_v1_1.json` and the factorized manifest stay untracked; both are hash-pinned, so a change is caught, but a fresh clone lacks them.

## Required next task

Objective: Session 2 — semantic audit and proposal drafting per `docs/catalog_audit_program/02_SESSION_SEMANTIC_AUDIT.md`: cluster the 29 worklist items and 63 coverage questions by physical subject, mechanism, kind, scene, and unsupported commitment; review the photos behind every supporting unit; compare failures with the positive uses, agreements, correct rejections, and notes already joined per item and family; assign one primary outcome per cluster; render a deterministic proposal artifact.

Required inputs: `00_OVERALL_CONTEXT.md`, the Session 2 brief, this handoff, `reports/catalog_audit_evidence.json` (verify sha256 and fingerprint first), `scripts/migrate_catalog_kind_v2.py` and `tools/catalog_validation.py` for expressibility checks, and only the source evidence and photos referenced by promoted records (`indexes.cases[*].photo_keys` join to `reports/review_queue.json` card strips and the queue's `lineage.per_photo[*].image_path`).

Authorized changes: `reports/catalog_audit_proposals.json`, `scripts/render_catalog_audit_proposals.py`, `tests/test_catalog_audit_proposals.py`, `docs/PROPOSAL_catalog_audit_<date>.md`, `docs/catalog_audit_program/HANDOFF_SESSION_2.md`.

Prohibited changes: migration decisions, either catalog, generated migration artifacts, runtime code, prompts, the frozen evidence, and this session's bundle except through a documented builder correction that regenerates it.

Exit criteria: every cluster has exactly one primary outcome with photo-reviewed evidence, evidence-bar result from unique units (not raw rows), successes/counterexamples present or explicitly absent, native proposals shown expressible in memory, and the real decisions and catalogs unchanged.

## Do not redo

- Do not regenerate frozen evidence unless a new audit baseline is explicitly approved.
- Do not repeat completed joins or semantic reviews whose output hashes match this handoff.
- Do not rebuild the evidence bundle unless `--check` reports a fingerprint mismatch; if it does, treat it as drift and read the failing pin before touching anything.
- Do not re-derive the evidence-era catalog identity or the 3.1↔3.2 comparison; both are in the bundle.
- Do not probe the embeddings sidecar for current retrieval neighbors in Session 2; that is Session 5 Tier 2 work.
- Do not read `p2c_surviving[].catalog_item_id` as the 2d mapping; use `indexes.cases[*].condition_item` / `issues[*].resolved_item_id`.

## Suggested opening prompt for the next task

> Continue the catalog-audit program from this handoff. Read `docs/catalog_audit_program/00_OVERALL_CONTEXT.md`, `docs/catalog_audit_program/02_SESSION_SEMANTIC_AUDIT.md`, `docs/catalog_audit_program/HANDOFF_SESSION_1.md`, and `reports/catalog_audit_evidence.md`, then verify `reports/catalog_audit_evidence.json` (sha256 `43354104…29b7`, fingerprint `e4c0a2b8…0c08`) with `.venv\Scripts\python.exe scripts\build_catalog_audit_evidence.py --check`. Draft a task-specific implementation/review plan for the semantic audit. Do not repeat completed work or expand beyond Session 2's authority. End by producing `HANDOFF_SESSION_2.md`.
