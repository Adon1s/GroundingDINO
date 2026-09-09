# Handoff — catalog audit session 4

Date: 2026-09-06

Session/task title: Approved surgical implementation (`05_SESSION_APPROVED_IMPLEMENTATION.md`) — CAP-007 only

**Status: COMPLETE, with two deliberate test failures handed to Session 5** (see *Validation gate* and
*Unresolved questions and risks*). The approved diff is implemented exactly, the generated diff equals the manifest's
`expected_generated_changes` field for field, double generation is byte-identical, validators report zero errors, and
every other proposal record is untouched. Two suite failures are created by the approved change on purpose; repairing
either needs work Session 4 is not authorized to do, so neither was papered over.

## Outcome

| Approved id | Disposition | Implemented | Result |
|---|---|---|---|
| CAP-007 | `approved_with_modification` | yes, all seven ops, nothing else | generated diff equals `expected_generated_changes` |
| CAP-008 | `deferred` | no | untouched, as required |
| 18 others | `deferred` / `reclassified_non_catalog` | no | untouched |

`dated_window_treatment_valance` splits by subject into a presence-only blinds successor
(`window_blinds_basic_or_plain`, `route_override: no_action`) and the billable fabric successor
(`dated_window_valance_or_curtains`, gated by `require_any`). The catalog goes from 128 to 129 items.

## Repository state

- Baseline commit: `9afe0fa5a0490a856abed507dccc4a02ed1de24c` (proposal baseline `e9a7dc3`; the decisions file is
  byte-identical at both, so the approved diff applied unchanged)
- Candidate commit: **`6e67eaa83887cf4d218100dcd060c07d3bd30522`**, branch `catalog_audit_session4`
- Worktree: `C:/Users/Steven/PycharmProjects/rv-catalog-audit-s4` (prepared by the gate session; see its
  `WORKTREE_NOTES.md`)
- Dirty state at start: clean (`git status --porcelain -uno` empty). At end: clean; everything is in the commit.
- Main checkout `C:/Users/Steven/PycharmProjects/realtorvision-backend` was **not modified**. It stays on
  `terra_factorized_verifier` at `9afe0fa` with no tracked changes, and serves as the Session 5 baseline arm.
- Pre-existing changes preserved: yes. All 16 protected artifacts re-hashed at the end and byte-identical (table below).

### The untracked-artifacts question (gate handoff risk 1) — resolved, not re-decided

The gate session had already answered it: it cut this worktree and **copied in** the 32 untracked program inputs,
because a worktree materialises only tracked files. This session verified every copy is hash-identical to the main
checkout and then treated all of them as **read-only inputs**. Nothing was committed that was not already tracked or
authored here. Whether the program artifacts should be committed at all remains delegated to
`docs/HANDOFF_git_repository_audit.md`; Session 4 did not pre-empt it.

Consequence Session 5 must know: `HANDOFF_SESSION_4.md` (this file) and the follow-up register live in the worktree as
untracked copies **and** were copied back to the main checkout. Neither location is committed. See *Outputs*.

## Inputs verified

Every pin in the manifest's `approved_against`, verified inside the worktree before any edit.

| Input | Expected | Actual | Result |
|---|---|---|---|
| HEAD | `9afe0fa` | `9afe0fa5a0490a856abed507dccc4a02ed1de24c` | match |
| `reports/catalog_audit_approvals.json` | `a725c89a…c871` | `a725c89af131e94b04867f4f5a226bda495278a8ba28fdd7162ed6188d76c871` | match |
| `reports/catalog_audit_redraft.json` | `449f8617…811b` | `449f86170dd4b01916988b2000b900e560403bda12fb93cc7e7d7d27bb31811b` | match |
| `reports/catalog_audit_renewed_review.json` | `aa12e42a…538b` | `aa12e42a0205c0ede8c01362e15f3c16e47cffd9b37de8dd468e70400e8b538b` | match |
| `reports/catalog_audit_gate_decisions.json` | `d89b2bf1…c980` | `d89b2bf1149738d9b3e2d5a9222a5b4e163e2cf97dbb0c9f6396d595b849c980` | match |
| `reports/catalog_audit_live_check_results.json` | `de47d13f…74e1` | `de47d13f1dfa5802e885027520ae318ab6a6166e8b609115eb2bdb60eaec74e1` | match |
| `reports/catalog_audit_proposals.json` | `1411352c…3ef99` | `1411352c42e9d3a555c3a1561470323da2234447badc2869c9874c3e8fa3ef99` | match |
| `reports/catalog_audit_adversarial_review.json` | `d1149875…b5c4` | `d1149875399a9f30c6cd56e924731c7e207d90b603cb248f402c692c3438b5c4` | match |
| `reports/catalog_audit_evidence.json` | `43354104…29b7` | `43354104cc109c768efbe2d2f8b9a825f33e0882210b81e9f12b847ca67e29b7` | match |
| `tools/catalog_migrations/kind_v2_decisions.json` | `47614d82…03e8` | `47614d822bdd6b672d8596050b46839d04a8a6df0a04c18e052f4a63bd1903e8` | match |
| `tools/issue_catalog_kind_v2.json` | `51bf7e26…aa54` | `51bf7e263ff98109d657ef730b0ce1a705598101f09843eb288b1bdc3e0eaa54` | match |
| `tools/issue_catalog.json` | `4ba046a1…14f2` | `4ba046a1a78337f1c8e47701a011ec16e700c296782ddedbe7bf52cf888314f2` | match |
| `scripts/migrate_catalog_kind_v2.py` (generator) | `b732022f…a053` | `b732022fcea8cd22109c7781eb0f93d0688d557230c759ccb6216d667250a053` | match, **unchanged by this session** |
| `tools/catalog_validation.py` (validator) | `041637a3…bee2` | `041637a3684bacd3bc06dd92955b845088b4fedda8bb7e2e31222ac9ec75bee2` | match, **unchanged by this session** |
| photo review / packet / register / HANDOFF_SESSION_3 / both renderers | as pinned | as pinned | match |

Two gates were run before anything was edited, both hard-stop conditions:

1. **Redraft versus manifest.** `reports/catalog_audit_redraft.json` `cap007.ops` compared element by element against
   `dispositions[CAP-007].approved_diff.ops`. **Equal.** Had they differed the session would have stopped without
   editing; two hash-pinned artifacts disagreeing is an input or tooling problem, not something to arbitrate here.
2. **Baseline reproduction.** The generator was run once on the unchanged decisions file. All three outputs came back
   byte-identical and `git status --porcelain -uno` stayed empty, proving this environment reproduces the pinned CRLF
   bytes before being trusted to produce new ones.

## Files created or changed

Eight files, all in commit `6e67eaa`. On-disk hashes are of the CRLF working tree (the value the program pins);
blob hashes are the LF objects.

| Path | Status | on-disk sha256 | git blob |
|---|---|---|---|
| `tools/catalog_migrations/kind_v2_decisions.json` | modified (one entry) | `2e49a82f9961c2623342ec84743a0997299fd50f0dfcc7939196029cf19b9c76` | `f77c46d6` |
| `tools/issue_catalog_kind_v2.json` | regenerated | `43d8147ee8abd4f4df51bcf88a83ba99d834c415b9d5d28295fa0a33f3457acf` | `4413f5db` |
| `tools/catalog_migrations/2.1_to_3.0.json` | regenerated | `6b944b83fee7b350bdc7183d3def8f5849d3b97ca4d017c7dcc4efcd64936a18` | `c8e690ff` |
| `tools/catalog_migrations/2.1_to_3.0_audit.md` | regenerated | `cac2750a20799b837293c04b7afd098fb071f8326c1f9edcc41e1dc1ed0e8a81` | `4aea1912` |
| `tests/test_catalog_cap007_window_treatment_split.py` | **new**, 24 tests | `867eb0e27320a6b79337e1e6b2c2472b71b613f8d05c274cc5d9407d6e7769c4` | `e3460c7b` |
| `tests/test_catalog_kind_v2.py` | pin update | `e803ed213cbb1481dcbe5f4086d3b8ba8e05ded5151427ee7de17be8ca341104` | `e67a7596` |
| `tests/test_renovation_architecture_catalog.py` | pin update | `2beeae1b4dfa02d0a20044676aeb4e5cd080bff92c334bcae3b02b2844be214b` | `f5750867` |
| `tests/test_kind_ontology_guards.py` | pin update | `271d1d10e9ff4091d641899928309be0c8dfa764e16d6259f8d19aee0ccfa7aa` | `34e24197` |

Pre-existing user changes: none were touched. The 32 copied program artifacts, the four copied run artifacts and
`WORKTREE_NOTES.md` remain untracked and byte-identical.

### Protected artifacts, re-verified at the end

`catalog_audit_approvals` `a725c89a…`, `redraft` `449f8617…`, `renewed_review` `aa12e42a…`, `gate_decisions`
`d89b2bf1…`, `live_check_results` `de47d13f…`, `evidence` `43354104…`, `photo_review` `2225db9c…`,
`photo_review_packet` `7b4621df…`, `proposals` `1411352c…`, `adversarial_review` `d1149875…`,
`PROPOSAL_…20260901.md` `771572e7…`, `REVIEW_…20260905.md` `538f2872…`, `FOLLOW_UP_REGISTER.md` `268cd370…`,
`HANDOFF_SESSION_3.md` `fbaa1bb4…`, `HANDOFF_SESSION_GATE.md` `9cf8f815…`, `issue_catalog.json` `4ba046a1…`,
generator `b732022f…`, validator `041637a3…`, both renderers `d0305682…` / `b5afbed9…`. **All unchanged.**

## Work completed

- Verified HEAD, all `approved_against` pins and the two hard gates above.
- Applied the seven ops with the renderer's own `apply_ops()`, reading them straight from the approvals manifest.
  Nothing was retyped. The writer refused unless: top-level sections unchanged, entry order and count unchanged, all
  **106 unrelated entries byte-identical**, entry key order preserved, the renderer's `metadata_problems()` empty, and
  every op classified `native`. All held; the file was written with the generator's own serialisation
  (`json.dumps(indent=2, ensure_ascii=False) + "\n"`, CRLF) and round-trips exactly.
- Regenerated twice; compared hashes.
- Proved the generated diff against the manifest with 61 mechanical checks (below).
- Added the focused test module; updated three existing count pins.
- Ran the full suite in the candidate, then reverted the worktree to baseline and ran it again, so every failure is
  classified against the *same* environment rather than against the main checkout.
- Restored the candidate state by re-running the apply script and the generator from the reverted baseline and
  confirmed all four production hashes came back identical — an independent replay of the whole pipeline.

## Decisions and invariants

- **The manifest is the only authorization.** Only CAP-007's `approved_diff.ops` were applied, to the single entry
  `dated_window_treatment_valance` (index 67 of 107). CAP-008 and the other 18 records were not read for content and
  not touched. No claim text was tidied (A-7 keeps them as drafted).
- **No generator or validator change.** Both hash unchanged. The split needed none: `route_override`, `require_any`
  and `deny_any` are all successor-authorable (`INHERITED_FIELDS`), and the parent is not one of the ten
  `repair_support_when_driven` marker items, so CCF-7 did not block it.
- **Economics are inherited, not authored.** Both successors carry the v1 parent's `cost {mode: heuristic}`,
  `work_item_code WINDOW_TREATMENT_UPDATE` and `package_role ignore` byte-identically, stamped
  `pricing_status: inherited_from_split_parent`; absent fields stay absent. On the blinds successor those fields are
  inert because projection applies `route_override` before the costing route, **and they are what keeps the override
  from being rejected as redundant** by `tools/catalog_validation.py` — do not "clean them up".
- **First same-kind split in the decisions file.** Both successors stay `modernization`, so `estimate_scope` text
  matching and the package-category vocabulary behave exactly as they did for the parent.
- **`00_OVERALL_CONTEXT.md` §3's decision table is now stale.** It reads 50 reclassified / 19 split against the
  proposal baseline; the candidate is **49 reclassified / 20 split**. The document already instructs a re-count
  whenever the decisions hash drifts. It was not edited here (not an authorized file).

## Deviations from the session brief

| Deviation | Reason | Impact |
|---|---|---|
| Three existing test modules updated, not the two the plan named | `tests/test_kind_ontology_guards.py` pins the deprecated-legacy-id count (21), which the split moves to 22. Found during validation, not planned. | Same class as the other pins: an arithmetic consequence of one item becoming two, in a file whose job is to pin it |
| Two suite failures left unfixed | Repairing either requires work outside Session 4's authority (benchmark gold authorship; a stored-artifact replay). Hiding them with a skip or xfail would conceal a real, load-bearing consequence of the approved change. | The candidate suite is **not green**. Both are fully diagnosed below and in the register |
| A "Session 4" block appended to `FOLLOW_UP_REGISTER.md` §9 | The register's §9 explicitly invites it; it is not in the brief's authorized file list | Recorded here; the register is untracked in both locations |

## Commands and verification

| Command / check | Result |
|---|---|
| redraft ops == manifest ops | **equal** (7 ops, one legacy id, all `set`) |
| baseline regeneration in the worktree | three outputs byte-identical, `git status -uno` empty |
| `apply_cap007.py` (renderer `apply_ops`) | 7/7 ops `native`, 0 metadata problems, 106/106 unrelated entries identical, 3551 CRLF lines / 0 lone LF |
| `scripts\migrate_catalog_kind_v2.py` run 1 | `129 items {defect 52, degradation 36, modernization 41}`, 107 manifest entries |
| `scripts\migrate_catalog_kind_v2.py` run 2 | **byte-identical to run 1** on all three outputs |
| full pipeline replayed from a reverted baseline | all four production hashes reproduced exactly |
| generated-diff proof | **61/61 checks passed** |
| `validate_issue_catalog` (candidate) | **0 errors**; warning set identical to baseline |
| `validate_migration_manifest` (candidate) | **0 errors** |
| Must-pass modules (catalog, kind-v2, validation, embeddings, projection, runtime, ontology guards/cutover, packages, new CAP-007 module) | **549 passed, 1 skipped** |
| `tests/test_catalog_cap007_window_treatment_split.py` | **24 passed** |
| Full suite, candidate worktree | **2759 passed, 6 failed, 21 skipped** |
| Full suite, same worktree reverted to baseline | **2740 passed, 1 failed, 21 skipped** |
| Full suite, main checkout at baseline | **2756 passed, 6 skipped, 0 failed** |
| Both renderers `--check`, **baseline**, worktree and main checkout | exit **0**, json + markdown match, validation ok |
| Both renderers `--check`, **candidate** | exit **1**, expected pin drift only; both wrote nothing (all four guarded artifacts re-hashed unchanged) |

### The generated diff equals the approved diff

```
added:              ['dated_window_valance_or_curtains', 'window_blinds_basic_or_plain']
removed:            ['dated_window_treatment_valance']
item_diff:          {}          <- no other catalog item changed in any field
order_ok:           True
manifest_diff_keys: ['dated_window_treatment_valance']
```

identical to `expected_generated_changes`. The proof also re-ran the renderer's `dry_run()` from the baseline and
matched `dry_run_at_approval` exactly: same seven `native` classifications, empty `metadata_problems`, empty validator
error lists, and identical `added_detail` economics for both successors. It further checks that the in-memory dry run
agrees with what generation actually wrote, that both successors' authored surfaces (`id`, `name`, `description`,
`embed_text`, `support_any`, `atomic_claim`, and every override) landed verbatim, and that root metadata, trade
buckets and the manifest's `package_routing` block are untouched.

### Semantic and economic diff, CAP-007

| Aspect | Before | After |
|---|---|---|
| Entry | `reclassified`, `deprecated: false`, one same-id successor | `split`, `deprecated: true`, `requires_re_resolution: true`, two successors |
| Items | `dated_window_treatment_valance` (modernization, sev 1) | `window_blinds_basic_or_plain` + `dated_window_valance_or_curtains`, both modernization sev 1 |
| Claim | `window treatments \| dated valance or treatments \| dated_style` | blinds: `window blinds or shades \| basic or plain window blinds \| presentation_only`; fabric: `window valance or curtains \| dated valance or curtains \| dated_style` |
| Retrieval | `support_any` 6 terms, `deny_any` 3 glazing terms, no `require_any` | blinds: 7 blind/shade support terms, 11 deny terms (fabric + glazing), no `require_any`; fabric: 8 support terms, 8-term `require_any`, 8 deny terms ending `shower`, `tub` (A-3) |
| Routing | `work` | blinds `no_action` (`route_override_no_action`); fabric `work`, action code `WINDOW_TREATMENT_UPDATE` |
| Economics | cost heuristic, `WINDOW_TREATMENT_UPDATE`, `package_role ignore` | **unchanged**, inherited verbatim by both successors |
| Packages | flat `ignore`, no affinity, not a marker item | unchanged on both; no package metadata added |
| Display | `defaultHidden false`, `display_class marketability` | unchanged on both (A-4: the blinds successor stays visible) |

Generated audit report totals moved accordingly: v2 items 128 → 129 (modernization 40 → 41), dispositions
50 reclassified/19 split → 49/20, inherited-economics successors 42 → 44.

### Validation gate — every failure classified in one environment

Baseline and candidate were both run **inside the same worktree**, so the comparison is exact. Of the 2741 tests
selected at worktree baseline, exactly **five moved from pass to fail**; the new module adds 24, all passing
(2740 − 5 + 24 = 2759 ✓).

| Test | Class | Verdict |
|---|---|---|
| `test_catalog_audit_proposals::test_parity_committed_proposals` | **expected baseline-pin failure** | Accepted. Message is exactly the decisions/catalog pin drift, naming both paths and both hash pairs. Passes at baseline in this worktree and in the main checkout. |
| `test_catalog_audit_adversarial_review::test_parity_committed_review` | **expected baseline-pin failure** | Accepted, same message. |
| `test_catalog_audit_adversarial_review::test_build_is_read_only_and_pins_verify` | **expected baseline-pin failure** | Accepted, same message. |
| `test_review_analysis::test_full_run_against_frozen_inputs` | **pre-existing, environment** | **Not caused by this change.** Fails identically with the production files reverted to baseline in this worktree; passes in the main checkout. Cause is the worktree's incomplete untracked inputs (`KeyError: ('canary', 'redfin_11185681')` at `scripts/review_analysis.py:622` — the run-artifact context lacks that property). |
| `test_catalog_resolution_benchmark::test_every_split_successor_is_gold_somewhere` | **candidate-caused, out of scope** | Left failing. See risk 1. |
| `test_benchmark_pass2a::test_compute_pre2f_totals_force_confirms_packages` | **candidate-caused, out of scope** | Left failing. See risk 2. |

No other failure appeared, no test changed from pass to skip (21 skipped in both worktree arms), and every must-pass
module in the plan's gate is green.

## Outputs for the next session

| Artifact | Identity | Contract |
|---|---|---|
| Candidate commit | `6e67eaa83887cf4d218100dcd060c07d3bd30522` on `catalog_audit_session4` | the candidate arm; eight files, listed above |
| Baseline commit | `9afe0fa5a0490a856abed507dccc4a02ed1de24c` | the baseline arm; the main checkout is already exactly this, clean |
| `tools/issue_catalog_kind_v2.json` | on-disk `43d8147e…7acf` | candidate catalog, version 3.2, 129 items, `publishable` |
| `tools/catalog_migrations/kind_v2_decisions.json` | on-disk `2e49a82f…9c76` | candidate decisions, 107 entries, 20 split |
| `docs/catalog_audit_program/HANDOFF_SESSION_4.md` | this file | in the worktree **and** copied to the main checkout; untracked in both |
| `docs/catalog_audit_program/FOLLOW_UP_REGISTER.md` | was `268cd370…3203`, now **`b0ecd66f93ed8ccdf2734453365950fab211b4366324873b636402189d2e9b2d`** | §9 gained a Session 4 block (five rows, S4-1…S4-5); same in both locations |

**Register pin superseded.** The gate handoff pins the register at `268cd370…3203`. Appending to §9 changes that
value to `b0ecd66f…9b2d`. This is intended: the register's own §7 and §9 say Sessions 4–6 append to it, so the gate's
hash is a snapshot of the gate, not a frozen artifact. Every genuinely frozen artifact — proposals, adversarial
review, evidence, photo review, packet, gate decisions, redraft, renewed review, live check, approvals — is
byte-identical, as re-verified above. A future session checking pins should expect exactly this one difference and no
other.

### Session 5 arm instructions

- **Baseline arm:** the main checkout at `9afe0fa`, already clean and complete (it has the untracked program inputs and
  the run-artifact trees). Do not regenerate anything there.
- **Candidate arm:** this worktree at `6e67eaa`, or a fresh worktree of `catalog_audit_session4`. A fresh worktree will
  **not** contain the untracked program artifacts or run-artifact trees; copy them as the gate session did, or expect
  the same environment-only failures noted above.
- Do **not** set `ISSUE_CATALOG_PATH`; under `observation_kind_v2` it is a hard configuration error. Use normal
  decisions-file generation per arm.
- Regenerating in the candidate arm must reproduce the three hashes in the table above exactly.

## Carried approval conditions — obligations, not just hashes

These come from `dispositions[CAP-007].conditions` and register §3. They are restated here in full because hashing the
source documents makes them easy to skip.

**Session 5 must measure (Tier 2, provider-free):**

1. The **lexical-shortcut fire rate** on blinds observations against `window_blinds_basic_or_plain`
   (`scene_classifier_passes.py:1096-1104` thresholds). The split creates a new shortcut surface.
2. **Post-split ranks against `dated_or_older_windows`** on "windows and blinds" bullets — the two items now compete
   where one previously stood.
3. The **new shortcut surface on the fabric successor's bare stems** (`curtain`, `drape`, `fabric`). Roughly a dozen
   evidence-era observations hit these stems and several are currently owned by other items; this is the neighbour
   hijacking risk the approval flagged.
4. The **embedding neighbourhood of both successors**. No offline store exists, so this needs the sidecar — probe it
   with a real POST, not `/health`.

**Sessions 5 and 6 must plan a full replay and an explicit cutover (CCF-8):**

- Both successors carry **new `atomic_claim`s**, which change the projection fingerprint, so **no stored Terra
  checkpoint survives**. Replay; never reuse.
- The deprecated parent has **no runtime alias**. A resumed run carrying `dated_window_treatment_valance` **fails
  hard** — this is intended, and is exactly what risk 2 below demonstrates.
- `backfill_kind_v2` is a no-op at the target version, so it cannot migrate stored artifacts for you.

**Accepted outcomes to preserve, not "fix":**

- **A-4 display.** The blinds successor stays visible (`defaultHidden` false, `display_class` marketability). The
  `no_action` route is the billing mechanism, not hiding. Session 6's baseline arm should report how many `no_action`
  blinds conditions render at zero dollars (about 28 evidence-era conditions move).
- **A-5 curtain hardware.** "The curtain rods are dated." (`oc1_649033275db36ca2`) keeps billing on the fabric
  successor for now. Its Terra verdict under the new claim is a Session 6 observation; the hardware-subject question
  goes to the catalog-wide ruling-3 sweep.
- **A-6 improvised coverings.** The one billed makeshift-covering condition (`oc1_c839d6e1118a7006`,
  `redfin_81000709` photo_029) reaches neither successor **by design** under Q-2. Accepted as undetected rather than
  `no_action`.
- **A-7 claim texts** kept as drafted. The tighter states ("basic or plain style, functional"; "dated style,
  functional") are an optional future change needing its own check. Not a tidy-up for Session 5.
- **Three predicted Terra flips** to `supported` on the presence-only claim (`rc_4f4b93c40ef3`, `rc_ffad088fa5fe`,
  `rc_a22a241b3bf0`) stay unmeasured; they are unbillable on every path.
- **Withdrawn billings accepted under Q-1:** `rc_7c5b9f9bb3aa` (v1.1 claim exact, work warranted) and gold g3 →
  `oc1_12e3660bf3923583`, plus 26 further evidence-era accepted conditions moving to `no_action` and one reaching
  neither successor.

## Unresolved questions and risks

1. **The catalog-resolution benchmark no longer scores two live items.**
   `test_catalog_resolution_benchmark::test_every_split_successor_is_gold_somewhere` fails with
   `never scored: ['dated_window_valance_or_curtains', 'window_blinds_basic_or_plain']`. No existing gold case
   referenced the removed parent (`test_shipped_cases_reference_real_catalog_items` still passes), so nothing is
   broken — the benchmark simply never covered this concept and now owes two cases. Authoring them means choosing
   observation text, kind and scene group for brand-new items, which is evaluation content, and `benchmarks/` is not
   in Session 4's authorized file list. **Owner: Session 5**, whose Tier 2 replay produces exactly the evidence needed
   to author them well. Until then the module is red.
2. **A stored canary artifact cannot be recomputed offline against the candidate catalog.**
   `test_benchmark_pass2a::test_compute_pre2f_totals_force_confirms_packages` fails with
   `ValueError: renovation_estimate: issue resolves to unknown/stale catalog id 'dated_window_treatment_valance',
   absent from the selected catalog (version '3.2')` (`tools/renovation_estimate.py:501`), on
   `artifacts_canary/candidate/redfin_11000447/20260808_085243_19c4a4d0`. **This is CCF-8 working as designed** — the
   approval states stale ids must fail hard. The fix is a replay, not an alias and not an edit to a frozen artifact.
   **Owner: Sessions 5/6.** Note the scope: *any* stored 3.2 artifact naming this item now fails offline recompute, so
   this test is a useful canary for the cutover.
3. **The register's stale Pass 2d worklist row.** `window_unit_vs_treatment` still names
   `dated_window_treatment_valance` as the better item. Deliberately **not** re-pointed here (the register is not an
   authorized file and the gate handoff asked for it after Session 4 lands). It should point at
   `window_blinds_basic_or_plain`, which makes the follow-up a naming question rather than a billing one.
4. **`00_OVERALL_CONTEXT.md` §3 decision counts are stale** (see *Decisions and invariants*).
5. **Nothing is pushed and nothing outside the candidate commit is tracked.** The branch is local-only, as is the whole
   program. This handoff and the register exist only as untracked copies in two places.
6. **Session 6 timing is unchanged from the gate's advice:** one approved change is a thin payload for a
   ~4M-token replica. Sessions 4 and 5 are cheap; hold Session 6 until more changes accumulate or the Catalog 3.2
   acceptance data is wanted.

## Do not redo

- Do not re-apply the ops or re-run the generator expecting a different result; the pipeline was replayed from a
  reverted baseline and reproduced all four hashes exactly.
- Do not "fix" the three audit-renderer test failures or the renderers' `--check` in the candidate arm. They pin the
  **baseline** decisions and catalog hashes by design and must keep failing there; they pass at baseline, which is the
  check that matters.
- Do not treat `test_review_analysis::test_full_run_against_frozen_inputs` as a regression; it fails at baseline in the
  worktree for want of untracked inputs.
- Do not add a runtime alias for `dated_window_treatment_valance`, and do not edit a stored artifact to make risk 2 go
  away.
- Do not tidy the claim texts (A-7), add `defaultHidden` to the blinds successor (A-4), or remove the inherited
  economics from it (they keep the `route_override` valid).
- Do not touch CAP-008 or any other manifest record.

## Required next task

Objective: Session 5 per `docs/catalog_audit_program/06_SESSION_DETERMINISTIC_VALIDATION.md` — independent Tier 1
deterministic and Tier 2 retrieval validation of candidate `6e67eaa` against baseline `9afe0fa`, ending in a pinned
live-experiment manifest if both tiers pass.

Required inputs: `00_OVERALL_CONTEXT.md`, the Session 5 brief, this handoff, `reports/catalog_audit_approvals.json`
(`a725c89a…c871`), `reports/catalog_audit_proposals.json`, `reports/catalog_audit_evidence.json`, and both commits.

Authorized changes: `reports/catalog_audit_validation_tier12.json`,
`docs/RESULT_catalog_audit_validation_tier12_<date>.md`, `reports/catalog_audit_live_experiment_manifest.json`,
minimal focused validation tooling/tests only if existing comparators cannot express the arm comparison, and
`HANDOFF_SESSION_5.md`.

Prohibited: changing the approved decisions or generated artifacts; prompts, model routes, Pass 2a–2e, Terra logic,
product policy, pricing, packages; live provider evaluation; publication or cutover. If validation finds a semantic
implementation defect, fail the candidate and return it — do not fix it inside the validation task.

Exit criteria: explicit Tier 1 and Tier 2 pass/fail; actual diff reconciled against the approved diff (this handoff's
proof is reproducible); all affected, support and control cases reconciled; no live provider call; a fully pinned live
manifest only if the candidate passed; and a decision recorded on risks 1 and 2 above.

## Suggested opening prompt for the next task

> Execute Session 5 of the catalog-audit program per `docs/catalog_audit_program/06_SESSION_DETERMINISTIC_VALIDATION.md`.
> Read `00_OVERALL_CONTEXT.md`, that brief, `HANDOFF_SESSION_4.md`, and the hash-verified
> `reports/catalog_audit_approvals.json`. Baseline arm is the main checkout at `9afe0fa`; candidate arm is
> `6e67eaa` on `catalog_audit_session4` in `C:/Users/Steven/PycharmProjects/rv-catalog-audit-s4`. Verify both arms and
> every hash in the Session 4 handoff, then write a validation plan. Do Tier 1 deterministic and Tier 2 provider-free
> retrieval validation only — probe the embeddings sidecar with a real POST, and measure the four Tier 2 items the
> approval requires (blinds lexical-shortcut fire rate, post-split ranks against `dated_or_older_windows`, the fabric
> successor's bare-stem shortcut surface, both successors' embedding neighbourhoods). Do not set `ISSUE_CATALOG_PATH`,
> do not change approved semantics, and do not run live providers. Decide what to do about the two known failures the
> handoff hands you (benchmark gold coverage for the new successors; the CCF-8 stored-artifact replay). Finish with a
> pinned live-experiment manifest if the candidate passes, and `HANDOFF_SESSION_5.md`.
