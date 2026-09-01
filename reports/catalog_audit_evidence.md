# Catalog audit evidence bundle (Session 1)

Schema 1 · fingerprint `e4c0a2b816395782cd84bc4abcdf6207d131a34a0e2d63eaebb39bec4c190c08` · starting commit `48be33a5d417f260c0eadfd1a055d7e99d196976` on `terra_factorized_verifier` (git block is not fingerprinted).

## Provenance

| Source | Tier | SHA-256 | Pinned |
|---|---|---|---|
| `scripts/build_catalog_audit_evidence.py` | recorded | `8874bc340b93c48ef0014bbf2529d750951709d70afb2bd6016c10d0e6deaf78` | recorded |
| `tools/issue_catalog.json` | recorded | `4ba046a1a78337f1c8e47701a011ec16e700c296782ddedbe7bf52cf888314f2` | recorded |
| `tools/issue_catalog_kind_v2.json` | frozen | `51bf7e263ff98109d657ef730b0ce1a705598101f09843eb288b1bdc3e0eaa54` | match |
| `tools/catalog_migrations/kind_v2_decisions.json` | frozen | `47614d822bdd6b672d8596050b46839d04a8a6df0a04c18e052f4a63bd1903e8` | match |
| `artifacts_canary/factorized_v1_20260831/manifest.json` | recorded | `89f8fccdf2769bdff0db404ffc1ed771328ba6ecc013c7bcd81a7b5463b3a4d0` | recorded |
| `docs/FINDINGS_catalog_3_2_deferred_issues.md` | recorded | `23d336fa55fbd90a7803106ff08506947de289b16fdb720e8f41085f2858f985` | recorded |
| `scripts/migrate_catalog_kind_v2.py` | recorded | `b732022fcea8cd22109c7781eb0f93d0688d557230c759ccb6216d667250a053` | recorded |
| `reports/error_attribution_gold_cases.json` | frozen | `77a7b515c594eb862bfc9ffbd0073556bc8a44f2fed644610f2412cb3bd0954a` | match |
| `benchmarks/pass2a-prompt/gold/reference.json` | frozen | `253075132988ac2a1abbdb6e0815c1bd75e89d9ad982e745ddb9ce5a9f424543` | match |
| `reports/labels_v1_1.json` | frozen | `7f9b03017195144195550074b49dbb0488ed3ad33fdde905ba4e5867dfdbe3bd` | match |
| `tools/catalog_migrations/2.1_to_3.0.json` | recorded | `3db144c907ce91b0c26d75b85c4136db38dd81da8505c8405f43cca246893666` | recorded |
| `reports/error_attribution_queue.json` | frozen | `b58dcca7c3779f2b0539f988077f865c6c1ca040d23a44646d89b04a398940df` | match |
| `reports/review_analysis.json` | frozen | `3652ba3c3feb15800c064ffafd076d15cd9773de29c3d77c7b383fe26a259c9a` | match |
| `reports/review_queue.json` | frozen | `8512b87c2af18d1104069c15a5eda977d5fe79eb736f9aedd976d174f89d318a` | match |
| `reports/review_verdicts.jsonl` | frozen | `0c7ca6a8b55d6dac69021a00e81315be1b15cef68ffd59fbe2973cb895a8d70f` | match |
| `reports/factorized_review_scorecard.json` | frozen | `60300ec5e957e42d2e443ab9493d0b9820b66498a2f57e3ac34a4479fcd9d103` | match |
| `tools/catalog_validation.py` | recorded | `041637a3684bacd3bc06dd92955b845088b4fedda8bb7e2e31222ac9ec75bee2` | recorded |
| `reports/error_attribution_verdicts.jsonl` | frozen | `166e8bec20641c0a8fb0ca5ddbd6c042f31b828447304ccbbbd4d585cd2a894e` | match |

Run artifacts: canary 17 pinned / 0 unpinned, production 8 pinned / 1 unpinned; all pinned artifacts re-hashed and verified; every pinned artifact records catalog_sha256 `d1cef743f379918fa17f0684c245779180d7d692ebc8fd3e66b7c69bebf68347`.

## Catalog identity

- Evidence era: version 3.1 at `07ee112` — blob (LF) `20adb6e342f2d3a5304e90ae6dae7d8bf1724c53feab22358130081582abfd90`, checkout (CRLF) `d1cef743f379918fa17f0684c245779180d7d692ebc8fd3e66b7c69bebf68347`, matched via crlf (checkout convention: crlf).
- Proposal baseline: version 3.2 at `a9ed7ad` — on-disk `51bf7e263ff98109d657ef730b0ce1a705598101f09843eb288b1bdc3e0eaa54`, blob (LF) `53a7b8eaa822b3d6a77254591849d98c9902633a2df0dba3c8c6eb5233c3e11d`, working tree matches HEAD: True.
- Git history examined: 6 commits of the generated catalog.

## Baseline comparison (3.1 → 3.2)

- Items: 128 → 128; added [], removed [], order changed: False, trade buckets changed: False.
- atomic_claim unchanged: **True**; claim_text unchanged: **True**.
- Per-field change counts: {'package_affinity': 10}.
- Changed items (10): `peeling_or_discolored_paint`, `worn_or_stained_carpet`, `vinyl_linoleum_worn_or_stained`, `cabinets_worn_finish`, `appliances_worn_or_neglected`, `baseboard_wear_scuffs`, `wall_scuffs_marks_or_dents`, `vanity_countertop_worn`, `vanity_countertop_dated`, `worn_or_stained_flooring`.
- Repair-support markers added by 3.2: 24; product-quarantined trades: {'evidence_era': ['electrical'], 'proposal_baseline': ['electrical']}.
- Decision counts: {'narrowed': 4, 'reclassified': 50, 'retired': 2, 'split': 19, 'unchanged': 32}.

## Seeds and evidence units

| Rule | Records | Reason |
|---|---:|---|
| R1 | 20 | latest effective attribution is downstream at stage 2d |
| R2 | 8 | queue lane appendix_misnamed (human: claim misnamed, work warranted) |
| R3 | 21 | miss lane with latest attribution downstream at stage terra |
| R4 | 50 | factorized verifier answered claim_accurate_as_written=no (model lead, not truth) |
| R5 | 40 | gold finding with no matching v5 condition (gold miss candidate) |
| R6 | 27 | gold finding the review judged outside the catalog (open coverage question) |
| R7 | 11 | previously deferred catalog finding (backlog/constraint reference, not evidence) |

Records before dedup: 162 (177 rule/record pairs; by type {'case': 34, 'deferred_finding': 11, 'factorized_lead': 50, 'gold_case': 40, 'gold_row': 27}). After dedup: 140 units ({'gold': 67, 'runtime': 73}), 140 independent, 0 corroborating, 77 attached to an item; record roles {'independent': 140, 'method_corroboration': 3}; unattached records {'backlog_reference': 11, 'run_identity_unpinned': 8}. Every seed reconciled once: True.

## Worklist (counts only; no evidence-bar verdict)

| Item | Lane | Units | Indep. | Props | Corrob. | Pos. uses | Agree | Correct rej. | Notes | 2a halluc. | Family | Rules |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `baseboard_wear_scuffs` | catalog_candidate | 2 | 2 | 2 | 0 | 3 | 0 | 3 | 1 | 0 | 39 | R3 |
| `bath_fixtures_stained_or_worn` | catalog_candidate | 4 | 4 | 4 | 0 | 3 | 0 | 0 | 1 | 0 | 30 | R1 R2 R4 |
| `boarded_up_entry_or_window` | catalog_candidate | 1 | 1 | 1 | 0 | 0 | 0 | 0 | 1 | 0 | 2 | R1 R2 |
| `brick_weathered_or_discolored` | catalog_candidate | 2 | 2 | 2 | 0 | 1 | 0 | 1 | 0 | 0 | 11 | R3 |
| `cabinets_damaged_or_water_stained` | catalog_candidate | 1 | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 27 | R3 |
| `cabinets_worn_finish` | catalog_candidate | 1 | 1 | 1 | 0 | 0 | 0 | 2 | 0 | 0 | 27 | R4 |
| `damaged_drywall_or_cracks` | catalog_candidate | 1 | 1 | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 42 | R4 |
| `damaged_or_rotted_siding_or_trim` | catalog_candidate | 1 | 1 | 1 | 0 | 1 | 0 | 1 | 1 | 0 | 12 | R3 |
| `dated_interior_trim` | catalog_candidate | 2 | 2 | 2 | 0 | 4 | 0 | 1 | 1 | 2 | 41 | R1 R3 R4 |
| `dated_or_older_windows` | catalog_candidate | 2 | 2 | 1 | 0 | 0 | 0 | 1 | 0 | 0 | 5 | R5 |
| `dated_wallpaper_present` | catalog_candidate | 1 | 1 | 1 | 0 | 0 | 0 | 0 | 1 | 0 | 29 | R1 |
| `dated_window_treatment_valance` | catalog_candidate | 2 | 2 | 2 | 0 | 1 | 0 | 2 | 0 | 1 | 5 | R1 R4 |
| `exterior_siding_discoloration_fading` | catalog_candidate | 2 | 2 | 2 | 0 | 1 | 0 | 0 | 0 | 0 | 12 | R3 |
| `fence_weathered` | catalog_candidate | 1 | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 7 | R4 |
| `hard_flooring_broken_or_warped` | catalog_candidate | 1 | 1 | 1 | 0 | 1 | 1 | 1 | 1 | 0 | 32 | R4 |
| `hard_flooring_scratched_or_worn` | catalog_candidate | 3 | 3 | 3 | 0 | 3 | 0 | 2 | 1 | 1 | 32 | R1 R2 R3 R4 |
| `older_flooring_style` | catalog_candidate | 12 | 12 | 7 | 0 | 2 | 0 | 1 | 2 | 1 | 32 | R4 |
| `paint_refresh_recommended` | catalog_candidate | 4 | 4 | 4 | 0 | 4 | 1 | 3 | 0 | 0 | 32 | R3 |
| `patio_or_porch_surface_wear` | catalog_candidate | 2 | 2 | 2 | 0 | 3 | 0 | 0 | 0 | 0 | 10 | R1 R2 |
| `peeling_or_discolored_paint` | catalog_candidate | 5 | 5 | 5 | 0 | 3 | 0 | 2 | 4 | 1 | 31 | R1 R2 R3 R4 |
| `popcorn_or_acoustic_ceiling_texture` | catalog_candidate | 1 | 1 | 1 | 0 | 1 | 0 | 1 | 1 | 0 | 37 | R4 |
| `soffit_or_porch_ceiling_failed` | catalog_candidate | 1 | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 9 | R3 |
| `tub_surround_or_shower_pan_damage` | catalog_candidate | 1 | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 23 | R3 |
| `unfinished_interior_wall_osb_exposed` | catalog_candidate | 2 | 2 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 3 | R1 R5 |
| `vanity_worn_finish` | catalog_candidate | 2 | 2 | 2 | 0 | 1 | 0 | 1 | 2 | 0 | 23 | R1 R2 R4 |
| `vinyl_linoleum_torn_or_lifted` | catalog_candidate | 2 | 2 | 1 | 0 | 0 | 0 | 0 | 1 | 0 | 32 | R3 |
| `vinyl_linoleum_worn_or_stained` | catalog_candidate | 1 | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 31 | R1 |
| `wall_scuffs_marks_or_dents` | catalog_candidate | 13 | 13 | 9 | 0 | 6 | 0 | 0 | 1 | 0 | 40 | R3 R4 R5 |
| `worn_or_stained_flooring` | catalog_candidate | 4 | 4 | 4 | 0 | 3 | 0 | 0 | 1 | 0 | 31 | R2 R4 |

Coverage questions (no implicated item): 63 records; product-policy lane: 0 records; deferred findings: 11 sections; hallucination annotations: 10 cases.

## Unavailable data

- current_retrieval_neighbors `*`: no offline embedding store exists; tools/catalog_embeddings.py builds vectors at construction against the live sidecar, which this session may not call
- factorized_lead `lead:redfin_126224899:oc1_17e2478acfc349e8`: run_identity_unpinned
- factorized_lead `lead:redfin_126224899:oc1_4900757ed55e17c9`: run_identity_unpinned
- factorized_lead `lead:redfin_126224899:oc1_4f7604e7064dce56`: run_identity_unpinned
- factorized_lead `lead:redfin_126224899:oc1_8b6bf7c321af5b5d`: run_identity_unpinned
- factorized_lead `lead:redfin_126224899:oc1_999efd853008d114`: run_identity_unpinned
- factorized_lead `lead:redfin_126224899:oc1_a72e30e0a8cb65b6`: run_identity_unpinned
- factorized_lead `lead:redfin_126224899:oc1_d32b38b4d91549c0`: run_identity_unpinned
- factorized_lead `lead:redfin_126224899:oc1_efe915a40d1bb026`: run_identity_unpinned
- run_artifact `production:redfin_10965375:20260821_233007_c5b989ee`: queue stores no artifact path/hash for this run (orphan lane); not read

## Validation

| Check | OK | Detail |
|---|---|---|
| frozen_pins_match | True | "verified before build (fail-closed)" |
| queue_lane_counts_match_header | True | {"appendix_inconclusive": 6, "appendix_misnamed": 8, "appendix_trivial": 6, "counted_agreement": 7, "counted_correct_rejection": 31, "counted_orphan": 11, "halluc_label": 9, "halluc_v1only": 3, "miss_ |
| run_artifacts_verified | True | {"pinned": 25, "verified": 25} |
| artifact_catalog_sha256_uniform | True | ["d1cef743f379918fa17f0684c245779180d7d692ebc8fd3e66b7c69bebf68347"] |
| catalog_identity_resolved | True | "07ee112b79b0a96d116b38eb5dc03e8c7492a2a9" |
| working_tree_matches_head | True |  |
| canary_run_identity_unique | True |  |
| decision_counts_total | True | {"narrowed": 4, "reclassified": 50, "retired": 2, "split": 19, "unchanged": 32} |
| issue_items_equal_condition_item | True |  |
| label_item_matches_card | True |  |
| labels_without_card | True |  |
| artifact_resolution_agrees_with_queue | True |  |
| every_seed_reconciled_once | True |  |
