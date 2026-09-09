# Phase B reconciliation — catalog audit Session 2

Date: 2026-09-02

Provenance: neutral reconciliation authored by Steven after delivery of the Phase B photo/lineage review, pasted into the review session and saved here verbatim on 2026-09-02 so Phase C can read it from the repository rather than from conversation. It joins `reports/catalog_audit_photo_review.json` (sha256 `2225db9caae573b52b4270f846c314749d39aff468fab4e1a8e2012c49919150`) to the frozen error-attribution cohort. It is an input to Phase C, not a decision; its case interpretations and Phase C questions are open.

---

# Neutral reconciliation report

## Bottom line

The two evidence layers mostly reconcile after matching each adjudication to its actual claim target.

- **Exact overlap:** 83 packet rows map to **80 unique underlying cases/units**: 40 runtime conditions and 40 gold findings, across 71 photos and 22 properties.
- **My reconciliation, deduplicated:** 67 aligned; 5 direct human/gold-versus-photo-review disagreements; 3 wording/billability convention cases; 1 mixed multi-photo case; 4 unresolved visual ambiguities.
- The headline attribution findings are not overturned, but several individual cases should not carry catalog conclusions without Phase C adjudication—especially `rc_a22a241b3bf0`, `rc_3a19f759f2d9`, `rc_7b23faa0bd8f`, `rc_b642fe69b86a`, and gold `photo_001#g1`.
- No repository files were modified. `git diff --name-only` remained empty; no provider or pipeline call was made.

## Verified artifacts

| Artifact | Verified SHA-256 |
|---|---|
| HEAD / branch | `9afe0fa5a0490a856abed507dccc4a02ed1de24c` / `terra_factorized_verifier` |
| [Final attribution result](C:/Users/Steven/PycharmProjects/realtorvision-backend/docs/RESULT_error_attribution_20260831.md) | `e3c1a5350907a11fb546d04c4e3d99bde3f4f6e3e79409687e35826fe0833d35` |
| [Attribution JSON](C:/Users/Steven/PycharmProjects/realtorvision-backend/reports/error_attribution.json) | `90447b955ff89e4a8b13e1e31c19036411b141593484a1560974a8890ddd1a2c` |
| Frozen attribution queue | `b58dcca7c3779f2b0539f988077f865c6c1ca040d23a44646d89b04a398940df` |
| Attribution verdict ledger | `166e8bec20641c0a8fb0ca5ddbd6c042f31b828447304ccbbbd4d585cd2a894e` |
| [Session 1 evidence](C:/Users/Steven/PycharmProjects/realtorvision-backend/reports/catalog_audit_evidence.json) | `43354104cc109c768efbe2d2f8b9a825f33e0882210b81e9f12b847ca67e29b7`; fingerprint `e4c0a2b816395782cd84bc4abcdf6207d131a34a0e2d63eaebb39bec4c190c08` |
| [Photo packet](C:/Users/Steven/PycharmProjects/realtorvision-backend/reports/catalog_audit_photo_review_packet.json) | `7b4621dfe6c72aa42369c2f5ef7051dfecbec05b6333a80651b1b57a5b37bc80` |
| [Photo review](C:/Users/Steven/PycharmProjects/realtorvision-backend/reports/catalog_audit_photo_review.json) | `2225db9caae573b52b4270f846c314749d39aff468fab4e1a8e2012c49919150` |

The review self-check passed: 223 unique answers, no unknown or missing required rows, valid vocabulary, 153 `supports_claim`, 60 `refuted`, 10 `unclear`, and all 60 optional rows completed. All 191 photo records say `sha256_verified=true`. I independently rehashed and reopened the 25 photos involved in direct disputes or ambiguity; all matched.

## Denominators

| Population | Rows | Unique underlying keys | Notes |
|---|---:|---:|---|
| Entire packet | 223 | 208 | 15 duplicate/multi-role or multi-cluster annotations |
| Human/gold packet evidence | 184 | 169 | 128 supports, 47 refuted, 9 unclear |
| Model-only leads | 39 | 39 | 25 supports, 13 refuted, 1 unclear |
| Exact attribution overlap | 83 | 80 | 58 supports, 21 refuted, 4 unclear at row level |
| Overlap by row type | 39 unit, 36 coverage, 8 control | — | All are human/gold-backed; no model-only lead overlaps the finalized attribution cohort |

The overlap covers 80 of 95 attributable cases, but only 83 of 223 packet rows. Aggregate percentages from the two reports therefore remain incomparable.

Three overlap cases produce duplicate rows:

- `rc_022497229bd3` and `rc_a0e75a18620e` each appear as the same hallucination control in CAP-006 and CAP-008.
- `rc_9c702893a6ce` appears under two different flooring conjunctions. CAP-004 refutes a material mismatch; CAP-005 supports a style-versus-wear mismatch. This is not a contradiction.

## Reconciliation counts

These are my mutually exclusive interpretations, not fields stored in either artifact:

| Category | Unique cases | Cases |
|---|---:|---|
| No disagreement after aligning the claim target | 67 | Remaining overlap |
| Human/gold basis versus photo-review disagreement | 5 | `ga_…photo_001_g1`, `rc_a22…`, `rc_3a19…`, `rc_7b23…`, `rc_b642…` |
| Disjunctive wording or billability convention | 3 | `rc_022…`, `rc_a0e…`, `rc_4cab…` |
| Mixed multi-photo evidence | 1 | `rc_506…` |
| Unresolved visual ambiguity | 4 | `rc_128…` plus three gold findings |

## Case-level disagreements and distinctions

### 1. Gold stairs: direct visible-fact disagreement

**Record:** `coverage:gold:redfin_10806500:photo_001.jpg:g1`; unit `gold:redfin_10806500:photo_001.jpg:g1`.

- **Gold basis:** "Long concrete entry steps are weathered and stained."
- **Pass 2a:** the stair run "looks fairly substantial and generally serviceable."
- **2d/Terra:** no stair issue reached either stage.
- **Final attribution:** `pass_2a`, medium confidence.
- **Photo review:** `refuted`; light surface dirt only, without defined staining, spalling, or visible weathering.
- **My inspection:** agrees with the photo reviewer. The frozen gold premise is disputed, so this case does not currently establish a Pass 2a miss or catalog coverage gap.

### 2. Flooring material/lifting: unresolved and partially contradicted

**Record:** `CAP-004:runtime:canary:redfin_11000447:20260817_224602_d422d7ae:oc1_476cd4f80565d162`.

- **Human basis:** v1.1 `claim=exact`, `work=warranted`.
- **Pass 2a:** "dark plank flooring is heavily worn/scuffed" and appears to have "lifting/open seams."
- **2d:** selected `vinyl_linoleum_torn_or_lifted`, rank 1; candidates were `vinyl_linoleum_torn_or_lifted`, `trip_hazard_or_unlevel_floor`, `hard_flooring_broken_or_warped`, `bare_or_missing_finish_flooring`, `ceiling_cracks_or_sagging`, `boarded_up_entry_or_window`, `rotted_subfloor_or_structural_framing`.
- **Terra:** unsupported because it saw hard plank material and no vinyl/linoleum tears, lifting, or open seams.
- **Final attribution:** Terra, high confidence.
- **Photo review:** `unclear` on LVP versus laminate, but finds no tearing or lifting.
- **My inspection:** material is genuinely ambiguous; visible seams and wear do not establish lifting. Terra overstates material certainty, but its no-lifting observation is better supported than Pass 2a's lifting claim. The finalized Terra attribution is therefore not secure on the defect branch.

### 3. Ordinary mini-blinds: human/v1-only versus photo evidence

**Record:** `CAP-007:runtime:canary:redfin_10803207:20260817_222851_c0551630:oc1_bd04d765dac55f9b`.

- **Truth basis:** v1-only `terra_claim_supported`.
- **Pass 2a:** "the blind and window trim look dated."
- **2d:** selected `dated_window_treatment_valance`, rank 1; its atomic claim is "dated valance or treatments." Remaining candidates were `dated_interior_trim`, `dated_overall_decor_style`, `dated_electrical_outlets_switches`, `dated_or_older_windows`, `outdated_bathroom_finishes`, `dated_wood_paneling`, `paint_refresh_recommended`.
- **Terra:** plain horizontal blinds; no dated valance.
- **Final attribution:** 2d, medium confidence.
- **Photo review:** `refuted`; clean, complete, ordinary white blinds, no valance.
- **My inspection:** agrees with the photo reviewer. The catalog item explicitly covers treatments and its embedding text covers mini-blinds, so 2d did not necessarily invent a valance-only claim. The weak point is Pass 2a's datedness judgment or the underlying v1-only premise, not demonstrated 2d misresolution.

### 4. Weathered deck mapped to porch-surface wear

**Record:** `CAP-009:runtime:production:redfin_10735912:20260825_045924_209d10ee:oc1_9312f2aa56516320`.

- **Human basis:** `misnamed_billed`, work warranted.
- **Pass 2a:** weathered floorboards, possible splitting/cupping and repair/replacement; the front-photo observation has no porch.
- **2d:** selected `patio_or_porch_surface_wear`, rank 1. `deck_surface_weathering` was rank 4 on both relevant rear photos.
- **Terra:** supported "clear weathering and surface wear."
- **Final attribution:** 2d, medium confidence.
- **Photo review:** `refuted` the cluster mismatch; the visible greyed porch/deck boards satisfy "patio or porch surface aging [or] surface wear," without clear rot or structural failure.
- **My inspection:** agrees. A more specific reachable item existed, but the selected atomic claim is not false. Whether "less specific" counts as misnamed is an ontology convention, not a pixel fact.

### 5. Exterior trim resolved to siding discoloration

**Record:** `CAP-002:runtime:production:redfin_10740044:20260821_233648_4cd603bb:oc1_30375086ad2fd9f6`.

- **Human basis:** v1.1 exact/warranted.
- **Pass 2a:** "the white-painted trim against the stone looks unfinished in spots."
- **2d:** selected `exterior_siding_discoloration_fading`, rank 5. Candidates were `soffit_or_porch_ceiling_weathered`, `exterior_door_paint_failure`, `baseboard_wear_scuffs`, `shed_exterior_paint_failure`, selected item, `fence_weathered`, `brick_weathered_or_discolored`.
- **Terra:** "No exterior siding with fading, chalking, or uneven discoloration is visible."
- **Final attribution:** Terra, medium confidence, explicitly because human `claim=exact` overrode the catalog mismatch concern.
- **Photo review and my inspection:** the facade is stone; the rough white trim is real, but siding/fading is absent. Terra evaluated the atomic Terra claim accurately. The visible condition first changes meaning at 2d unless the broader catalog description's "siding or trim" is allowed to override the atomic claim. No exact candidate was available.

### 6. Heavy dated trim resolved to "plain/thin/builder-grade"

**Record:** `CAP-008:runtime:canary:redfin_80990371:20260817_221632_3d4a8269:oc1_c6c4c4017b1ed9dd`.

- **Human basis:** v1.1 exact/warranted.
- **Pass 2a:** heavy dark wood trim, wide rough window trim, and aged doors/jambs.
- **2d:** selected `dated_interior_trim`; no candidate names heavy/substantial dated trim.
- **Terra:** substantial painted wood, not plain, thin, or builder-grade.
- **Final attribution:** Terra, medium confidence, again because the human exact axis was treated as controlling.
- **Photo review and my inspection:** agree with Terra's literal assessment. The item name/description includes "dated," but the atomic Terra claim is "plain, thin, or builder-grade." Under atomic-claim authority, 2d/catalog wording owns the mismatch; under item-name authority, Terra owns it. Phase C must choose the governing convention.

### 7. Plain/basic trim controls: pixels agree, billability does not

**Records:** `rc_022497229bd3` and `rc_a0e75a18620e`, duplicated in CAP-006/CAP-008.

- **Human basis:** `claim=absent`, `work=none`, hard false billed.
- **Pass 2a:** explicitly classifies basic trim/baseboards as dated cosmetic upgrades.
- **2d:** `dated_interior_trim`, rank 1 or 2 across all four issues.
- **Terra:** supports plain/narrow/basic trim.
- **Final attribution:** Pass 2a, high confidence.
- **Photo review and my inspection:** literal atomic claim is supported: the trim is plain/basic. Nothing is dated, damaged, or evidently worth replacing.
- **Reconciliation:** no visible-fact dispute. This is a program convention: neutral low-grade presence versus billable datedness. The stage attribution can remain Pass 2a while catalog ownership remains open because the catalog itself treats plain/builder-grade trim as the condition.

### 8. Tired walls mapped to peeling/discolored paint

**Record:** `CAP-001:runtime:canary:redfin_10806500:20260817_224354_3633e400:oc1_f46cc039d3163f19`.

- **Human basis:** misnamed/warranted.
- **Pass 2a:** "Visible scuffs, patchy areas, and generally tired finishes…"
- **2d:** selected `peeling_or_discolored_paint`, rank 7; `wall_scuffs_marks_or_dents` was rank 6.
- **Terra:** supports visibly aged, uneven, patchy discoloration.
- **Final attribution:** 2d, high confidence.
- **Photo review and my inspection:** peeling/bubbling is absent, but the atomic claim's "visibly aged finish" disjunct is true.
- **Reconciliation:** catalog ownership depends on whether the full disjunction controls or the named mechanism must characterize the condition. A more precise candidate was reachable, but the selected claim is not literally unsupported.

### 9. Brick condition aggregated from brick and stone photos

**Record:** `CAP-002:runtime:canary:redfin_80925528:20260818_220543_1af5ac93:oc1_fd62a1d71743be6b`.

- **Human basis:** v1.1 exact/warranted.
- **Pass 2a:** accurately identifies stone on several photos and a brick chimney on `photo_023`.
- **2d:** resolves all seven issues to `brick_weathered_or_discolored`. The item ranked first for four issues, second for two stone issues, and sixth for the parged-masonry issue. No stone-masonry item appears in any list.
- **Terra:** masonry is weathered, but mostly stone and with deteriorated joints.
- **Final attribution:** Terra, high confidence, because one brick photo supports the aggregate condition.
- **Photo review and my inspection:** five photos are stone or mixed masonry; `photo_023` contains real weathered brick. Terra's rationale fits most constituent issues but overlooks the supporting brick issue.
- **Reconciliation:** both layers are locally correct. The stage attribution is defensible under "any evidence photo supports the aggregated condition"; catalog ownership remains implicated for the five stone observations. These must not be treated as six independent cases.

### 10. Three genuinely ambiguous gold findings

Independent inspection agrees with `unclear` for:

- `gold:redfin_10806500:photo_006.jpg:g7`: possible ceiling discoloration versus light falloff.
- `gold:redfin_11000447:photo_001.jpg:g7`: transom shadow/grime versus finish wear.
- `gold:redfin_11000447:photo_004.jpg:g8`: generic round flush mount; basic versus dated is not visually decidable.

## Findings directly supported by the join

**Fact:** Of the 40 overlapping gold cases, 36 underlying findings remain visibly supported, one is refuted (`photo_001#g1` stairs), and three remain ambiguous.

**Fact:** Several row-level `refuted` results reinforce rather than challenge Terra attribution. Examples include `rc_48ea…`, `rc_790b…`, `rc_5b3b…`, `rc_a35…`, `rc_d60…`, `rc_d019…`, `rc_ace…`, and `rc_eace…`: the row's provisional mismatch conjunction was refuted because the catalog mechanism was actually visible, while Terra had rejected it.

**Fact:** `rc_3deb5aaef0f0` reconciles cleanly once targets are separated. Pass 2a's "uneven-looking" is unsupported, while the selected patio-surface-wear claim is supported. This is a Pass 2a error without catalog ownership.

**Fact:** `rc_9c702893a6ce` is a duplicate-claim artifact, not inconsistent evidence: material matches vinyl, but dated style does not match the selected wear claim.

**Program convention:** Attribution follows the frozen human claim axis even when the photo reviewer finds the atomic catalog commitment incomplete. That convention explains, but does not independently validate, the Terra attributions in `rc_7b23…` and `rc_b642…`.

**My inference:** The joined record does not support treating "Terra-attributed" as synonymous with "Terra-owned." Some Terra attributions are catalog-wording or aggregation problems under a complete-claim reading.

## Limitations and review-quality concerns

- The photo review is one model annotation, not new human truth.
- Some batches contained 17–18 images plus crops. Reopening the 25 disputed/ambiguous photos confirmed the key distinctions, but I did not independently re-review all 191.
- Cluster unit claims are often conjunctions. `refuted` frequently means only that one branch failed.
- Gold covers two properties, so its 40 overlapping findings are highly correlated.
- Style judgments—dated beige, basic lighting, plain trim—are intrinsically convention-sensitive.
- Pass 2a excerpts are sentence-level causal interpretations even when verbatim.
- Candidate lists are frozen evidence-era retrieval, not current 3.2 retrieval. Current reachability was not tested.
- Catalog name, description, atomic claim, and embedding text are not always semantically identical. `dated_interior_trim` and `exterior_siding_discoloration_fading` make this especially material.
- Five stone photos in `rc_506…` are annotations on one condition/property, not five independent catalog cases.

## Questions Phase C must answer

1. Which text is authoritative when catalog name/description and atomic Terra claim differ?
2. For disjunctive claims, is any satisfied branch enough, or must the item's named mechanism be characteristic?
3. Does multi-photo condition support use any-photo, majority-photo, or issue-specific semantics?
4. Should the direct truth conflicts be re-adjudicated by a human, especially gold `photo_001#g1`, `rc_a22…`, `rc_3a19…`, `rc_7b23…`, and `rc_b642…`?
5. Is neutral "plain/basic/builder-grade" trim billable, or only trim that is visibly dated/defective?
6. For `rc_3a19…`, does selecting broad porch-surface wear when `deck_surface_weathering` was rank 4 constitute a wrong selection or merely lower specificity?
7. For `rc_4cab…`, does the true "visibly aged finish" disjunct make the peeling item acceptable despite absent peeling?
8. How should Phase C represent catalog absence where no exact candidate existed: stone masonry, unfinished exterior trim, and substantial-but-dated interior trim?
9. Which excluded gold findings are catalog questions versus product-policy/nonbillable observations—especially missing base trim versus quarantined electrical rows?
10. After resolving those conventions, the evidence bar must be recomputed from unique units, keeping the 39 model-only leads separate from human/gold evidence.

No proposal, migration, catalog, prompt, runtime, or evidence conclusion should be changed from this reconciliation alone.
