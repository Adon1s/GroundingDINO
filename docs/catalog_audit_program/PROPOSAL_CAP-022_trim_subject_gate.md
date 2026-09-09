# CAP-022 — Subject gate on `dated_interior_trim` (`require_any`)

Date: 2026-09-07. Drafted by Fable 5.1 after Steven's S6-1 ruling ("fix first"), as the follow-up
proposal the Session 6 handoff's fix-first path calls for. Status: **drafted, awaiting Steven's
approval of the exact op at the gate.** No catalog, decisions, prompt, route or threshold has been
changed; no provider call was made; nothing is committed.

Program context: `00_OVERALL_CONTEXT.md`, `04_HUMAN_REVIEW_GATE.md`, `HANDOFF_SESSION_6.md`,
`RESULT_catalog_audit_live_validation_20260907.md` (sections "Error migration", "Repeat stage",
"Rulings"), bundle `reports/catalog_audit_live_validation.json` (`repeat.s6_1`, `rulings`).

## 1. The observed failure

CAP-007 removes `dated_window_treatment_valance` and adds a presence-only `no_action` blinds successor
and a `require_any`-gated fabric successor. Stage A and its same-day repeat showed that, with the parent
gone, two pinned observations migrate onto the billable `dated_interior_trim` item reproducibly:

| Row | Observation | Baseline (`9afe0fa`) | Candidate (`6e67eaa`), Stage A + 5 replicas | Terra on the trim claim |
|---|---|---|---|---|
| R1 `canary:redfin_10803207:…:photo_005.jpg:8b3fe5d13b4504e2` (review card rc_a22a241b3bf0) | "Window blind and trim appear dated." | parent 6/6, Terra unsupported | trim **6/6**, LLM path, margin 0.0019 | supported 2/2 ("plain, narrow, builder-grade window and door trim") |
| R2 `canary:redfin_10803207:…:photo_017.jpg:4e670ca509e764c4` (3.1 condition `oc1_414b89a8d035a952`) | "Window treatments are dated and basic." | parent 6/6, Terra supported, billed | trim **5/6** (once no item), LLM path, margin 0.0075 | supported 2/2 |
| R3 `canary:redfin_80990371:…:photo_025.jpg:322eb2b7f52e83c3` | "The blinds and window trim are dated." | trim 5/6 | trim 6/6 | supported in both arms |
| R4 `production:redfin_80917686:…:photo_016.jpg:c39a0526af0466c9` | "The window treatments and trim are dated stylistically." | parent 5/6 | no item 5/6, trim 1/6 | n/a |

R3 behaves the same in both arms and is not a candidate effect. R1 and R4 name trim; R1's landing is
Terra-supported on a photo that shows plain builder-grade trim, so it is a real condition the parent
used to absorb, not a subject error. **R2 is the defect**: the bullet never names trim, the photo
(opened during the redraft, `reports/catalog_audit_redraft.json` `cap007.require_any_decision`) shows
three bay windows with plain white mini-blinds and no fabric, and the candidate bills about $76 low
for trim on it. Terra supports "plain trim" whenever it is asked about trim, so Terra does not catch
subject errors of this class.

Mechanism (bundle `repeat.s6_1`, RESULT "Error migration"): `dated_interior_trim` has no `require_any`;
its retrieval text ("basic window trim", "older painted window surround", "dated window casing")
sits next to window-treatment bullets in the embedding; subject-generic treatment bullets cannot reach
the presence-only blinds successor (its `support_any` is blind/shade vocabulary and the bare token
"window treatment" was deliberately kept out of the fabric successor's `require_any`), so Pass 2d's
LLM picks the nearest billable modernization neighbour at margins under 0.01.

## 2. Levers evaluated, offline, on the whole replay corpus

Method: the candidate catalog (`43d8147e…7acf`, 129 items) patched in memory; all 1,969 rows of the
Session 5 replay corpus (`reports/catalog_audit_replay_corpus.json`, the product-lane observations of
the 26 pinned runs) re-retrieved through the production retriever (`CatalogEmbeddingsRetriever`,
top-8, guardrails) and the production lexical shortcut (`_resolve_candidate_via_lexical_shortcut`,
min score 0.72, min margin 0.03); embeddings sidecar only; no provider call. Every delta below is
between two variants of the same embedding run, so it is exact. Script, report and self-check are
frozen under `artifacts_canary/catalog_audit_session6_20260907/provenance/cap022_lever_eval/`
(hashes in bundle `rulings.s6_1.offline_lever_evaluation.provenance`).

Corpus facts: 108 rows are owned by trim (18 by lexical shortcut, 90 by the LLM); 49 rows contain
"treatment", of which 2 are trim-owned ("The doorway trim treatment appears dated.", "The crown and
trim treatment appears older."); 25 rows contain "window treatment" (owners: parent 12, no item 9,
windows 2, decor 2).

| Lever on `dated_interior_trim` | Authorable today | Rows losing trim from their top-8 | …where trim was first | Trim-owned rows lost | Shortcut changes | Pinned rows whose first candidate changes |
|---|---|---:|---:|---:|---:|---|
| A. `deny_any: ["window treatment"]` | yes (carryover wording override) | 14 | 2 | 0 | **1, new** | R2; "Window treatment is basic." |
| B. `deny_any: ["treatment"]` | yes | 30 | 2 | **2** (the two trim bullets above) | 1, the same | same two |
| C. `require_any` set B | **no** (system gap, §4) | 699 | 22 | 4, all subject-mismatched | **0** | same two |
| D. `require_any` set C (= B + `surround`, `jamb`) | no | 691 | 21 | 3 | 0 | same two |

Set B = `["baseboard", "trim", "casing", "millwork", "molding", "moulding", "wainscot", "chair rail", "crown"]`
(the item's own `support_any` vocabulary plus the trim-family words the corpus uses: "The chair rail is
dated.", "The chair rail and wainscot-style division are dated."). Terms are word-start anchored stems
under `tools/pipeline_common.py` `term_matches`: `trim` admits trims/trimmed/trimming, `crown` admits
"crown molding", `wainscot` admits wainscoting; no trailing-boundary marker is needed. The item is
interior-only (`scene_groups` kitchen, bathroom, bedroom, living_areas, utility), so exterior "trimmed
hedges" never reaches it.

**The new shortcut under A and B.** "Window trim and window treatment are dated."
(`canary:redfin_125970550:…:photo_024.jpg:98f322926d11c81e`, parent-owned, one of the four unpinned
rows excluded from Stage A) has, under the candidate as-is, `dated_or_older_windows` 0.7847 first and
trim 0.7843 second (margin 0.0004, LLM path). Removing trim leaves windows 0.7847 over the blinds
successor 0.7111: margin 0.0736, the support term "window" hits, and the lexical shortcut fires onto
the billable windows item for a bullet about trim and treatments. That is new false-positive behaviour
of exactly the S5-3/S6-2 collapsed-margin kind, on a row the pinned case set would not see.

**The four trim-owned rows lost under C** are "Windows are basic.", "Doors and jambs appear aged.",
"Exposed framing appears old." and "The built-ins are dated.": none is about trim, all were billed as
trim by the LLM, and under ruling 3 (plain or basic presence is not billable by itself) none should
bill as trim. Their predicted next candidates include other billable items (blinds successor or
interior doors; decor or entry door; windows or decor; wood paneling or interior doors), so they are
existing subject errors that may move rather than vanish; Session 7's local Pass 2d pass on the 22
rows where trim was first measures where they go.

**R2 under C**: `dated_overall_decor_style` 0.6103 first (generic: `drop_if_generic`, so a resolution
onto it is dropped by the Pass 2e kill switch), `older_flooring_style` 0.5845 second, margin 0.026,
LLM path. Predicted: no billing. **R1, R3, R4 under C**: unchanged (they name trim). **"Window
treatment is basic."** (pinned, affected): trim leaves the first slot; the blinds successor is first at
0.454 (below the shortcut floor), LLM path; Stage A already resolved it to the blinds successor.

Instrument note: the fresh embeddings reproduce the Session 5 candidate snapshot's top-8 identity on
1,918 of 1,969 rows; the 51 differences are 46 order-only swaps of near-tied neighbours and 5 eighth-slot
substitutions (first candidate differs on 3 rows; maximum score delta on identical lists 0.0013), the
batch-position noise the Session 5 harness documents. The Session 5 vector cache no longer exists, so
this evaluation re-embedded; a Session 7 snapshot compare will use its own cache.

## 3. The proposed change (exact)

Recommended: lever C, authored as a carryover override on the existing `dated_interior_trim` entry of
`tools/catalog_migrations/kind_v2_decisions.json`, in the renderer's op format
(`scripts/render_catalog_audit_proposals.py` `apply_ops`; `_walk` creates the `overrides` dict on
demand):

```json
{"legacy_id": "dated_interior_trim", "op": "set", "path": "/successors/0/overrides/require_any",
 "after": ["baseboard", "trim", "casing", "millwork", "molding", "moulding", "wainscot", "chair rail", "crown"]}
```

Generated effect: the `dated_interior_trim` item gains `require_any` with those nine terms; no other
field of that item and no other item changes. `id`, `name`, `description`, `embed_text`,
`support_any`, `deny_any` (empty), `atomic_claim`, kind, severity, trade bucket, scene groups, route,
`work_item_code` (`TRIM_REPLACE`), cost mode, package role and affinity are untouched. Economics are
not authorable and are not touched. `requires_re_resolution` is already `true` on the entry.

Variant for the gate: set C adds `surround` and `jamb` (keeps "Doors and jambs appear aged." on trim;
`surround` also admits "tub surround" and "fireplace surround" bullets into trim's candidate pool
without forcing them). Not recommended: "jamb" is door-frame, not casing, and the item's claim is
"plain, thin, or builder-grade trim package".

Precedent: 24 catalog items already carry `require_any`, including the approved fabric successor
(`dated_window_valance_or_curtains`), `dated_fireplace_surround` and `dated_bathroom_wallpaper`. The
mechanism is the retriever's hard gate (`tools/catalog_embeddings.py` `_passes_guardrails`: deny
first, then require), applied after top-K scoring and before the candidate list reaches the shortcut
and the LLM. It has no effect on the shortcut's own logic and no effect on any other item.

## 4. System gap: `require_any` is not a carryover override field today

`scripts/migrate_catalog_kind_v2.py` `WORDING_OVERRIDE_FIELDS` is
`{name, description, embed_text, support_any, deny_any}`; `_build_carryover` fails on any other override
key, and the renderer's `classify_op` classifies a `require_any` override on a reclassified entry as
`gap` ("carryover overrides are limited to wording fields; retrieval metadata, gates, and routes are not
authorable here"). Split successors may override `require_any` (it is in `INHERITED_FIELDS`), which is
how CAP-007's fabric successor got its gate; carryovers may not.

Closure, in Session 7, before the op is applied: admit `require_any` to the carryover override set (one
line; rename the constant, since it is no longer wording-only) with a test that a carryover override
of `require_any` lands verbatim and that an economic or unknown key still fails. The evidence bundle's
`authoring_surface.wording_override_fields` is pinned and must not be edited; Session 7 records the op
as a gap closed by that generator change, with the generator's new blob hash in its handoff.

Alternative, recorded and not recommended: author `require_any` on the v1 item in
`tools/issue_catalog.json`, which the carryover copies verbatim with no code change. It reopens v1
authoring, which the program moved away from, and it changes legacy-v1 behaviour too: `.env` does not
set `KIND_ONTOLOGY_VERSION`, so any invocation without the per-invocation override runs the v1 catalog.

## 5. Expected effects and risks

- Routing: `dated_interior_trim` is reachable only when the observation names a trim-family subject.
  In the replay corpus that removes trim from the candidate list of 699 rows (all of them not naming
  trim; trim was the first candidate on 22), changes no lexical shortcut, and re-resolves 4 trim-owned
  rows that were subject errors. 104 of 108 trim-owned rows keep their owner reachable.
- The S6-1 defect: R2 is predicted to stop billing (generic decor first, kill switch). R1 stays on trim
  and is declared an expected, Terra-supported landing; after CAP-022 the rubric's migration clause is
  evaluated with R1 declared, so the combined change can pass it.
- No new false-positive behaviour is predicted by the offline instrument (0 shortcut changes); the LLM
  path on the 22 rank-1 rows is the residual risk and is measured locally in Session 7.
- Loss risk: any real trim observation phrased without a trim-family word is lost to trim. Measured
  rate in the corpus: 4 of 108, none about trim. The catalog-wide ruling-3 sweep that owns trim
  (CAP-008, deferred) can widen the term list if production shows real losses.
- Economics, packages, routes, display: unchanged. Dollar effect in the pinned set: about $76 low
  removed on R2; nothing else.
- The fabric successor, the blinds successor and CAP-007's own ops are untouched; the CAP-007
  generated-diff proof continues to hold for those items, and Session 7 adds a second proof for this op.

## 6. Validation plan (Session 7)

1. Tier 1: apply the op with `apply_ops`, regenerate twice (byte-identical), generated diff equals the
   op (only `dated_interior_trim` gains `require_any`; `item_diff` empty elsewhere), validator 0 errors,
   generator test green.
2. Tier 2: Session 5's snapshot compare on the replay corpus must show trim leaving 699 top-8 lists,
   0 shortcut changes, and exactly the four trim-owned losses in §2 (fresh cache; small near-tie
   differences from the frozen Session 5 snapshot are expected and are not a finding).
3. Stage A re-run on the same 66 cases: the baseline arm is reusable from
   `artifacts_canary/catalog_audit_session6_20260907/baseline/` (same commit, catalog and fingerprint);
   only the candidate arm runs on the new commit under a new manifest pin, resuming from disk. Expected:
   R2 no longer bills; R1 on trim as declared; nothing else moves; A-4's 23 blinds conditions and the
   kitchen shortcut unchanged; both material findings as in Session 6.
4. A local, free Pass 2d pass (x5) on the 22 corpus rows where trim was first; Terra only if a new
   billable landing appears (under 50k tokens).
5. Rubric applied to the combined change; if it passes, the publication task is justified.

## 7. Gate checklist (04_HUMAN_REVIEW_GATE.md)

| Item | Status |
|---|---|
| Observed failure and evidence reviewed | R2, photo opened at the redraft; reproduced 5/6 live; Terra 2/2 |
| Successful uses, correct rejections, counterexamples considered | 104/108 trim-owned rows keep trim reachable; R1 and R3 (trim-naming) unaffected; the two "trim treatment" bullets survive (they would not under lever B) |
| Evidence independence | two properties (canary redfin_10803207, production redfin_80917686) plus the unpinned redfin_125970550 row for the deny side effect |
| Catalog ownership stronger than other stages | the leak is a retrieval-pool property; Terra cannot catch it; the Pass 2d prompt is frozen by program rule |
| Supported by the migration system, or a classified gap | **gap**, §4, one-line generator closure |
| Retrieval/kind/scene/embedding risks | §5; no embedding text changes; interior-only |
| Economic, work-item, package, route, product-policy effects | none |
| Adversarial conditions | the deny form's new windows shortcut (S6-11) is why the require form is chosen |
| Exact approved change unambiguous | §3 op; the term list is the only free choice |

Disposition requested from Steven: `approved` (set B) or `approved_with_modification` (set C, or an
edited term list). A `deferred` disposition returns CAP-007 to the publish-and-carry path Part B
recommended, which Steven declined on 2026-09-07.
