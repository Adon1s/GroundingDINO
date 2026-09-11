# Backend acceptance — Session A decision packet

Prepared against `master` at `a656152`, 2026-09-10. Implements the offline
checkpoint in [the acceptance plan](PLAN_backend_acceptance_20260910.md).
Terra calls: **0**. Sol calls: **0**. No gate adopted and no acceptance batch
started. The catalog and all historical artifacts are unchanged.

**Recommendation: no justified new Terra experiment from these measurements;
retain the six cases as unresolved misses pending Steven's ruling. Keep the
corroboration gate off. Confirm Qwen at 0.1 for acceptance and approve the two
prepared reliability fixes. These are recommendations, not recorded approvals.**

## 1. Six cases: established evidence, separate interpretations

The [mechanism census JSON](../reports/terra_miss_mechanisms_20260910.json)
contains the exact observation texts, production rationales, factorized
responses, human answers and provenance for every row below. All six have
non-blind `exact_and_warranted` re-asks and null adjudication notes. Their
separate attribution records have `skeptic_checked: false`. These labels remain
the scoring labels; they do not
explain the disagreements.

| Case | Observation / selected item | Stored / factorized verdict | Call conditions / photos | What the record supports |
| --- | --- | --- | --- | --- |
| `rc_790b583fcbf0` | “The paint appears tired.” / `peeling_or_discolored_paint` | unsupported / absent | 4 / 1 | Neutral-paint/aged-finish disagreement. Production negates peeling and bubbling; factorized also negates an identifiable aged finish. Adjacent wording alone does not account for both. |
| `rc_905575ed8a3d` | Waviness and sagging at porch-roof connection / `soffit_or_porch_ceiling_failed` | cannot_assess / inconclusive | 4 / 1 | Connection-versus-panel subject/coverage ambiguity. Both models decline the panel claim. No new visibility-hedging hypothesis is inferred. |
| `rc_a35f1b3a231e` | Weathered siding, visible wear and weathered trim / `exterior_siding_discoloration_fading` | unsupported / absent | 12 / 13 | Separate siding/trim wording issue. Both rationales deny fading/chalking/discoloration. This is not proof of why the human/model disagreement occurs. |
| `rc_ace7463d6837` | “The shower valve and trim appear damaged.” / `tub_surround_or_shower_pan_damage` | unsupported / absent | 19 / 4 | Explicit subject mismatch. Factorized says “The shower hardware is damaged” while declining cracking/chipping of the enclosure. A model acknowledging hardware damage is not a recovery of the selected enclosure claim. |
| `rc_d0197df43080` | Scuffed/worn hardwood and minor wear/color variation / `hard_flooring_scratched_or_worn` | unsupported / absent | 9 / 7 | Wear-perception disagreement on the same material. This particular case is not evidence of a hardwood-versus-vinyl dispute. |
| `rc_eace19540d2d` | Damaged cabinet doors/drawers / `cabinets_damaged_or_water_stained` | unsupported / absent | 9 / 2 | Both responses negate broken components/water staining. An adjacent alternative term exists, but component damage is also being denied. |

All six rebuilt request fingerprints match the stored requests. The eight
condition-evidence photos are full resolution (1200–1280 pixels wide), and
all eight match the recorded normalized RGB identities. **Precision correction
to the plan:** `image_sha256` here hashes dimensions plus decoded RGB pixels,
not encoded JPEG file bytes. This proves identical decoded images, not a
historical byte-for-byte comparison of JPEG files. Current encoded-file hashes
are also recorded, separately.

Related rows remain separate in the census. The material-dispute example
`rc_128caa6212b8` has a lifting/open-seam observation resolved to a vinyl item;
Terra describes hard plank material instead. The trim miss `rc_b642fe69b86a`
and correct-rejection control `rc_73d15ca14405` both have rationales contrasting
substantial/decorative trim with plain/thin trim. D1 already routes trim to
`no_action`; this closeout does not reopen it. Weak-label provenance applies
to all six, not just an inconvenient subset.

## 2. Candidate mechanisms across the full census

The [script](../scripts/analysis/terra_miss_mechanisms.py) joins **1,047/1,047**
run_1 reviews to conditions, evidence, calls and the historical 3.1 catalog
(`git show 07ee112:tools/issue_catalog_kind_v2.json`). There are 148 calls,
882 supported, 136 unsupported and 29 cannot_assess conditions. The human
comparison is **47 binary dirB cards: 20 misses and 27 correct rejections**,
across 20 properties and 34 calls. Four inconclusive dirB cards are retained
separately; they cannot supply a binary control label. Production cards join
their exact reviewed run, never a newer artifact.

Lexical screening uses explicit morphology groups and terms after a negation
within a clause. “Adjacent any” flags a negated catalog term absent from the
observation. “Adjacent only” additionally requires that no observed claim term
was negated in the same rationale. Every clause is exposed in the JSON. These
are imperfect screening proxies, not semantic adjudications. Visibility
hedging and the failed factorized rubric are excluded as candidate mechanisms.

| Screen | All 47: flagged misses / flagged correct rejections | Excluding six: flagged misses / flagged correct rejections |
| --- | --- | --- |
| Adjacent any | 14 / 13 | 8 / 13 |
| Adjacent only | 9 / 8 | 6 / 8 |
| No shared claim/observation mechanism term | 4 / 4 | 2 / 4 |

Among the remaining 14 misses, adjacent-any flags 8 (57.1%); among 27 correct
rejections, it flags 13 (48.1%). The apparent separation is small after the
six selected cases are removed. It also varies by source: canary adjacent-any
flags 11/16 misses and 6/17 correct rejections; production flags 3/4 and 7/10.
The script supplies source-separated and six-excluded tables rather than
treating the pooled association as a general recovery rule.

| Batch condition count | Human misses / binary dirB cards | Census not-supported / conditions |
| --- | --- | --- |
| 1 | 0 / 0 | 1 / 10 |
| 2–4 | 4 / 7 | 20 / 140 |
| 5–8 | 4 / 11 | 51 / 286 |
| 9+ | 12 / 29 | 93 / 611 |

| Batch photo count | Human misses / binary dirB cards | Census not-supported / conditions |
| --- | --- | --- |
| 1 | 7 / 11 | 32 / 202 |
| 2–3 | 4 / 13 | 43 / 296 |
| 4+ | 9 / 23 | 90 / 549 |

There is no consistent increasing batch-size pattern. Rows sharing a call or
property are correlated, and the targeted dirB slice cannot estimate recall.
These measurements do not establish that splitting calls would recover misses
without reviving correct rejections.

**Conclusion: no justified experiment from this bounded screen.** This does
not prove every alternative recovery mechanism unsafe or explain away the six
misses. The proposed experiment allocation is zero. The directional replica
floors (supported→not 4.05%, unsupported→supported 23.1%) remain context for
the planned acceptance; they excuse no individual lost case. No new control
design is proposed without a supported hypothesis beyond the selected six.

## 3. Corroboration gate tradeoff

The [second script](../scripts/analysis/corroboration_gate_tradeoff.py) joins
**125/125 v1.1 cards and 12/12 rc_ hallucination cases**, a deduplicated union
of **128 cards**, with zero unresolved joins. Three hallucination cases have
no v1.1 two-axis label and remain explicitly inconclusive; they are not
silently upgraded to hard-false labels. Each row records class, distinct views,
original disposition, membership in both populations and simulated outcome.

The default twelve-item family set is exactly the plan's union of RESULT §9.4
and review_analysis §3's by-item rows, minus `dated_interior_trim` and CAP-007's
retired **`dated_window_treatment_valance`** parent. `worn_or_stained_flooring`
is still a live item and is included. No successor is guessed from old ids.
`--families` permits a per-item decision. The production disposition function
is reused: a rejected or already-unbilled condition is never counted as newly
withheld. Distinct views, not photo filenames, control the threshold.

| Label class | Newly withheld | Not newly withheld | Of the latter, already not billed |
| --- | --- | --- | --- |
| Absent | 6 | 3 | 0 |
| Misnamed | 2 | 7 | 1 |
| Trivial | 2 | 4 | 0 |
| Supported and warranted | 13 | 86 | 19 |
| Inconclusive / no repaired label | 1 | 4 | 0 |

Source matters: **canary catches 1 absent + 2 misnamed + 2 trivial while
withholding 10 legitimate conditions**; production catches 5 absent while
withholding 3 legitimate and 1 inconclusive condition. These are reviewed
outcomes, not population precision or recall. Two views establish evidence
coverage, not independent corroboration of the condition.

| Item | Absent withheld | Misnamed withheld | Trivial withheld | Legitimate withheld | Inconclusive withheld |
| --- | --- | --- | --- | --- | --- |
| cabinets_dated_style | 1 | 0 | 1 | 1 | 0 |
| older_flooring_style | 1 | 0 | 1 | 1 | 0 |
| peeling_or_discolored_paint | 2 | 1 | 0 | 1 | 0 |
| worn_or_stained_carpet | 2 | 0 | 0 | 1 | 0 |
| patio_or_porch_surface_wear | 0 | 1 | 0 | 1 | 0 |
| hard_flooring_scratched_or_worn | 0 | 0 | 0 | 2 | 1 |
| bath_fixtures_stained_or_worn | 0 | 0 | 0 | 1 | 0 |
| floor_dirty_or_heavily_soiled | 0 | 0 | 0 | 1 | 0 |
| wall_scuffs_marks_or_dents | 0 | 0 | 0 | 4 | 0 |
| baseboard_wear_scuffs / outdated_bathroom_finishes / worn_or_stained_flooring | 0 | 0 | 0 | 0 | 0 |

Full per-item withheld/**not withheld** tables and all original cards are in
the [tradeoff report](../reports/corroboration_gate_tradeoff_20260910.md) and
[JSON](../reports/corroboration_gate_tradeoff_20260910.json). Run_1 has **377**
unlabelled single-view billed conditions, **163** in the default families;
each is listed. Zero reviewed effect on an item does not imply zero unreviewed
effect. “Unlabelled” here means absent from the v1.1/rc-hallucination union.

The prior three-item statement reproduces: blanket removal of
`older_flooring_style`, `cabinets_dated_style`, `dated_interior_doors` removes
6 supported-and-warranted and 6 trivial cards (plus 2 absent). That was an
item-exclusion inventory, **not** the measured effect of a two-view gate.

Recommendation: keep the gate off for acceptance. A narrower choice such as
paint/carpet has a more favorable observed tradeoff, but still loses legitimate
work and has limited reviewed coverage. Steven may select items from the table;
adoption requires generator support for `min_photo_evidence`, regeneration and
Tier 1 proof before Batch 1. No catalog edits have been made.

## 4. Temperature and production routing

`run_pass_2d` now applies `setdefault("temperature", PASS_2D_TEMPERATURE)` to
its copied request kwargs. The config defaults to **0.1**; explicit per-call
values still win, including zero. Neither the prompt nor shared Qwen config
changed. The benchmark loader now records the effective local temperature.
The audit harness sampling note reads the config rather than claiming 0.2.

The newest three complete production artifacts (2026-08-25) all record 2d as
`unsloth/qwen3.6-27b@q6_k`, source `standard_default`. `workerMode.ts` confirms
that plain `--premium` overrides 2d to the upstream Terra model; adding
`--local-2d` restores local routing. Historical production artifacts therefore
support Qwen, while the default premium invocation alone does not pin it.
Recommendation: acceptance explicitly pins Qwen@0.1 and the eventual production
smoke verifies that effective route. A live worker's future launch flags are
not established by old artifacts.

Transport-boundary tests capture synthetic HTTP payloads with temperatures
**2d 0.1 / 1a 0.2 / summarizer 0.2**. The OpenAI text adapter receives no
temperature. These probes use mocked HTTP, not paid or synthetic-photo model
calls. See [verification record](../reports/backend_acceptance_session_a_verification_20260910.json)
for payloads, routing provenance and benchmark comparisons.

| Live slice | Cases × replicas | Final ID accuracy | Recall@5 | Unstable rows | Failures |
| --- | --- | --- | --- | --- | --- |
| dev | 62 × 3 | 98.28% | 100% | 0 | 0 |
| holdout | 63 × 5 | 100% | 100% | 0 | 0 |
| checkpoint_20260908 | 2 × 3 | 100% | 100% | 0 | 0 |

Dev repeats the stored Terra baseline's single mismatch (`res-moi-002`,
generic mold gold vs bathroom mold selection), 3/3; no dev selection changed.
The stored Qwen holdout has 59 cases, not today's 63. Its one changed common
selection is `res-mod-010`: the layout item now returns null, matching the
authorized layout retirement and re-golding in `58c073e`. The four added
layout hard negatives also pass. Catalog fingerprints differ, so this is a
regression comparison, not a temperature-only causal experiment. Raw results
are preserved in the [benchmark archive](../reports/backend_acceptance_session_a_benchmarks_20260910.json).
This supplies the previously missing S7-22 benchmark scores.

The acceptance plan's “1 unstable of 93” should not be used as the unchanged
prompt's promised stability. The earlier closeout assigns that figure to the
rejected candidate prompt and reports 7 unstable rows for its base prompt.
This session's zero-unstable result is measured on the benchmark slices above;
it does not overwrite the older 93-row experiment or predict whole-property
acceptance stability.

## 5. Prepared reliability changes and required rulings

The [reviewable patch](../reports/backend_acceptance_reliability_proposal_20260910.patch)
is prepared but **not applied**:

1. Serialize the artifact before opening its temporary file; normalize unpaired
   UTF-16 surrogate code points to U+FFFD while preserving ordinary Unicode and
   valid surrogate pairs; then retain atomic replacement. This addresses the
   C5 `property.description` write failures without changing model engines.
2. In `new` mode, require a nonblank explicit `RENOVATION_SOL_MODEL` during
   configuration initialization instead of inheriting `OPENAI_MODEL`. Keep
   current/shadow behavior unchanged. Set the acceptance value to `gpt-5.6-sol`.

| Ruling | Recommendation | Status |
| --- | --- | --- |
| Six cases | Accept as unresolved misses; no further Terra experiment justified by this screen | Awaiting Steven |
| Gate | Off; any item adoption must name the items from the table | Awaiting Steven |
| Production 2d | Explicit local Qwen `unsloth/qwen3.6-27b@q6_k` at 0.1 | Awaiting Steven |
| Reliability fixes | Apply both prepared changes in Session B, with regression tests | Awaiting Steven |

These are the decision gates explicitly required by the supplied plan (§4 A6
and §6). Session B's replay construction/proof and Sessions C–E remain owed.
No new Codex task or session was created. No GO/NO-GO/RETAIN decision is claimed.
Fresh-2a variance, FE/pricing deferrals, electrical quarantine, D1–D8 and the
shared-budget requirements remain as specified in the plan.

## 6. Reproduction

Run the two analysis scripts with `.venv/Scripts/python.exe`. Neither script
makes model calls. They fail on missing source joins
and record source hashes. Re-running both produces byte-identical JSON and
Markdown outputs.

Targeted validation covers scene passes, attribution, the analysis semantics
and the catalog benchmark harness. Live benchmark slices use local Qwen,
`LM_STUDIO_URL=http://127.0.0.1:1234`, `PASS_2D_TEMPERATURE=0.1` and the Jina v5
embeddings sidecar. The existing sidecar was stopped; the cached Q8_0 model was
started for this work and real embeddings POSTs succeeded. No fallback encoder
or old prompt was used. Exact scores and comparability limits are retained in
the verification record rather than implying unchanged historical gold/catalogs.
