# Handoff: Pass 2c absence safety — decision recorded, safety shipped, prompt strategy retired

## Status

Branch `benchmark_system`. Opened 2026-07-31 as a decision, closed 2026-08-04 as a change,
narrowed 2026-08-04 to the absence-safety half.
Decision: **option (c), benchmark first.** Absence claims stay debug-only until human-scored
benchmarks show that reporting them improves accuracy. What shipped is the corpus audit
(work item 1) and a catalog-level safety fix the original write-up did not anticipate needing
(work item 4) — see "What the corpus showed".

**The 2c prompt edit (work item 2) was shipped and then reverted.** It routed worn/weathered
exterior finishes to `upgrade_candidate`, which is a `kind`-semantics decision — and `kind` is
being reworked wholesale into `defect | degradation | modernization` by the semantic-overhaul
work. Settling where exterior wear lands under a two-value ontology that is about to be replaced
buys nothing, and the A/B behind it is not reproducible (see below). `PASS_2C_SYSTEM_PROMPT` is
byte-for-byte back to its `db9484c` state; the overhaul session starts from a clean prompt.
See "Deferred" at the end.

Context: a customer reported missing outdoor findings on 131 Alex Ln (`redfin_126975740`). The
root cause was packaging, not detection, and that is fixed. While tracing it, a *separate*
recall gap surfaced in Pass 2c that is worth understanding before anyone touches the 2c prompt.

## The observation

On `photo_009.jpg` (run `20260731_101711_1dc6f7ee`), Pass 2a/2b produced 12 grounded
observations and Pass 2c forwarded 4. Terra's output was clean — every observation is verifiable
in the image, no confabulation. Among the 8 dropped:

- "No clearly functioning gutter system is visible along the porch edge." → labeled `other`
- "The downspout arrangement appears limited." → labeled `other`
- "An outdoor HVAC condenser is located close to the house in a confined area." → labeled `other`

Two structural reasons, both in
[scene_classifier_passes.py:924-947](../tools/scene_classifier_passes.py#L924):

1. **No rule covers the absence of an expected component.** Every allowed label describes a
   thing that is present: `defect_or_damage` (visible wear/damage/missing on an observed item),
   `upgrade_candidate` (a dated fixture/finish), `good_condition`, `generic_presence`. An
   observation whose content is "X is not there" has nowhere to go but `other`.
2. **The `upgrade_candidate` rule enumerates interior finishes only** —
   "floors, cabinets, counters, fixtures, tile, paint"
   ([:939](../tools/scene_classifier_passes.py#L939)). Siding, trim, soffit, decking, and
   masonry have no example anchor, so exterior findings drift toward `generic_presence`/`other`.
   Still true after the revert — and now the semantic overhaul's problem, not this handoff's.

## Binding constraint — do NOT map absence onto the existing gutter items

The tempting fix is to route "no gutter visible" to `clogged_or_damaged_gutters` or
`gutter_maintenance_needed`. **This is semantically wrong and must not ship.** Three different
claims:

| Claim | Meaning | Cost implication |
|---|---|---|
| absent | no gutter system exists | install new — largest scope |
| damaged | a gutter exists and is broken | repair/replace a component |
| maintenance | a gutter exists and is dirty | service call |

Collapsing them produces a confidently wrong scope and price. Worse, "not visible in this photo"
is not the same claim as "not present on the building" — a gutter can be out of frame, behind
vegetation, or on the far elevation. An absence claim from a single photo is an inference about
the world from missing evidence, which is precisely the reasoning pattern this pipeline is built
to refuse.

## The decision, recorded: (c) benchmark first

The three options were:

- **(a) Keep as a note.** Absence observations stay out of the issue lane and surface, if at all,
  as an inspection/planner note. Cheapest, zero hallucination surface, no dollars.
- **(b) Introduce a narrowly-defined issue** (e.g. `gutter_system_absent_or_inadequate`) with
  strict evidence rules: requires a roofline visible at an eave edge in the photo, must not fire
  when any gutter run is visible anywhere in the property's photo set, and — given it would be a
  *new* absence-shaped catalog concept — probably `inspection_risk` scope rather than a priced
  repair.
- **(c) Leave unsupported** until benchmark evidence shows the recall loss actually costs
  accuracy. The benchmark system on this branch is the natural place to prove or refute it.

**Chosen: (c).** Absence claims remain debug evidence only. No new catalog item, no planner-note
lane, and no priced absence repair is authorized by this handoff.

**Any future `gutter_system_absent_or_inadequate` proposal is gated on all of:** a completed
benchmark runner *and* evaluator; human reference scoring that shows a whole-property absence
finding is required or acceptable; property-level contradiction checks rather than single-photo
inference; and `inspection_risk` scope, unpriced. It must never map onto
`clogged_or_damaged_gutters` or `gutter_maintenance_needed`.

## What the corpus showed

Measured with `scripts/audit_pass2c_funnel.py` over 824 latest-run properties (822 parse; 2
malformed debug artifacts under `redfin_10937111` and `redfin_10993751`):

```
observations=216,299  forwarded=66,666  rate=30.82%
absence: obs=276  forwarded=179  unresolved=131  properties=200  contradictions=115
absence->gutter_item=26
```

Forward rate by scene group: exterior 33.31%, kitchen 34.95%, utility 33.93%, bathroom 32.51%,
bedroom 29.00%, living_areas 27.38%, other 22.07%. Exterior is *not* an outlier on the funnel —
the recall gap is about which exterior observations get made forwardable, not a group-wide drop.

Two findings changed the plan:

1. **The forbidden mapping was already live.** 26 of the 173 historical resolutions onto the two
   gutter items were absence-shaped. The constraint below was being violated in production.
2. **Pass 2d was already doing most of the work.** 131 of the 179 forwarded absence observations
   resolved to nothing at all. The residual defect was 26 occurrences, not 179.

So the guard went into the **catalog**, not Pass 2c. Suppressing absence language at 2c would
have dropped ~179 forwarded observations to stop 26 bad resolutions — 85% collateral — and would
have contradicted constraint #2 below. `deny_any` is a hard reject at candidate retrieval
([catalog_embeddings.py:447](../tools/catalog_embeddings.py#L447)) applied to the observation
text, which is exactly where the forbidden mapping occurs.

Deny terms landed in two waves. The first wave (`no visible`, `no downspout`, `no downspouts`,
plus `missing gutter` / `missing gutters` on `clogged_or_damaged_gutters`) blocked 18 of 26 and
left 8 residuals. The second wave closes the phrasings those 8 used, and is now on **both**
gutter items:

```
lack of visible          no obvious functioning       appear limited or poorly detailed
no well-maintained       appears limited or absent    no clearly functioning
not visible              not apparent
```

Six of those are corpus-fitted. `no clearly functioning` covers the customer phrasing that opened
this handoff. `not visible` / `not apparent` are the one deliberate over-reach: they block **zero**
additional historical resolutions and exist to cover the `component_not_visible` pattern family
(30 corpus observations) before it lands a resolution. Their exposure was measured — of the 32
gutter observations any of the three broad terms touch, 31 are absence-shaped and the one that is
not ("Gutters appear functional along the roofline, though their full condition is not visible")
is not a damage claim and should not reach a damage item anyway.

Replayed over every historical gutter resolution — **173 total = 26 absence-shaped + 147
legitimate** — the completed set blocks **24 of 26 absence-shaped and 0 of 147 legitimate.**

**Two absence-shaped strings deliberately still resolve.** They assert absence and damage in
one sentence, and blocking them would lose a real clogged-gutter finding:

- *"The roofline appears uneven with missing or poorly installed gutters."*
- *"Gutters appear clogged or missing downspouts along the roofline."*

Do **not** add a bare `missing` or `appears limited` to `clogged_or_damaged_gutters` to chase
them; each buys 1–2 blocks and both are semantically loose. Note the shipped
`appears limited or absent` is phrase-scoped for exactly this reason — the bare stem would have
swallowed both residuals. `tests/test_catalog_validation.py::test_mixed_absence_and_damage_claims_stay_eligible`
enforces the outcome; `::test_gutter_deny_lists_stay_narrow` pins the two bare stems out of the list.

The deny set was fitted on the corpus it was validated against. That is acceptable only because
the failure mode is asymmetric — a deny term can only *prevent* a mapping already ruled
semantically wrong — and because `--fail-on-absence-resolution` re-measures rather than assumes.

## The prompt widening did NOT improve recall — measured, but NON-GATING

**Read this section as history, not as evidence.** It is retained because its *negative* finding
(a prompt-level absence rule costs recall) is what kept absence safety out of the 2c prompt, and
that decision still stands. Its *positive* finding — the defect→upgrade reclassification — must
not be used to justify anything, for two reasons:

- **Not reproducible.** The `ab_2c_prompt.py` harness and every per-run output lived only in a
  session scratchpad and were not preserved. The numbers below cannot be re-derived or extended.
- **Wrong model tier.** The A/B ran on `gpt-5.6-terra`. Production Pass 2c defaults to the
  `standard` profile — all Qwen ([pipeline_config.py:174-175](../tools/pipeline_config.py#L174)) —
  and a per-pass model only reaches Terra via an explicit `--model-map`. A labeling-behaviour
  measurement on one model tier does not transfer to the tier that actually runs.

Anyone re-opening 2c prompt work should re-create the harness under `scripts/` (~120 lines) and
re-measure on the model the pipeline actually uses.

The interior-only `upgrade_candidate` examples looked like "a plain gap… the low-risk half of
this handoff". A controlled A/B says otherwise, and the write-up above was wrong to assume it.

Method: replay the **same 266 Pass 2b observations** (this property's stored `labeled_debug`)
through Pass 2c under each prompt, same model (`gpt-5.6-terra`), same scene. That isolates the
prompt from Pass 2a/2b run-to-run variance, which dominates any whole-pipeline comparison — a
naive re-run showed 53% → 43% forwarding, almost all of it 2a rewriting its own observations.

Aggregates over five paired runs; the ±band is the observed run-to-run spread on a fixed prompt.

| Prompt | forwarded / 266 | upgrade_candidate | defect_or_damage | other |
|---|---|---|---|---|
| pre-change | 141–144 (~143) | 76–81 (~78) | 63–66 (~64) | 33–41 |
| anchors + a whole-system-absence rule | 129–131 | 98 | 31 | 47 |
| **shipped** — anchors, no absence rule | 135–138 (~137) | 98–107 (**~102**) | 31–35 (~32) | 35–40 |
| anchors scoped to "dated" only | 142 | 79 | 63 | 37 |
| shipped + widened interior enumeration | 138 | 106 | 32 | 40 |

Three findings, all counter to the original write-up:

1. **The exterior anchors do not increase forwarding — they reclassify.** The +24
   `upgrade_candidate` is almost entirely `defect_or_damage` → `upgrade_candidate` (−32). Net
   forwarding is *down* ~6. Scoping the anchor to "dated" (row 4) returns everything to parity,
   which is the proof that the reclassification is what the wording buys. This changes `kind`,
   and therefore catalog routing, severity and price — not recall.

   **The reclassification was intended, and is now deferred.** Steven's call, 2026-08-04: a
   weathered exterior finish is an upgrade, not a defect, because it is not an immediate repair.
   The live re-run showed the consequence: "wood siding appears weathered, with staining" went
   `defect`→`exterior_siding_discoloration_fading` before, `upgrade`→`curb_appeal_upgrade` after.
   That intent stands — but it is a statement about what `kind` *means*, so it belongs to the
   three-kind overhaul, expressed once against the new ontology rather than twice against the old
   one. The prompt lines are reverted; the intent is recorded here.

2. **A prompt-level absence rule costs recall.** "A claim that a whole system is absent… label
   other" over-generalised from gutters to kitchen fixtures — "No visible modern vent hood is
   present over the range", "No apparent task lighting is visible" — where absence *is* a
   legitimate priced upgrade. It cost ~8 forwarded observations to suppress 3 gutter claims the
   catalog deny lists already handle. There is **no absence rule in the 2c prompt**; absence
   safety rests entirely on the catalog deny lists.

3. **The ~6 forwarded observations lost are a real, accepted cost.** They drop into
   `generic_presence`/`other` and rotate between runs, so they are borderline rather than a
   systematic hole — but "Darkened siding and exposed porch components are visible" was lost in
   every run, which is the uncomfortable one. Widening the interior enumeration with
   trim/windows/doors/ceilings/lighting (row 5) was tried as a recovery and made no difference
   (138 vs 138 forwarded), so those words were not kept.

`test_pass_2c_prompt_has_no_whole_system_absence_rule` pins finding 2, which is the one that
still binds. The test pinning finding 1 went with the revert.

**Deferred to the overhaul, not resolved:** the ~6-observation forwarding loss was never chased
to a root cause, and the defect→upgrade shift never had a pricing review. `curb_appeal_upgrade`
and `exterior_siding_discoloration_fading` do not carry the same cost model, so corpus-wide
`final_rehab` will move whenever a re-analysis rolls through under whatever the new ontology
decides. Run `scripts/audit_exterior_estimate_coverage.py` before/after a re-analysis batch to
size it.

## Binding constraints for any 2c prompt edit

1. **2a stays minimal.** The 2a prompt is deliberately `"What stands out here to a renovator"`
   ([:776](../tools/scene_classifier_passes.py#L776)). Naming target defects there primes
   confabulation and suppresses real detections — a property of the prompt, verified repeatedly
   against local models. Recall work belongs at 2c (a text-only labeling pass over what 2a
   already said), not upstream.
2. Precision control lives downstream of 2c: 2d's hallucinated-ID validation, 2e's speculation
   and deny gates, and 2f visual verification. Recall lost at 2c is unrecoverable; precision
   errors at 2c get three more chances to die. Bias 2c toward forwarding.
3. Any new label value must be added to `VALID_LABELS`
   ([:953](../tools/scene_classifier_passes.py#L953)) or `_coerce_labeled_2c` silently coerces it
   to `other` — a change that would look like a no-op prompt edit and fail invisibly. This is the
   sharpest trap waiting for the three-kind work.
4. **There is no `PASS_2C_PROMPT_VERSION`.** Only Pass 2f is versioned and hashed into artifact
   provenance (`PASS_2F_PROMPT_VERSION`). A 2c prompt change leaves no trace in the artifacts, so
   a re-analysed run cannot be told from an old one by inspection. If the overhaul changes 2c
   labelling, add a version constant first or the shadow comparison has no ground truth.

## Work items

| # | Change | Files | Risk | Status |
|---|---|---|---|---|
| 1 | Measure the 2c funnel by scene group across the corpus (read-only) | `scripts/audit_pass2c_funnel.py` | none | shipped |
| 2 | Widen 2c `upgrade_candidate` examples with exterior anchors (siding/trim/soffit/deck/masonry) | `scene_classifier_passes.py` | low | **reverted** — strategy retired, see Status |
| 3 | Record the absence-ontology decision (a/b/c) | this doc | — | shipped — (c) |
| 4 | Stop absence claims resolving onto the two gutter items | `issue_catalog.json` | low | shipped — 26 → 2 |
| 5 | Only if (b): new catalog item with strict evidence rules + tests | `issue_catalog.json` | high | **not authorized** — see gate above |

## Verification

```
.venv\Scripts\python.exe -m pytest tests/test_scene_classifier_passes.py tests/test_catalog_validation.py tests/test_audit_pass2c_funnel.py -q
```

Note the earlier revision of this doc named `tests/test_scene_classifier_stub.py`, which has never
existed. The real file is `tools/test_scene_classifier_stub.py`: an executable manual smoke script
with no `test_` functions, so pytest collects nothing from it, and it would fail if run (a bare
`import pipeline_config` at `:61` shadows the module-level import, and `:74` is a placeholder image
path). Do not put it in a pytest command.

Corpus audit:
```
.venv\Scripts\python.exe scripts/audit_pass2c_funnel.py --artifacts-root C:/Users/Steven/IntelliJProjects/renointel-prod/artifacts --json artifacts/pass2c_funnel_after.json --report docs/pass2c_funnel_audit.md --baseline artifacts/pass2c_funnel_before.json --fail-on-absence-resolution --max-absence-resolutions 2
```

Two things about that command that are easy to get wrong:

1. It reads `photo_intel_debug.json`. Pass 2c output is stripped from the slim
   `photo_intel.json`, so a walk over the slim artifact finds nothing.
2. **The gate replays the current catalog's deny lists over stored descriptions.** This is a
   deny-list replay, *not* a fresh embedding/retrieval simulation — `require_any` and the
   embedding score are not reconstructible offline, so the number answers "would this stored
   resolution still be allowed?" and never "what would retrieval produce today?". A stored
   resolution records the catalog as it was when that run was analysed, so the raw historical
   count (26) cannot fall without re-analysing all 824 properties. The replayed count is the
   actionable one, and it moved 26 → 8 → **2**. `--max-absence-resolutions 2` is the documented
   allowance for the two mixed-claim residuals above; the default is 0, which is the right
   setting for any newly analysed corpus.

Expected output of the command above: `absence->gutter_item: stored=26 under_current_catalog=2`,
`delta absence_gutter_item_resolutions_live: -24 (26 -> 2)`, exit 0. The funnel totals
(216,299 → 66,666) are catalog-independent — if they move, the corpus changed, not the catalog.

`artifacts/pass2c_funnel_before.json` is the pre-deny baseline. It is not regenerable: its
`catalog` field points at a session scratchpad copy of the pre-change catalog that no longer
exists. To rebuild an equivalent, replay a historical catalog over the same stored corpus:
```
git show db9484c:tools/issue_catalog.json > %TEMP%/issue_catalog_before.json
.venv\Scripts\python.exe scripts/audit_pass2c_funnel.py --artifacts-root C:/Users/Steven/IntelliJProjects/renointel-prod/artifacts --catalog %TEMP%/issue_catalog_before.json --json artifacts/pass2c_funnel_before.json
```

Prompt changes are not covered by unit tests in a meaningful way — re-run a property and diff the
2c funnel. Restart `analyzer_server.py` first; it is long-running and keeps old code. Then a full
analyzer run on `redfin_126975740` into a scratch artifact root, comparing `debug.labeled_debug` →
`labeled_forward` per scene group against run `20260731_101711_1dc6f7ee`. The re-run is a live LLM
pass and is not bit-reproducible; judge it on substance, not string equality.

## Gotchas

- **The embeddings sidecar reports healthy while being completely broken.** A re-run of
  `redfin_126975740` on 2026-08-04 aborted at `embeddings_init` with
  `decode() failed: vk::Queue::submit: ErrorDeviceLost`. `GET 127.0.0.1:8081/health` returned
  **200** the whole time; only `POST /v1/embeddings` failed, on every request. The sidecar is
  started `--n-gpu-layers 0` and still initialises a Vulkan backend, so it can lose the device
  while the 27B model has the GPU. Recovery is a process restart — health-checking it is not a
  sufficient readiness probe. Pass 2d fails closed here, so the whole run dies before any pass
  executes (and before any paid API call is made).
- Lexical absence cohorts are **review queues, not human truth**. Cohort size swings 276→370 on
  pattern wording alone, so the audit embeds its pattern set in every JSON snapshot. Never quote a
  cohort count without it.
- The `245 properties` figure quoted in early exploration was the count of properties *exhibiting*
  the pattern, never the corpus denominator. The denominator is 824.
- `scene_group` is stamped onto `labeled_forward` rows only and never onto `labeled_debug`, so a
  forward rate must be grouped by `photos[k].scene.group`.

## Deferred — owned by the semantic-overhaul session

Nothing in this handoff touches `kind`. It changed no `label`, no `kind`, no catalog ID, no
pricing, no artifact schema, and no API enum. That was the point: absence safety is
ontology-independent, so it can land ahead of the overhaul and stay correct through it.

The following are explicitly **not** decided here and belong to the parallel session:

- **The three-kind classifier** — `defect | degradation | modernization`, with maintenance
  classified as degradation. Everything this handoff learned about exterior wear (the intent
  recorded above) should be expressed once against that ontology.
- **Catalog migration** of every item's `kind`, and the downstream semantics that read it —
  severity, routing, estimate scope, pricing.
- **Rollout**: shadow-first, with v3 artifacts immutable and fresh v4 analyses rather than
  reprojection.
- **Prompt provenance**: add a `PASS_2C_PROMPT_VERSION` before changing 2c labelling, per binding
  constraint 4 above.

Two things this handoff produced that the overhaul should inherit rather than redo:

1. `scripts/audit_pass2c_funnel.py` measures the 2c funnel across the corpus read-only and is
   ontology-agnostic — it counts labels and joins resolutions, so it keeps working under new
   `kind` values and is the natural before/after instrument for a shadow rollout.
2. The gutter deny lists are a *semantic* guard (absent ≠ damaged ≠ maintenance), not a `kind`
   guard. They must survive the migration intact. `--fail-on-absence-resolution` with the
   default allowance of 0 is the check to run against any freshly analysed v4 corpus.

**Still a review queue, not truth:** the absence and exterior-finish cohorts are lexical. Cohort
size swings 276→370 on pattern wording alone. Never quote a cohort count without the `patterns`
manifest embedded in the JSON snapshot, and never treat a cohort as human-scored labels.
