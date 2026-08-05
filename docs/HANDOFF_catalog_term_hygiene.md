# Handoff: catalog term hygiene after word-anchored matching

## Status / framing

The **code** half is done and shipped on `property_overview_page_contract`. Catalog keyword
terms are no longer matched by raw substring containment; a single matcher,
`term_matches` ([tools/pipeline_common.py](../tools/pipeline_common.py)), anchors every term
at a word start (`\b` + `re.escape(term)`, **no trailing** `\b`).

Sites converted:

| site | field | was |
|---|---|---|
| `CatalogEmbeddingsRetriever._passes_guardrails` ([catalog_embeddings.py:441](../tools/catalog_embeddings.py#L441)) | `deny_any`, `require_any` | `x in t` |
| `_resolve_candidate_via_lexical_shortcut` ([scene_classifier_passes.py:1130](../tools/scene_classifier_passes.py#L1130)) | `support_any` | `phrase in observation_lower` |
| `_collect_signal_hits`, `_has_high_signal_damage` | routing / damage tokens | inline `re.search`, recompiled per call |

Measured effect over a corpus of **4,572 unique observations** harvested from the 21
`catalog_audit_*.json` runs in the repo root:

- guardrails: 322 spurious decisions removed, 142 wrongly-blocked observations released,
  **no `require_any` item dropped to zero corpus matches**
- 2d shortcut: 624 spurious `support_phrase_hit` opportunities removed (that hit bypasses
  the Pass 2d LLM call entirely, so each one was a silent misroute)

Sixteen catalog terms were added at the same time to preserve matches that had been riding
on the old substring behavior (`drywall`, `repaint`, `outdated`, `outdated light/tile/
wallpaper/style/outlet`, `bathtub`, `backyard`, `doorknob`, `undercarport`).

**The data work described below is now also done.** See "Outcome" — §1 and §2 are complete,
§3 and §4 are deliberately deferred with their measurements recorded. The audit is no longer
a snippet: it is [scripts/audit_catalog_terms.py](../scripts/audit_catalog_terms.py), which
writes [catalog_term_audit.md](catalog_term_audit.md).

---

## Outcome (completed)

Run the audit any time to reproduce these numbers:

```bash
.venv/Scripts/python.exe scripts/audit_catalog_terms.py --report docs/catalog_term_audit.md
```

**§1 sweep — the 16-term patch held.** Exactly one term had lost all its matches to
anchoring: `dated outlet` on `dated_electrical_outlets_switches`, whose only corpus hit was
inside "outdated outlets". The item already carried `outdated outlet`, so the dead term was
removed rather than replaced.

Two `require_any` gates were found closed against observations the corpus actually contains —
the "too tight" direction, and the reason to sweep gates rather than only terms:

| item | added | why |
|---|---|---|
| `trees_or_vegetation_too_close` | `shrub`, `bush`, `hedge` | Its own description is "vegetation contacting or very close to siding/roof/foundation", but the gate carried only `tree`/`vine`/`ivy`. "Overgrown shrubs are touching the brick foundation area" and ~25 similar observations never reached it. |
| `layout_modernization_opportunity` | `layout` | Every term was authored adjective-noun (`cramped layout`, `choppy layout`) while the VLM writes predicate order ("the layout is cramped with limited counter space"). All 14 terms scored zero; the item was unretrievable. |

`require_any` should err loose and `deny_any` tight: a loose gate only keeps an item
*eligible*, leaving embedding rank and Pass 2d to decide, whereas a tight gate makes the item
permanently invisible. Neither addition steals from a competing item —
`landscape_improvement_needed` has no `require_any` at all, so it stays eligible regardless.

**§2 — see the corrected section below.** Four terms now carry the whole-word marker.

**Still knowingly dead:** `pest_or_rodent_evidence` matches nothing, and that is correct —
`pest`, `rodent`, `droppings`, `nest`, `termite`, `infestation` and `mouse` have zero corpus
hits because pests are not visible in listing photos. Its only prior "matches" were four
false hits on `rat` inside "rather". Same for `leak` (0 hits): photos show staining, not
leaking. Do not chase these; they are a corpus property, not a term bug.

## Sections

### 1. Sweep the terms the corpus never exercised — **done**

The catalog carries 649 `support_any` and 252 `deny_any`/`require_any` term instances (674
distinct). The corpus only exercises a fraction of them, so the 16-term patch above covers
only collisions that actually fired in real data. Every unexercised short term is an untested
stem — **396 instances have zero corpus hits** (294 support, 53 deny, 48 require).

Two failure directions to check per term:

- **too loose** — the term is short enough to sit inside a common longer word. Word-start
  anchoring fixed the *interior* collisions (`ding`⊂`siding`, `aged`⊂`damaged`,
  `rat`⊂`discoloration`, `shed`⊂`unfinished`, `trip`⊂`stripped`, `eave`⊂`leaves`,
  `range`⊂`orange`, `rot`⊂`protection`) but not prefix collisions — see §2.
- **too tight** — the term is a stem the corpus happens to use only in a prefixed form
  (`wall` where observations say `drywall`, `tub` where they say `bathtub`). Anchoring makes
  these silently stop matching. Author the prefixed form explicitly; the code stays dumb and
  the data stays authoritative.

This is no longer a snippet. [scripts/audit_catalog_terms.py](../scripts/audit_catalog_terms.py)
harvests the corpus and reports, per `(item, role, term)`, anchored vs whole-word vs raw
substring hits, interior collisions, prefix extensions with their enclosing words, zero-hit
terms, and the per-item `require_any` viability roll-up. It calls the real `term_matches`
rather than reimplementing the regex, so the report cannot drift from production.

```bash
.venv/Scripts/python.exe scripts/audit_catalog_terms.py \
    --json artifacts/catalog_terms_before.json --report docs/catalog_term_audit.md
# ...apply term edits, then diff:
.venv/Scripts/python.exe scripts/audit_catalog_terms.py \
    --baseline artifacts/catalog_terms_before.json --report docs/catalog_term_audit.md
```

Two corpus gotchas it encodes, both verified:

- The three oldest runs (`catalog_audit_126224899_full/_v2/_v3.json`) write
  `gpt_observations` where the other 18 write `cloud_observations`. The default two-key tuple
  reproduces the 4,572 baseline quoted above; `--include-legacy-key` raises it to 4,879.
- `catalog_audit_test{,_126224899}.json` are 3- and 5-image smoke runs. They are included by
  default (again, to reproduce 4,572); `--exclude-smoke` drops them to 4,550.

Anything the corpus does not reach needs eyeballing instead — the report lists every zero-hit
term at or under 6 characters, which is where prefix collisions live. Longer multi-word
phrases carry near-zero collision risk under word-start anchoring.

### 2. The trailing-edge class anchoring cannot fix

Leading-only `\b` was a deliberate choice: full `\b...\b` was measured against the same
corpus and drops `require_any` to **zero** hits for ~12 items whose terms are singular stems
(`grout line`, `roof shingle`, `kitchen cabinet`, `handrail`, `branch`, `vine`,
`exposed wire`, `window screen`, `baluster`, `receptacle`, `missing tile`, `floral wall`);
`stain` alone falls 616 → 10. Those items would become unretrievable. Do not "tighten" this
without re-running the audit.

The cost of that choice is that a term still matches a longer word it *prefixes*. Measured
against the corpus, four of these were real and one was not:

| term | prefixes | corpus cost |
|---|---|---|
| `mold` | "crown molding" | **18 observations — as many as the 18 real mold observations** |
| `wall` | "wallpaper" | 46 |
| `tub` | "tube" | 6 |
| `rat` | "rather" | 4 — *all four* of its hits, on a `require_any` gate |
| `stain` | "stainless" | 1 — not chased; a marker here would cost 606 wanted matches |

`mold` was the severe one. It sits in `support_any` on `visible_mold_or_mildew`, and a
`support_phrase_hit` **skips the Pass 2d LLM call outright**, so every crown-molding
description was silently resolved as a high-severity moisture defect with nothing downstream
to catch it. It is also `deny_any` on four interior items, which deny-blocked those same
descriptions.

> **Correction.** An earlier revision of this section proposed fixing this class by splitting
> the term into explicit inflections (`"stain", "stains", "stained", "staining"`) to "gain the
> trailing boundary for that one term". **That does not work.** Under leading-only `\b`,
> `mold`, `molds` and `moldy` all still prefix-match `molding`. Every authored-phrase
> alternative was measured: `mold growth` recovers 4/18 real mold observations, `mold damage`
> 2/18, `black mold` 1/18. Only a trailing word boundary gets 18/18 real and 0/18 molding.

So the trailing boundary is **opt-in per term, authored in the data** as a trailing `$`
([pipeline_common.py](../tools/pipeline_common.py)):

```python
@lru_cache(maxsize=4096)
def _term_pattern(term: str) -> Pattern[str]:
    if term.endswith(TERM_WHOLE_WORD_MARKER):
        return re.compile(r"\b" + re.escape(term[:-1]) + r"\b")
    return re.compile(r"\b" + re.escape(term))
```

The global default is unchanged, so the ~12 stem items above are untouched. `$` is safe as
the sentinel — across all 674 distinct catalog terms the only non-alphanumeric character in
use was `-`. `catalog_validation` rejects `$` anywhere but the final character.

Applied to exactly four terms, across 12 item/role slots:

| term | hits | as stem | excluded | items |
|---|---|---|---|---|
| `mold$` | 18 | 37 | 19 × molding/molded | 2 support, 4 deny |
| `wall$` | 385 | 755 | 370 (`walls` **separately authored on both items**, so plural coverage holds) | 1 support, 1 require |
| `tub$` | 64 | 70 | 6 × tube | 2 support, 1 require |
| `rat$` | 0 | 4 | 4 × rather | 1 require |

Note `-` is not a word char to `\b`, so `mold$` still matches "mold-like" and `wall$` still
matches "wall-to-wall". That is intended: "mildew or mold-like buildup" is a real mold
observation.

**When you reach for the marker:** only where the excluded word is a *different concept*.
Do not mark terms whose prefix extensions are wanted inflections — the corpus is full of
legitimate ones (`cabinet`→"cabinetry" 50, `chip`→"chipped" 15, `uneven`→"unevenness" 25,
`roof`→"roofline" 14, `entry`→"entryway" 7, `structural`→"structurally"), and
`patterned wall`→"patterned wallpaper", `floral wall`→"floral wallpaper" and
`kitchen cabinet`→"kitchen cabinetry" are hitting their intended targets. Anything you mark
must not leak to a model: `strip_term_marker` is applied at every prompt-render site
(`format_candidates_text`, `_candidate_name_and_support_signal_terms`, and the auditor's
`current_keywords`).

### 3. Should coarse support terms trigger the 2d bypass at all?

`support_phrase_hit` skips the Pass 2d LLM call. Even fully anchored, these one-word support
terms auto-resolve on very thin evidence:

| item | term | corpus obs still hitting |
|---|---|---|
| `appliance_damage_or_missing` | `old` | 631 |
| `outdated_bathroom_finishes` / `outdated_kitchen_finishes` | `dated`, `old` | ~1,270 each |
| `worn_or_stained_flooring` | `worn`, `dated` | 1,514 |
| `wall_scuffs_marks_or_dents` | `wall` | 1,250 |

The only thing protecting that path is the score ≥ `PASS_2D_SHORTCUT_MIN_SCORE` (0.72) and
margin ≥ `PASS_2D_SHORTCUT_MIN_MARGIN` (0.03) gate above it
([scene_classifier_passes.py:1148](../tools/scene_classifier_passes.py#L1148)).

**Deferred — both proposed rules were measured and both fail.**

- **"≥ 2 words to be shortcut-eligible"** strips *all* `support_phrase_hit` eligibility from
  **17 items**, including `worn_or_stained_carpet`, `damaged_or_aged_roof_shingles`,
  `damaged_or_rotted_siding_or_trim`, `missing_or_damaged_handrails` and
  `wall_scuffs_marks_or_dents`. Too blunt.
- **"≥ N characters"** is a no-op: 0 items are affected at 5 or 6 chars, because eligibility
  is per-item and every item retains some qualifying term. Only at 8 chars does it start to
  bite (5 items), by which point it is arbitrary.

**It also cannot be measured offline.** None of the 1,781 files under `artifacts/` carry
`resolution_path` or `shortcut_reason` — those fields exist on `Pass2dResult` and are written
to the orchestrator's debug rows, but no stored artifact retains them. A before/after needs a
live pipeline run.

Pruning the generics per-item is the remaining option, but note `support_any` is **also**
concatenated into `estimate_scope._catalog_text`, so pruning silently moves money between
scope tiers as a side effect. If this is picked up, do it as its own change with a live A/B.

Meanwhile §2 removed the three worst offenders on this exact path — `mold`, `rat` and `wall`
no longer fire the bypass on molding/rather/wallpaper — which is most of the benefit without
an unmeasurable global rule.

### 4. Adjacent surface, now converted: `estimate_scope._contains_any` — **done**

> **Converted.** `_contains_any` now calls `term_matches`, `_catalog_text` applies
> `strip_term_marker` to `support_any`, and `mold`/`visible mold` carry the marker in
> `_REQUIRED_TERMS` and `_VISIBLE_REQUIRED_CONDITION_TERMS`. **There is no raw-substring
> keyword matcher left in the pipeline.** The measurements below were reproduced exactly
> (307 pairs, 4 items); the nine `estimate_scope` overrides that pin the affected items
> — and the wider curb-appeal sweep they prompted — are recorded in
> [catalog_estimate_scope_audit.md](catalog_estimate_scope_audit.md), "Pass 2".
>
> **This section missed the larger bug in the same code path.** `mold` prefix-matches
> "crown molding", and word-start anchoring does not fix prefix collisions — only
> interior ones. It promoted **14 cosmetic items** to `required_rehab` on the 19 corpus
> observations mentioning crown molding (266 pairs, versus 307 for the `finish` fix).
> Exactly the class §2 documents for the catalog's own terms; the code's term tuples
> needed the same treatment and had not been checked. Every other prefix extension of
> every scope term was verified against the corpus and all are wanted inflections, so
> `mold$` is the only marker warranted.

The original analysis, left for the record:


[tools/estimate_scope.py:573](../tools/estimate_scope.py#L573) still uses raw
`term in text`, and its `_catalog_text` concatenates `supporting_observations` — so free-form
VLM text does flow through it. Measured collisions in `_MARKETABILITY_TERMS`:

| term | matches inside | count |
|---|---|---|
| `finish` | "unfinished" | 86 |
| `older` | "holder" | 7 |
| `"old "` | "mold growth", "threshold and", "mold damage" | 12 |

This drives which line items land in `MARKETABILITY_REHAB`, i.e. which tier the money shows
up in.

**Deferred to [HANDOFF_catalog_scope_audit.md](HANDOFF_catalog_scope_audit.md)**, where
per-item `estimate_scope` overrides are already the prescribed method — but the blocker this
section describes is **already resolved**, and the conversion is far safer than assumed:

- `outdated`, `repaint` **and** `finishes` are all present in `_MARKETABILITY_TERMS` today,
  so the "must be added in the same commit" condition is already satisfied.
- Catalog text alone: **0 of 106** items change scope under `term_matches`.
- Across all 106 items × 4,572 observations: **307 of 484,632 pairs (0.063%)** flip, touching
  only **4** items — `gutter_maintenance_needed`, `driveway_or_walkway_cracking`,
  `trees_or_vegetation_too_close`, `brick_weathering_or_mortar_deterioration` — every one
  caused by `finish` no longer matching "unfinished".
- All 4 flip *out* of `marketability_rehab` (210 → `required_rehab`, 97 →
  `optional_value_add`). That exposes a real latent issue: `defect_default → required_rehab`
  is the wrong fallback for cosmetic exterior maintenance, and the substring bug was masking
  it. Those 4 items want explicit `estimate_scope` overrides in the same change.

## Verification

Terms are data, so the regression net is the test files that lock the code behavior plus the
catalog-backed assertions:

```bash
.venv/Scripts/python.exe -m pytest tests/test_pipeline_common.py tests/test_catalog_embeddings.py tests/test_scene_classifier_passes.py tests/test_catalog_validation.py -q
```

- [tests/test_pipeline_common.py](../tests/test_pipeline_common.py) — `term_matches` and
  marker semantics directly, including the hyphen-is-a-boundary case.
- `test_real_catalog_require_terms_survive_word_anchoring` and
  `test_real_catalog_marked_terms_do_not_fire_on_prefixed_words`
  ([tests/test_catalog_embeddings.py](../tests/test_catalog_embeddings.py)) read the real
  `issue_catalog.json`. **Extend the first with a case per term you add**, so the next
  anchoring change cannot silently un-match it. It asserts the item id actually has
  guardrails, because `_passes_guardrails` returns `True` for an unknown id and a typo'd case
  would otherwise pass vacuously.
- `test_pass_2d_marked_support_term_*`
  ([tests/test_scene_classifier_passes.py](../tests/test_scene_classifier_passes.py)) —
  `FakeTextClient.calls` is the real assertion: `0` proves the LLM was bypassed, `1` proves it
  was not.

Then re-run the audit and confirm the invariants:

```bash
.venv/Scripts/python.exe scripts/audit_catalog_terms.py --baseline artifacts/catalog_terms_before.json --report docs/catalog_term_audit.md
```

1. `unretrievable require_any items` is **1** — `pest_or_rodent_evidence`, knowingly dead
   (see Outcome). Anything else appearing there is a regression; the diff section flags newly
   unretrievable items explicitly.
2. `lost-to-anchoring` is **0**.
3. Marked terms report `mold$` 18, `wall$` 385, `tub$` 64, `rat$` 0, and the excluded words
   are only molding/molded, walls+wallpaper, tube, rather.
4. Every unmarked term's hit count is unchanged from the baseline snapshot.
