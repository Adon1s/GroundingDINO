# Handoff: catalog estimate-scope classification audit (the "long pole" for the renewal toggle)

> # ✅ COMPLETE — do not work from this document
>
> Both passes have shipped. The record is
> [catalog_estimate_scope_audit.md](catalog_estimate_scope_audit.md); the enumeration
> tool is [scripts/audit_estimate_scope.py](../scripts/audit_estimate_scope.py).
>
> - **Pass 1** (steps 1–5 below) — committed in `49a01de "Catalog hygiene audit fixes"`.
> - **Pass 2** — the `_contains_any` → `term_matches` conversion this document's
>   "Converting `_contains_any`" section deferred, plus a curb-appeal sweep of every
>   item landing in `required_rehab`.
>
> **The body below is stale on two counts.** It says "only ~4 items currently set an
> explicit `estimate_scope`" — the catalog now carries **23**. And its measured claim
> that the conversion touches only 4 items is correct but incomplete: `mold`
> prefix-matching "crown molding" was promoting 14 cosmetic items to `required_rehab`,
> which word-start anchoring alone would not have fixed. Current distribution is
> **37 / 57 / 11 / 1** (required / marketability / optional / inspection_risk).
>
> Kept for the method — steps 1–4 are still the right way to run this audit again.

## Status / framing

This is the data-hygiene follow-up to [HANDOFF_renewal_costing_toggle.md](HANDOFF_renewal_costing_toggle.md).
The toggle's backend has shipped: `compute_renovation_estimate_v4` now emits three nested headline
tiers — `final_rehab_required`, `final_rehab_resale_ready`, `final_rehab_full_renewal` — derived
from `totals_by_scope_capped` via `build_scope_headline_tiers`
([tools/estimate_scope.py](../tools/estimate_scope.py)).

**The tiers are only as trustworthy as the per-scope classification that feeds them.** Each line
item and each package is sorted into one of four scopes, and that sort is mostly **heuristic**
(term-matching + `kind`/`tier`/posture). This handoff is the audit pass to confirm the heuristic
buckets real catalog items correctly. It is **data work, largely no code change** — the primary
deliverable is catalog `estimate_scope` overrides, not algorithm edits.

Scope→toggle mapping (unchanged, repeated here for the auditor):

| Scope | Tier bucket |
|---|---|
| `REQUIRED_REHAB` | repair (must-fix) — counts in all three tiers |
| `MARKETABILITY_REHAB` | strong renewal — counts in `resale_ready` and `full_renewal` |
| `OPTIONAL_VALUE_ADD` | nice-to-have renewal — counts in `full_renewal` only |
| `INSPECTION_RISK` | latent risk — never in any headline tier |

A mis-bucket moves real money between tiers: a repair mis-tagged as renewal vanishes from
`final_rehab_required` (and from the headline entirely if the product defaults renewal OFF); a
cosmetic mis-tagged as required inflates the "must-fix" number.

## Why this is the long pole (the heuristic is substring-based)

There are **two classification surfaces**:

1. **Line items** — `classify_estimate_scope` / `classify_estimate_scope_with_reason`
   ([estimate_scope.py:94](../tools/estimate_scope.py#L94)), whose real logic is
   `_classify_baseline_scope_with_reason` ([estimate_scope.py:160](../tools/estimate_scope.py#L160)).
   It matches four term lists against `_catalog_text` ([estimate_scope.py:484](../tools/estimate_scope.py#L484)),
   a lower-cased concatenation of the item's `id, name, category, description, scope, trade_bucket,
   work_item_code, embed_text, tier, support_any` plus a few candidate fields:
   - `_REQUIRED_TERMS` ([:36](../tools/estimate_scope.py#L36)) / `_VISIBLE_REQUIRED_CONDITION_TERMS` ([:44](../tools/estimate_scope.py#L44)) → `REQUIRED_REHAB`
   - `_MARKETABILITY_TERMS` ([:70](../tools/estimate_scope.py#L70)) → `MARKETABILITY_REHAB`
   - `_VALUE_ADD_TERMS` ([:76](../tools/estimate_scope.py#L76)) → `OPTIONAL_VALUE_ADD`
2. **Packages** — `classify_package_scope` ([estimate_scope.py:274](../tools/estimate_scope.py#L274)).
   When a package carries an explicit `package_category` (+`package_strength`) the mapping is
   **deterministic** (`repair→required`, `modernization(strong)→marketability`,
   `modernization(moderate)→optional`, `turnover→marketability`, `inspection_risk→inspection`).
   Packages **without** a category fall back to the supporting drivers' scopes. So package routing is
   mostly trustworthy *if catalog package metadata is set* — audit that the category/strength fields
   exist, then focus effort on the line-item heuristic.

Because line-item matching is **substring, not token**, it over-fires. Concrete false-positive
paths to look for:
- `_MARKETABILITY_TERMS` contains `"finish"`/`"finishes"` → matches `"refinish"`, `"unfinished"`;
  `"style"` → matches `"lifestyle"`; `"old "`/`"older"`; `"fixture"`, `"lighting"`, `"paint"`.
  A **required** plumbing/electrical defect whose text contains `"fixture"` or `"lighting"` but has
  `severity < 3`, is not in a `_STRUCTURAL_CATEGORIES` category, and matches no `_REQUIRED_TERMS`
  lands in `cosmetic_defect_marketability` — a repair silently demoted to renewal.
- `_VALUE_ADD_TERMS` contains `"addition"` → matches `"additional"`, `"in addition"`; `"open wall"`
  → matches descriptions about "open wall**s**" that are actually damage.
- `"unfinished floor"` is a required term but `"finish"` is a marketability term — ordering in
  `_classify_baseline_scope_with_reason` matters (required checks run before marketability for
  defects; confirm the precedence holds for the items you review).

### Converting `_contains_any` to `term_matches` — measured, and safer than it looks

[HANDOFF_catalog_term_hygiene.md](HANDOFF_catalog_term_hygiene.md) §4 left
`estimate_scope._contains_any` ([:573](../tools/estimate_scope.py#L573)) as the one
un-converted raw-substring surface, on the grounds that swapping it would break `dated` (59
corpus hits inside "outdated") and `paint` (8 inside "repainting") unless `outdated` and
`repaint` were added in the same commit. **That condition is already satisfied** — `outdated`,
`repaint` *and* `finishes` are all present in `_MARKETABILITY_TERMS` today. Measured against
the 4,572-observation corpus:

- **0 of 106** items change scope on catalog text alone.
- Across all 106 items × 4,572 observations: **307 of 484,632 pairs (0.063%)** flip, touching
  only **4** items, every one caused by `finish` no longer matching "unfinished":

| item | flips to | n |
|---|---|---|
| `driveway_or_walkway_cracking` | `required_rehab` | 70 |
| `trees_or_vegetation_too_close` | `required_rehab` | 70 |
| `brick_weathering_or_mortar_deterioration` | `required_rehab` | 70 |
| `gutter_maintenance_needed` | `optional_value_add` | 97 |

All four flip **out** of `marketability_rehab`. That is worth reading carefully: it exposes
that `defect_default → required_rehab` is the wrong fallback for cosmetic exterior
maintenance, and the substring bug had been accidentally masking it. Give those four an
explicit `estimate_scope` override in the same change rather than shipping the flip bare.

Reproduce with `scripts/audit_estimate_scope.py --baseline`, or by monkeypatching
`_contains_any` to `lambda text, terms: any(term_matches(t, text) for t in terms)`.

Note also that `_catalog_text` concatenates `support_any`, so a term now carrying the
whole-word marker (`mold$`, `wall$`, `tub$`, `rat$` — see the term-hygiene handoff §2) lands
in the scope blob with its `$`. Under the current substring `_contains_any` that is harmless
(`"mold" in "mold$"` is still true); if you convert, apply `strip_term_marker` in
`_catalog_text` so `wall$` does not stop matching a `wall` scope term.

## Current state (the audit surface)

`tools/issue_catalog.json` is the catalog. **Only ~4 items currently set an explicit `estimate_scope`
override** (grep `estimate_scope` in the file) — everything else is heuristic-classified. That is the
gap: the heuristic is doing nearly all the work unaudited.

The override is the existing, intended escape hatch: `_catalog_scope_override`
([estimate_scope.py:431](../tools/estimate_scope.py#L431)) reads `estimate_scope` (or
`estimate.estimate_scope`) from the catalog item and, when present and valid, **short-circuits the
heuristic** with reason `catalog_override`. Fixing a mis-bucket is therefore a one-line data edit, no
code change.

> Note: `tools/catalog_auditor.py` is a *different* tool — a cloud-model judge for diagnosing
> pipeline/detection failures. It is not scope-aware. This audit is deterministic and local; don't
> conflate the two.

## Method

1. **Enumerate.** Loop every item in `tools/issue_catalog.json` and call
   `classify_estimate_scope_with_reason({}, item, None)` (an empty candidate makes the catalog values
   the source of truth and applies no inspect posture, so you get the pure baseline scope + reason).
   Emit a table: `id, kind, tier, category, trade_bucket, scope, severity, estimate_scope, reason`.
2. **Triage by reason.** Split rows into:
   - **Hard / structural reasons** (trust): `catalog_override`, `required_category`,
     `defect_severity_threshold`, `defect_default`.
   - **Soft / term-driven reasons** (review): `*_signal`, `upgrade_marketability`,
     `cosmetic_defect_marketability`, `marketability_signal`, `optional_*`, `upgrade_default`,
     `fallback_required`.
3. **Adjudicate** each soft row against the intent: defects/must-fix → `required`, cosmetic &
   modernization-with-turnover → `marketability`, layout/value-add/nice-to-have → `optional`,
   inspect-only → `inspection_risk`. For every item the heuristic gets wrong, set an explicit
   `estimate_scope` in `issue_catalog.json` (prefer the override over editing term lists).
4. **Tighten term lists only as a last resort** — a term removal/addition affects every item, so a
   bad edit causes wide regressions. Prefer per-item overrides; reserve term-list edits for clear,
   broadly-correct fixes (e.g. word-boundary issues), and re-run step 1 to measure the blast radius.
5. **Spot-check on real properties.** Run the pipeline on a handful (e.g. the
   `redfin_*` properties used elsewhere) and read `totals_by_scope_capped` + each item's
   `estimate_scope_reason`; confirm the split reads correctly (the renewal-toggle handoff records the
   `redfin_126231380` split as a known-good reference: bathroom finishes + popcorn + lighting →
   marketability/optional, gfci-repair → its own repair candidate).

## Deliverable

- Corrected `estimate_scope` overrides committed to `tools/issue_catalog.json`.
- A short before/after report (the enumeration table from step 1, pre- and post-override) — append to
  or model on `docs/catalog_audit_report.md`.
- Optionally: minimal, well-justified term-list tightening with the measured blast radius.

## Verification

- Re-run the enumeration; every previously-flagged item now resolves to the intended scope (via
  `catalog_override` for the corrected ones).
- `python -m pytest tests/test_renovation_estimate_v4.py -q` still green (the scope-classification
  tests around [test_renovation_estimate_v4.py:864+](../tests/test_renovation_estimate_v4.py#L864)
  assert specific items' scopes — update them if an intended override changes a tested item).
- Spot-checked real properties show sane `required ≤ resale_ready ≤ full_renewal` splits with no
  obviously mis-bucketed line items.

## Gotchas

- `_catalog_text` includes `embed_text` and `support_any`, so a term can match via retrieval text the
  catalog author didn't expect — always read the *actual* concatenated text when a reason looks wrong.
- Overrides accept only the four `VALID_ESTIMATE_SCOPES` values; anything else is ignored and the
  heuristic silently takes over (`_catalog_scope_override` returns `""`).
- Package metadata (`package_category`/`package_strength`) and line-item `estimate_scope` are
  independent surfaces — an item can be a correctly-categorized package driver yet still produce a
  mis-bucketed standalone line item, and vice-versa. Audit both where an item does both.
