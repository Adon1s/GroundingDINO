# RenoIntel benchmark

Compares pipeline output against **human truth**, not against another model run.
The existing harnesses (`model_comparison`, `pass_2f_model_comparison`,
`quant_artifact_comparison`) all answer "did the numbers move?" This one answers
"which layer moved?" — detection, catalog mapping, product retention, package
formation, judgment, or pricing.

Nothing here writes to the production database, image folders, or artifact tree.

## Layout

```
benchmarks/
  datasets/renovation-v1/
    manifest.json                 listings, tiers, slices, fingerprints
    reference_vocabulary.json     written at seal time; truth validates against THIS
    listings/<id>/
      metadata.json               frozen listing facts + photo order/hashes
      reference.draft.yaml        hand-authored (git source of truth)
      reference.json              compiled, canonical (what the evaluator reads)
      photos/                     Git LFS
  configs/                        one file per pipeline configuration
  baselines/<name>/               accepted normalized predictions + metrics
benchmark-results/                run outputs (git-ignored)
```

## Commands

```bash
python -m tools.benchmark import --dataset renovation-v1 --listing hsv-001 --property-key redfin_125970550
```
```bash
python -m tools.benchmark reference template --dataset renovation-v1 --listing hsv-001
```
```bash
python -m tools.benchmark catalog search "water stain"
```
```bash
python -m tools.benchmark reference check --dataset renovation-v1 --listing hsv-001
```
```bash
python -m tools.benchmark reference compile --dataset renovation-v1 --listing hsv-001
```
```bash
python -m tools.benchmark validate --dataset renovation-v1
```
```bash
python -m tools.benchmark seal --dataset renovation-v1
```

`vocabulary diff` reports how live code has drifted from a sealed dataset's
frozen vocabulary.

---

## The four annotation phases

The order matters. Each phase exists to stop a specific way the reference could
become anchored to the system it is supposed to be judging.

### Phase 1 — blind human inventory

Photos and frozen listing metadata only. **No Terra, Qwen, or package output.**

Record: the physical rooms and which photos show the same room; the visible
findings; which photos support each finding; whether the finding is actionable;
whether the photos are sufficient to determine it; and what billable component
or physical scope it represents.

Use `benchmark catalog search` freely — the catalog is hand-authored, not model
output, so consulting it anchors nothing. When nothing fits, set
`catalog_status: missing_catalog_item` and leave `catalog_item_ids` empty. That
records a **catalog gap**. Without that option the gold set would silently
inherit the catalog's current blind spots.

Note what a missing catalog item does *not* mean. Two dimensions are tracked
separately:

- the model described it but could not map it → detection success + catalog failure
- the model never described it → detection miss (plus a catalog gap)

### Phase 2 — blind package judgment

Using your finalized findings, still **without** seeing production packages:
do these findings justify a package? which physical room or billable unit? which
category? which components belong? what tier? which findings stay standalone?

This is the phase people skip, and skipping it is fatal: run the package
calculator first and approve its suggestions, and your "package truth" is just a
restatement of the package architecture you were trying to evaluate.

### Phase 3 — model-assisted reconciliation

Only after the blind draft is saved. Reveal the **union** of findings from every
configuration under test — source-blinded, shuffled, deduplicated, presented as
candidates and never inserted automatically.

For each candidate decide: matches an existing reference finding / a real finding
you missed / valid but optional / unsupported / indeterminate from the photos /
a catalog-mapping error rather than a detection error.

Prefilling from one model would let that model define the universe of possible
findings, so every other model would be scored on whether it agrees with that
one. The union prevents that. An adjudication log records how far this phase
moved the blind draft.

### Phase 4 — cost review

Human scope, per-component and listing-level expected bands, band confidence,
and the basis. Never approve the calculator's own output as truth.

Market inputs live only in `metadata.json` and must not be restated in the
reference. If a frozen input is wrong, fix the metadata and bump the dataset
version.

---

## Why the truth fields are shaped this way

**Orthogonal, not one enum.** `required | acceptable | unsupported |
indeterminate` conflates presence, reporting duty, actionability, and visual
sufficiency. Sol confirming drywall damage while rejecting kitchen modernization
is a *correct* pair of judgments that one enum cannot express. So five
independent fields are stored and the reporting category is derived
(`schemas.reporting_category`).

Contradictory combinations are rejected at compile time rather than resolved by
precedence:

| Rule | Why |
|---|---|
| `absent` → `must_not_report` | nothing to report |
| `indeterminate` → `required` or `acceptable` | you cannot demand silence about something you could not determine |
| `insufficient` → `presence: indeterminate` | visible but unsizable is `present` + `limited` |
| `critical: true` → `present` + `required` + `sufficient` | it is a hard release gate, so it must be unambiguous |

Deliberately **allowed**: `present` + `must_not_report` — a real, visible, but
trivial condition a good analyzer should stay quiet about. That is the mechanism
for scoring noise. Also allowed: `indeterminate` + `required`, for a condition
photos cannot settle that should still be flagged for inspection.

**Finding ≠ evidence occurrence ≠ billable work.** Six photos of dated cabinets
is one finding, six evidence occurrences, one billable component with
`expected_units: 1`. That last number is what detects repeated-evidence cost
inflation instead of blessing it.

**Rooms are separate from per-photo scene truth.** One photo can show several
spaces (open kitchen/living, a hallway view, a vanity vs a closet), so physical
room grouping and scene classification are scored as two different metrics.

**Package truth is separate from finding truth**, with a required `reason` on
every rejection — that reason is exactly the judgment the 2f judge is scored
against.

## Tiers and slices

`gold` — blank-first, reconciled, exhaustive within a declared `coverage_domain`,
packages and cost bands authored independently. Gates releases. 8–12 listings.

`silver` — prefilled from the blinded union, human-adjudicated. Trend analysis
only, never gates. 30–50 listings.

Because gold failures drive prompt and catalog work, the gold set gradually
becomes a development set. Slices: `gold-development`, `gold-holdout`,
`silver-development`, `stress-duplicates`, `stress-high-photo-count`,
`stress-updated-home`, `stress-heavy-rehab`. **Keep 3–4 gold listings in
`gold-holdout`** — it is the only way to tell general improvement from tuning to
familiar houses.

`coverage_domain` is mandatory for `exhaustive`, because photographs cannot
establish hidden electrical, plumbing, foundation, or HVAC condition. Those are
`presence: indeterminate`, never forced.

## Sealing

Sealing writes `reference_vocabulary.json` and fingerprints the dataset. After
that, references validate against their **frozen** vocabulary forever — never
against live code. Otherwise renaming a catalog id or consolidating a scene would
invalidate the gold set before you could measure the effect of that change, which
is precisely backwards.

Every mutating command (`import`, `reference template`, `reference compile`,
`seal`) refuses a sealed dataset. Create a new dataset version instead.

Catalog, prompt, aggregation, and cost fingerprints are **treatment and
provenance dimensions**, not compatibility blockers: a catalog change is
something you want to benchmark, not something that should invalidate a baseline.
