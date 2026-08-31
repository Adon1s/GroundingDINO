# Handoff — Factorized verifier: canary replay (shadow, measurement only)

Written 2026-08-31 · backend branch `terra_factorized_verifier` · related:
`docs/ROADMAP_quality_program_sessions_20260826.md` (Session F / QP2 heavy
variant), `docs/PROPOSALS_output_quality_improvements_20260826.md` (QP1, QP2),
`docs/RESULT_label_v1_1_20260829.md` (the label set this scores against),
`docs/HANDOFF_quality_program_direction_20260829.md` (the open direction call).

**What this session builds:** an experimental verifier that asks three separate
questions per condition instead of one, replayed over the frozen Session-9
canary with the same `gpt-5.6-terra` model, and scored against the v1.1 human
labels. **It changes no production behaviour.** Terra stays unchanged and
authoritative; nothing here touches `REVIEW_VERDICTS`, `decide_disposition`,
work items, packages, or estimates. The output is a measurement report and a
go/no-go on whether the factorization is worth pursuing.

Inherit the roadmap's **Simplicity guardrails** and **Hygiene rules** blocks in
full (`ROADMAP…:57-89`). Two of them bind hard here: the frozen canary root is
untouchable (read-only input, never an output root), and `RV_ROOT` never gets a
`git checkout` of another branch — §4 uses a worktree.

---

## STATUS — updated 2026-08-31, after implementation

Everything that does not spend tokens is **built and verified**. Read the
sections below as the design record; the table is what is actually left.

| section | state |
|---|---|
| §2 module `factorized_review.py` | **DONE** — 56 tests green |
| §3 harness `--factorized` arm | **DONE** — 10 tests green (`test_redecide_harness.py`) |
| §6 scorer `score_factorized_review.py` | **DONE** — 11 tests green, incl. the oracle and degeneracy arms |
| §8 tests | **DONE** — full suite **2,674 passed, 0 failed** |
| §4 pinned worktree | **DONE** — `../rv-factorized-replay` on branch `factorized_replay_v1` at `c833da4` |
| §5 step 1 — dry run | **DONE, proof obligation met: 148/148 verified, 0 refusals, 1,047 conditions, 148 distinct fingerprints.** Output: `artifacts_canary/factorized_v1_dryrun_20260831/` |
| §7 gate ratification | **OWED — Steven** |
| §5 steps 2–3 — live smoke + full run | **BLOCKED on ratification** (~1.0–1.4M tokens) |
| §6 run against live output | blocked on the above |
| post-L2 rescore | blocked, and mandatory when L2 lands |

Confirmed empirically rather than argued: the drift in §4 is real (the main
tree refuses every listing with `refused_catalog_drift`), and the §7 degeneracy
concern is real (`tests/test_score_factorized_review.py` asserts that an
always-`yes` arm passes G1 **and** G2 while G4 catches it).

---

## 0. Scope and placement

### 0.1 The problem this measures

Terra reduces every proposed condition to `supported` / `unsupported` /
`cannot_assess` (`contracts.py:115`). That single verdict cannot separate four
different situations, and the v1.1 review says the ones it cannot see are the
big ones. Measured billed-error mass on the canary: **misnamed 49.4%, trivial
26.5%, absent 24.1%** (`RESULT_label_v1_1_20260829.md`). Terra's verdict
vocabulary can only express the smallest of the three. A real-but-misnamed
condition and a real-but-trivial one both look like `supported`, and both get
billed.

The hypothesis: asking *visible*, *accurate-as-written*, and *material* as
three bounded questions surfaces the two invisible categories without
suppressing legitimate work. This session tests that hypothesis and nothing
else.

### 0.2 What this is NOT

- **Not a Terra change.** `TERRA_SYSTEM_PROMPT`, `TERRA_REVIEW_PROMPT_VERSION`,
  and `parse_unit_reviews` are untouched. No production fingerprint moves.
- **Not the rejected 4th-verdict lane.** QP2 considered adding
  `present_but_misdescribed` to `REVIEW_VERDICTS` and rejected it: it breaks a
  closed frozenset, `decide_disposition`, `DISPOSITION_REASON_CODES`, and
  forces a `CONDITION_DISPOSITION_POLICY_VERSION` bump that invalidates every
  stored artifact. **None of that happens here** — the factorized contract
  lives in its own module with its own prompt version and never enters the
  pipeline.
- **Not the catalog-remap lane.** When the verifier says a claim is misnamed it
  records a short free-text description of what it actually sees. It never
  proposes a catalog item. Remapping stays the deliberately-deferred separate
  change QP2 names (`PROPOSALS…:205`).
- **Not a severity/threshold policy.** "Below the standalone-work threshold" is
  a measurement, not a billing decision. Whether trivial conditions get
  absorbed by turnover packages or dropped is downstream policy nobody decides
  here.
- **Not a production-wiring step.** There is no flag, no shadow key, no
  envelope field. If the gates pass, wiring is a separate session with its own
  decision row.

### 0.3 Why this is Session F's structured-fields arm, run early

Session F charters a "heavy variant" that "asks per-condition sub-answers —
subject match / claimed-state match / location match / evidence sufficiency —
and derives the final verdict deterministically from them" (`PROPOSALS…:166`).
This is that arm, with two deliberate differences:

1. **A materiality axis F does not have.** F's four sub-answers all decode to a
   verdict, so they can only move `supported`↔`unsupported`. The scorer says so
   itself: `trivial_billed` "is reported but never decides an arm", because "no
   wording change makes a model reject a true claim about a mildly dated
   finish" (`score_redecide_variants.py:220-222`). 27% of the error mass is
   structurally invisible to every arm F is chartered to run. Adding the third
   question is the only way to measure it.
2. **No verdict derivation.** F's arm collapses sub-answers back into a verdict
   to feed the existing scorecard. This one keeps the factors as the output.

### 0.4 The gate-2b exception (recorded)

Roadmap gate 2b: "**no Terra tokens are spent until L2 lands**"
(`ROADMAP…:122-128`). Its stated reason is specific: after the label repair,
`hard_false_billed` = 3 and "the scorecard can no longer separate an arm from
the noise floor **on its primary leg**".

That reasoning is about **verdict-flip arms**, whose win condition is hard-false
flips. This experiment's decisive legs are different populations entirely:
false-suppression on **57** `supported_billed` cards and recovery on **16**
`dirB_recovery` cards. Neither depends on `hard_false_billed` (n=3), which here
is only a directional check.

**Decision (Steven, 2026-08-31):** run live now under this exception, and
rescore for free after L2. Recorded terms:

- The gate's rationale does not bind this experiment's decisive gates (G1, G2).
- The categories L2 fattens — misnamed (n=7), trivial (n=5), absent (n=3) — are
  gated as **exploratory** here (G3) and are not allowed to decide anything.
- **The rescore after L2 is mandatory, not optional.** The replay covers all
  1,047 canary conditions, so every label L2 adds joins the same stored outputs
  at zero token cost. Re-running §6 is free; skipping it is not permitted.
- This exception covers this experiment only. It does not unblock Session F.

---

## 1. The factor contract and the derivation

### 1.1 The three questions

Each is answered `yes` | `no` | `unclear`, per condition.

**1. `visible` — is the claimed thing there, on the object and in the room the
claim names?** Wrong-object and wrong-room cases are `no`, not `yes`. This is
the closest analogue to what Terra answers today.

**2. `claim_accurate_as_written` — read literally, does every part of the claim
hold?** This is a *wording* judgment, not a severity or reality judgment. A
claim can be inaccurate and still sit on a real, significant problem — that is
precisely the `misnamed` class. Two rules make this measurable:

- **Compound claims:** a claim is accurate only if **every** assertion in it
  holds. The known failure card is `rc_025ca614449b`, where the human wrote
  that the casing being chipped and dirty was true but the ceiling
  discolouration was "a hallucination based on the reflected colors on the
  ceiling from the window". Half-true compound claims are `no`, not `yes`.
- **Presupposition:** accuracy is only read when `visible = yes`. Nothing is
  misnamed if nothing is there.

**3. `material_enough_for_work` — judged on what is actually VISIBLE, not on
what the claim says.** Otherwise "misnamed and trivial" is incoherent (trivial
relative to which condition?). Anchored concretely: would a contractor write
this up as its own line item, or is it the minor wear a routine turnover clean
and touch-up absorbs?

**"Below the standalone threshold" ≠ "never billed."** Turnover and refresh
packages exist precisely to absorb cosmetic-minor items. Whether a trivial
condition disappears or rolls into turnover is downstream policy. The verifier
answers only whether the condition stands on its own.

### 1.2 The derivation table (pure code, not model output)

The model is never asked for a class. It answers three factors; code derives
the class. This is deliberate: the per-axis vocabulary stays small and closed,
agreement statistics come out **per axis** (so a failure tells you *which*
question is unreliable — the entire point of factorizing), and rare cells like
"misnamed and trivial" fall out of the cross-product instead of needing the
model to reason about a taxonomy.

| `visible` | `claim_accurate_as_written` | `material_enough_for_work` | derived class |
|---|---|---|---|
| `no` | *(any)* | *(any)* | `absent` |
| `unclear` | *(any)* | *(any)* | `inconclusive` |
| `yes` | `unclear` | *(any)* | `inconclusive` |
| `yes` | *(any)* | `unclear` | `inconclusive` |
| `yes` | `yes` | `yes` | `exact_and_warranted` |
| `yes` | `yes` | `no` | `exact_but_trivial` |
| `yes` | `no` | `yes` | `misnamed_but_warranted` |
| `yes` | `no` | `no` | `misnamed_and_trivial` |

All 27 combinations are covered: 9 → `absent`, 14 → `inconclusive`, 4 → the
named classes. Any `unclear` anywhere in a visible condition yields
`inconclusive` — the simplest defensible rule, and it makes the **inconclusive
rate a health metric in its own right** (if the factorized prompt punts far more
than Terra's 2.8% `cannot_assess`, that is a finding).

### 1.3 The class names are the label slugs, deliberately

`tools/label_schema.py::ADJUDICATION_KEYS` already carries exactly this
vocabulary, because the v1.1 label repair independently split the same two
axes ("v1 recorded one label for three different judgments — is the condition
there, is the exact claim accurate, is the work worth doing",
`label_schema.py:9-13`). Emit the slug strings verbatim so the scorer joins
without a translation table:

`exact_and_warranted` · `misnamed_but_warranted` · `exact_but_trivial` ·
`misnamed_and_trivial` · `absent` · `inconclusive`

**One fold:** the label vocabulary has seven slugs — it separates
`wrong_object_or_place` from `absent`, and `CLAIM_AXIS` maps both to `absent`
(`label_schema.py:45-53`). The derivation emits one `absent` class covering
both, matching the design's "absent or on the wrong object" as a single
outcome. The scorer folds label `{absent, wrong_object_or_place}` → derived
`absent`. (Both label sets have 0 instances of `wrong_object_or_place`, so this
fold currently costs nothing.)

Because §4's worktree pin predates `tools/label_schema.py`, the module carries
the six strings as a documented **data mirror** (the house pattern —
`review_analysis.py` does the same with a `# data mirror of …` comment). Add
the pin-test asserting `set(DERIVED_CLASSES) - {"inconclusive"} ⊂
set(ADJUDICATION_KEYS.values())` when merging back to a tree that has
`label_schema.py` (§9).

---

## 2. Module: `tools/renovation_architecture/factorized_review.py`

New file, ~220 lines. Mirrors `terra_review.py` + `review_pipeline.py:75-173`
in structure so the two read side by side.

**Boundary invariant.** This module is imported only by
`scripts/redecide_renovation_architecture.py` under `--factorized` and by its
tests. It is **not** re-exported from `tools/renovation_architecture/__init__.py`
and no pipeline module imports it (§8 enforces this with a test). It exists in
the package for readability against `terra_review.py`, not because anything
wires it in.

### 2.1 Constants

```python
FACTORIZED_PROMPT_VERSION = "terra_factorized_review_v1"
FACTOR_VALUES = frozenset({"yes", "no", "unclear"})
FACTOR_KEYS = ("visible", "claim_accurate_as_written", "material_enough_for_work")
OBSERVED_DESCRIPTION_MAX_CHARS = 200
DERIVED_CLASSES = (...)   # the six strings from §1.3, data mirror
```

Reuse `REVIEW_RATIONALE_MAX_CHARS` (400) from `contracts.py` — do not invent a
second rationale cap.

### 2.2 The system prompt (draft — review before implementing)

Deliberately mirrors Terra's shape: strict, bounded, closed schema, no
package/price/quantity/confidence content. **One scoped deviation:** Terra's
prompt forbids mentioning renovation work at all, and `parse_unit_reviews`
enforces that at the parse layer ("Terra may not emit work, package, price,
quantity, or confidence content", `terra_review.py:266-269`). Question 3 *is* a
work-materiality judgment, so this contract permits exactly that one thing and
keeps every other prohibition. **This is the structural reason the factorized
contract cannot be a Terra prompt-version bump and must have its own parser.**

```
You are a strict photographic evidence reviewer for property renovation
conditions. For each supplied condition, answer three separate questions
about the attached photos. Do not collapse them into one judgment.

Rules:
- Judge only the conditions listed in the request. Never add, merge, or
  invent conditions.
- Answer each question with exactly one of: yes, no, unclear.

1. visible — is the specific thing the claim describes present in the
   photos, on the object and in the room the claim names?
   * yes: it is there, on the named object and in the named room.
   * no: the photos show the relevant area and it is not there, or what is
     there is on a different object or in a different room than the claim
     names.
   * unclear: the photos do not show the area well enough to judge.

2. claim_accurate_as_written — read the claim text literally. Does every
   part of it hold for what you can see?
   * yes: every assertion in the claim is true of what is visible.
   * no: something real is there, but the claim names it wrongly — wrong
     mechanism (staining described as scuffing), wrong material, wrong
     object, or a claim joining two assertions where only one holds.
   * unclear: you can see the area but cannot tell whether the wording fits.
   Judge wording only. A claim can be worded wrongly and still sit on a
   real, significant problem.

3. material_enough_for_work — judge what you can actually SEE, not what the
   claim says. Would a contractor write this up as its own line item?
   * yes: it needs its own repair, replacement, or refinishing line.
   * no: it is the minor wear a routine turnover clean and touch-up paint
     absorbs — faint marks, light scuffing, ordinary aging without damage.
   * unclear: it is visible, but these photos cannot show its extent.

- When visible is `no`, answer the other two questions `unclear`.
- observed_description: when claim_accurate_as_written is `no`, one short
  phrase naming what you actually see instead. Otherwise the empty string.
  Describe only — never name a replacement claim, a repair, or a product.
- rationale: one short sentence describing what you can or cannot see.
- Never mention packages, prices, quantities, or confidence percentages.
Return JSON matching the provided schema with exactly one review per
supplied condition_id.
```

### 2.3 `build_factorized_response_schema(condition_ids) -> dict`

Same closed shape as `build_response_schema` (`terra_review.py:93-118`):
`additionalProperties: False` at both levels, `required` listing all six
per-review fields, `condition_id` enum = `sorted(condition_ids)`, each factor
enum = `sorted(FACTOR_VALUES)`, `observed_description` and `rationale` plain
strings.

### 2.4 `parse_factorized_reviews(raw_text, *, condition_ids, model="")`

Every rule from `parse_unit_reviews` (`terra_review.py:233-291`), same typed
failures (`PassExecutionError("factorized_review", "parse", …,
code="FactorizedReviewContract")`), same guarantee that operational failures
never become a factor value:

valid JSON → top level exactly `{"reviews"}` → list of dicts → no field outside
the closed six → unknown/duplicate `condition_id` rejected → each factor in
`FACTOR_VALUES` → every supplied id present.

**One deliberate asymmetry: strict on bounded vocabulary, lenient on free
text.** `rationale` and `observed_description` must be strings and are
truncated (400 / 200); an *empty* `observed_description` when
`claim_accurate_as_written == "no"` is **not** a parse failure. Reason: parsing
happens after settle (the tokens are already spent), the factors are what the
experiment measures, and failing a whole unit over a missing free-text field
would burn ~6k tokens for no measurement. The scorer counts these as a
prompt-adherence metric instead (§6). Same treatment for the "`visible = no` ⇒
other two `unclear`" instruction: the parser accepts any bounded value, the
derivation ignores them, and the report counts violations as adherence.

### 2.5 `derive_class(factors) -> str`

Pure function, no I/O, the §1.2 table. Follow `disposition.py`'s conventions
exactly: a module-level `_UPPER_SNAKE` dict for the 2×2, a public `derive_*`
returning a bounded string, `ValueError` on any token outside `FACTOR_VALUES`.

### 2.6 `factorized_unit_fresh(...)`

Mirrors `_review_unit_fresh` (`review_pipeline.py:75-173`) with the same
signature shape and the same ordering, which is load-bearing:

`estimate_reservation_tokens` → `ledger.reserve` (`TerraDailyBudgetExceeded` →
typed failure) → usage snapshot → `with external_reservation(): call_terra_review(...)`
→ snapshot → delta → **`ledger.settle` BEFORE parse** → `parse_factorized_reviews`
→ records.

Carry a comment naming `review_pipeline.py:75` as the mirrored source and
stating why settle precedes parse ("the tokens are spent whether or not the
response honors the contract"). On `PassExecutionError`, settle with
`provider_total_tokens=None` and re-raise, exactly as production does.

`call_terra_review` is reused unchanged — it already sends
`request.response_schema` (`terra_review.py:216`). Its hardcoded
`response_schema_name="terra_condition_review"` is a provider-side label only;
leave it alone rather than adding a parameter for cosmetics.

Returned `call` dict: same keys as production's but
`prompt_version=FACTORIZED_PROMPT_VERSION`. Returned reviews: per condition the
three factors, `observed_description`, `rationale`, and `derived_class`.

---

## 3. Harness extension: the `--factorized` arm

Extend `scripts/redecide_renovation_architecture.py` (~60 lines). It already
does the hard parts: envelope rehydration, byte-identical request rebuild,
per-unit fingerprint verification, resume, out-root guards, and a `--live` lane.

1. **Built-in arm.** Add `FACTORIZED_VARIANT = {"label": "factorized_v1",
   "target": "terra", "system_prompt": FACTORIZED_SYSTEM_PROMPT,
   "response_contract": "factorized_v1"}` and `--factorized` as a third member
   of the existing mutually-exclusive required group. Reserve the label the way
   `control` is reserved (`load_variant:99-104`). Add `response_contract` to
   `VARIANT_KEYS` so a hand-written spec cannot smuggle one in unvalidated —
   accept only the literal `"factorized_v1"`.

2. **Payload identity.** The arm sets `system_prompt` only. The payload half of
   `user_prompt` stays byte-identical to control — same claims, same
   observations, same photo keys — so the factorized run and the stored Terra
   run see exactly the same evidence. `apply_variant` (`:141-173`) already does
   this; no change needed.

3. **Schema swap.** After `variant_request = apply_variant(request, variant)`,
   when the variant declares the factorized contract:
   ```python
   variant_request = replace(
       variant_request,
       response_schema=build_factorized_response_schema(list(request.condition_ids)),
   )
   ```
   **Fingerprint note:** `response_schema` is not an input to
   `variant_fingerprint` (`:126-138`), but `label` is — and `factorized_v1` is
   unique — so no collision is possible with control, with production, or with
   any F arm. Leave `variant_fingerprint` unchanged; changing it would strand
   the existing `artifacts_canary/redecide_*_20260827` resume state for no
   benefit. If a future arm ever reuses a system prompt with a different
   schema, that is when the schema joins the fingerprint.

4. **Live branch.** In `_redecide_unit` (`:437-449`), dispatch on the contract:
   the factorized arm calls `factorized_unit_fresh`, everything else keeps
   `_review_unit_fresh`. Keep the function-local import discipline — the live
   path must stay unimportable in dry-run mode.

5. **Record shape.** Per condition, store the three factors,
   `observed_description`, `rationale`, and `derived_class`. Keep the existing
   `reviews` key so `load_arm`-style readers keep working.

6. **Manifest.** Add `response_contract` and `prompt_version` so a scorer can
   never misread a factorized root as a verdict root.

---

## 4. The pinned worktree (mandatory) and the drift it works around

**The working tree cannot run this replay today.** Verified 2026-08-31:

| | projection fingerprint |
|---|---|
| working tree (post-Catalog-3.2) | `6baacaf6254861409e1c6c4495e73d57c573523f581a138cc703a302f87b9308` |
| stored canary provenance | `ff62acaeb9a71cb9652687bd62ec2c1117c8f46ceb71f41a355c46b0b86bfacf` |

Catalog 3.2 (`a9ed7ad`) changed the projection, so `redecide_property` refuses
every property with `refused_catalog_drift` (`:250-259`) — and correctly so:
the claim text Terra saw would not be the claim text we send.

**Pin: `c833da4`** ("Canary and evidence review", 2026-08-28) = `a9ed7ad^`, the
last commit before Catalog 3.2. Evidence it reproduces the canary state:

- The catalog file is byte-identical from freeze commit `07ee112` through
  `c833da4` (raw sha `20adb6e342f2d3a5…`; note the stored
  `catalog_sha256 d1cef743…` is the *canonicalized* projection hash, not the
  raw file hash — do not compare them directly).
- `catalog_projection.py` is untouched between `07ee112` and `c833da4`.
- `c833da4`'s only `contracts.py` change is package-application/ledger reason
  codes — not fingerprint inputs (`terra_review.py:166-179`).
- Session B dry-ran all 148 units at `618473c` in this same state with **0
  refusals** (`artifacts_canary/redecide_control_20260827/report.md`: 141
  verified + 7 resumed from an earlier partial run). A fresh out-root reports
  148 verified / 0 resumed / 0 refused.

**Proof obligation:** the §5 dry-run must report **148 units matched and 0
refusals** before any live call. If it does not, stop and diagnose —
`FINGERPRINT_MISMATCH_DETAIL` (`:225-231`) lists the causes. Fallback pin:
`09e28d1`. **Never** loosen or bypass the fingerprint refusal to make a run go
through.

```bash
git worktree add ../rv-factorized-replay -b factorized_replay_v1 c833da4
```

Then, because both are gitignored and therefore absent from a fresh worktree:

- copy `.env` from the main repo (model config: `RENOVATION_TERRA_MODEL=gpt-5.6-terra`);
- pass **absolute** `--root` / `--out-root` paths into the main repo — the
  canary tree lives only there.

Run with the **main repo's** interpreter
(`C:\Users\Steven\PycharmProjects\realtorvision-backend\.venv\Scripts\python.exe`);
it supplies the interpreter and site-packages while the script's own
`REPO_ROOT` / `sys.path` insert (`:46-47`) resolves imports from the worktree.

`tools/label_schema.py` does **not** exist at `c833da4`. That is fine: the
scorer (§6) runs from the main tree and has no projection dependency.

---

## 5. Run plan

All commands from the worktree root. `<RV>` =
`C:\Users\Steven\PycharmProjects\realtorvision-backend`.

**Step 1 — dry run, zero tokens, all 18 listings.** Gate on 148/148.

```bash
"<RV>/.venv/Scripts/python.exe" scripts/redecide_renovation_architecture.py --factorized --root "<RV>/artifacts_canary/renovation_session9_20260818/run_1" --out-root "<RV>/artifacts_canary/factorized_v1_dryrun_20260831"
```

**Step 2 — one-unit live smoke.** Confirms schema acceptance, parser, ledger
debit, and record shape for ~6k tokens before committing to the full run.

```bash
"<RV>/.venv/Scripts/python.exe" scripts/redecide_renovation_architecture.py --factorized --live --limit-units 1 --properties redfin_10806500 --root "<RV>/artifacts_canary/renovation_session9_20260818/run_1" --out-root "<RV>/artifacts_canary/factorized_v1_smoke_20260831"
```

Inspect the written unit file: three bounded factors on every condition, a
`derived_class`, a non-empty `observed_description` wherever accuracy is `no`,
and `terra_call.budget_debited_tokens` > 0.

**Step 3 — full live run, all 18 listings.** Fresh out-root; resume is
automatic and safe if interrupted (per-unit, fingerprint-keyed, `:400-413`).

```bash
"<RV>/.venv/Scripts/python.exe" scripts/redecide_renovation_architecture.py --factorized --live --root "<RV>/artifacts_canary/renovation_session9_20260818/run_1" --out-root "<RV>/artifacts_canary/factorized_v1_20260831"
```

Run all 18 listings, including `redfin_10803207` (zero labeled cards): it costs
~35k tokens and contributes to the label-free full-sweep checks in §6/§7.

**Budget.** Stored Terra spend for this exact population was **918,580 tokens**
(842,895 in / 75,685 out) across 148 units and 538 images. The factorized arm
adds a longer system prompt ×148 and roughly 2–3× output per condition:
**forecast ~1.0–1.4M tokens**, inside one 2.5M free day. Keep
`RENOVATION_TERRA_USAGE_ROOT` at its production value so the daily ceiling
stays global, and run on a day with no production Terra load. `--limit-units`
plus a property subset is the cheap way to stage it if the ceiling is tight.

**Do not panic at the dry-run's token column.** It reports the summed
*conservative pre-call reservation ceiling* (`estimate_reservation_tokens` =
`max(25k, output_cap + request_bytes + 5k × images)`), which totals ~4.9M for a
full pass — Session B's control report shows `4,934,322`. That is not a
forecast and it does not mean the run cannot fit in a day: `_review_unit_fresh`
settles each reservation down to the provider's actual usage immediately after
the call (`review_pipeline.py:126-129`), so peak outstanding is one unit's
reservation (~33k) on top of settled actuals, and cumulative ledger spend
tracks the ~1.0–1.4M forecast. The report's own header says as much ("not
projected actual spend; canary actuals ran ~51k/listing").

**No live control arm is needed.** `run_2` already exists — replica 2 of the
same canary — so run-to-run variance is measured for free (§6.7) against the
known 6.4% (16/250) replica noise floor.

---

## 6. Scorer: `scripts/score_factorized_review.py`

Runs from the **main tree** (needs `tools/label_schema.py`). Emits the house
md + json pair, following `scripts/review_analysis.py` conventions: an inputs
table with sha256 + byte counts for every input, `indent=1, ensure_ascii=False,
sort_keys=True` json, `exploratory_small_n` flags below 10 judgments or 3
listings, per-section `*Evidence:*` paragraphs, exit code 0/1 on integrity, and
a closing "what this does not show".

Population loader mirrors `load_population_v1_1`
(`score_redecide_variants.py:106-137`): label-version check against
`ls.LABEL_VERSION`, `--allow-partial`, join on `(property_key, condition_id)`.
**Print the canary filter and the count it excludes** — a noted scorer defect
is that `load_population_v1_1` "silently excludes production"
(`ROADMAP…:127-128`); this one excludes it loudly, by design — per Appendix A,
production contributes only +1 misnamed and +1 trivial card, not enough to pay
for a second input path.

Sections:

1. **Integrity & coverage** — input shas; units found vs 148; conditions
   covered vs 1,047; parse failures; refusals; labeled cards joined vs 88.
2. **Full-sweep class distribution (label-free, all 1,047)** side by side with
   the stored Terra verdicts (`supported` 882 / 84.2%, `unsupported` 136 /
   13.0%, `cannot_assess` 29 / 2.8%). This is where degeneracy shows up
   (see G4) with no labels involved.
3. **Per-axis agreement** on the 88 labeled canary cards, each as its own
   confusion table:
   - `visible` vs label claim axis (`exact`/`misnamed` ⇒ expect `yes`;
     `absent` ⇒ expect `no`; `inconclusive` ⇒ no expectation)
   - `claim_accurate_as_written` vs label claim axis (`exact` ⇒ `yes`,
     `misnamed` ⇒ `no`; read only where `visible = yes`)
   - `material_enough_for_work` vs label work axis (`warranted` ⇒ `yes`,
     `trivial` ⇒ `no`)
4. **Class confusion** — derived class × label slug, with the §1.3 fold.
5. **Gates** — G1–G4 from §7, each with numerator/denominator and pass/fail.
6. **Disagreement audits** (the eyeball lists, condition_id + property + claim
   text): (a) Terra `supported` & factorized `absent` — one of the two is
   hallucinating; (b) Terra `unsupported` & factorized `visible = yes` — the
   recovery class; (c) factorized `misnamed` with its `observed_description` —
   the future remap lane's input, and the fastest way to judge whether the
   accuracy axis is saying anything real.
7. **Replica-noise benchmark** — factorized-vs-`run_1` disagreement next to
   `run_1`-vs-`run_2` disagreement (6.4%, 16/250). Never report a delta against
   zero.
8. **Adherence** — `unclear` rate per factor; empty `observed_description`
   where accuracy is `no`; non-`unclear` factors where `visible = no`.
9. **What this does not show** — at minimum: the labeled slice was sampled
   disagreements-first, so catch rates are on a hard, non-representative slice
   and the class distribution in §2 is not a production rate; there is **no
   dirB loss population** (`dirB_terra_correct` = 0 cards), so G2 measures
   recovery with no paired specificity check until L2 lands; misnamed/trivial/
   absent gates are exploratory at n = 7/5/3; nothing here measures Sol,
   packages, pricing, or downstream policy.

---

## 7. Pre-committed gates

Steven ratifies these **before** the live run (fill the decision row in §9).
Committing thresholds after seeing numbers is how a measurement stops meaning
anything.

| id | population | n | threshold | status |
|---|---|---|---|---|
| **G1** false suppression | canary `supported_billed` | 57 | derived ∈ {`absent`, `exact_but_trivial`, `misnamed_and_trivial`} on **≤ 5** | decisive |
| **G2** recovery | canary `dirB_recovery` | 16 | `visible = yes` on **≥ 11** | decisive |
| **G3a** misnamed catch | canary `misnamed_billed` | 7 | `claim_accurate_as_written = no` on **≥ 4** | exploratory |
| **G3b** trivial catch | canary `trivial_billed` | 5 | `material_enough_for_work = no` on **≥ 3** | exploratory |
| **G3c** absent catch | canary `hard_false_billed` | 3 | `visible = no` on **≥ 2** | directional |
| **G4** health & degeneracy | all 1,047 | — | four label-free checks below | decisive |

**G1 basis.** These 57 are conditions Terra supported, humans confirmed as
`exact_and_warranted`, and the pipeline billed. A perfect verifier suppresses
0. The allowance of 5 (8.8%) sits just above the 6.4% replica noise floor,
because a *different prompt* inherits at least that much run-to-run variance.
Note `misnamed_but_warranted` is **not** counted as suppression: under the
proposed downstream policy misnamed conditions route to remapping, keeping the
work. It is still an error against the label and is reported separately.

**G2 basis.** These 16 are conditions Terra rejected but humans confirmed as
present and exact — the direct test of the hypothesis that Terra's single
verdict conflates "not visible" with "the wording doesn't fit". If fewer than
11 come back visible, the factorization is not doing the work it claims.

**G4 — the degeneracy guard, and why it is decisive.** G1 and G2 are both
passed by a degenerate verifier that answers `yes` to everything, and the
category that would catch it (G3) is underpowered until L2. These four
label-free checks close that hole:

- `exact_and_warranted` ≤ **90%** of all 1,047 (Terra's own `supported` rate is
  84.2%; a verifier claiming near-total exactness is not discriminating);
- combined misnamed classes ≥ **3%** of all 1,047 (the labels imply the class
  is real and material — ~0% means the accuracy axis is inert);
- `unclear` ≤ **10%** on each factor individually;
- **0** parse failures and **0** fingerprint refusals.

**Reading the result.** Pass = G1 + G2 + G4 all pass, with G3 directional. Any
decisive failure means the design gets revised before any policy conversation —
not a threshold adjustment. G3 is re-evaluated **decisively** in the mandatory
post-L2 rescore (§0.4), which needs no new tokens.

---

## 8. Tests

Run with `.venv\Scripts\python.exe -m pytest`. Suite green before commit.

**`tests/test_factorized_review.py`** (new; follow
`tests/test_renovation_architecture_terra.py` conventions — module docstring
with its own `Run:` line, `_clean_runtime` autouse fixture, conftest
`FakeOpenAI.attach` + `tiny_png`, and the assertion that no live provider is
ever called):

- Parser matrix mirroring the terra parser tests: bad JSON, wrong top-level
  key, extra field, unknown id, duplicate id, missing id, out-of-vocabulary
  factor value, non-string rationale — each a typed failure.
- Truncation of `rationale` (400) and `observed_description` (200).
- The lenient cases explicitly asserted as **not** failures: empty
  `observed_description` when accuracy is `no`; non-`unclear` factors when
  `visible = no`.
- **Full 27-combination derivation matrix** in the `FULL_MATRIX` +
  `@pytest.mark.parametrize` style of
  `tests/test_renovation_architecture_disposition.py`, plus a test asserting
  every output is in `DERIVED_CLASSES`.
- `factorized_unit_fresh` ordering: settle-before-parse (a contract-violating
  response still debits), and a provider error settles with
  `provider_total_tokens=None` and re-raises.
- Schema shape: closed at both levels, per-call `condition_id` enum, factor
  enums.
- **Boundary test:** no module under `tools/renovation_architecture/` other
  than the module itself imports `factorized_review` (grep-style assertion over
  package sources) — the §2 invariant, enforced.

**`tests/test_redecide_harness.py`** (extend the existing 7):

- `--factorized` dry run writes unit files carrying the factorized system
  prompt and the factorized schema.
- **Payload byte-identity**: the factorized arm's `user_prompt` payload half
  equals control's (only the system prompt differs).
- `factorized_v1` is reserved like `control`; a spec file declaring
  `response_contract` with any other value is rejected.
- Variant fingerprints differ between control and factorized arms.

---

## 9. Deliverables, decision row, follow-ups, do-nots

**Deliverables**

1. Branch `factorized_replay_v1` with the §2 module, the §3 harness diff, the
   §6 scorer, and the §8 tests (suite green).
2. `artifacts_canary/factorized_v1_20260831/` — the harness output root.
3. `reports/factorized_review_scorecard.md` + `.json`.
4. `docs/RESULT_factorized_replay_<YYYYMMDD>.md` — the gate table with the
   measured numbers, the three disagreement lists, and a plain go/no-go.
5. Merge back to `terra_factorized_verifier`, and **add the §1.3 pin-test**
   against `tools/label_schema.py::ADJUDICATION_KEYS` on merge (that file does
   not exist at the `c833da4` pin).

**Decision row — fill before the live run**

| decision | owner | recorded |
|---|---|---|
| Gate-2b exception for this experiment | Steven | **2026-08-31, granted** (§0.4) |
| G1–G4 thresholds as written in §7 | Steven | ☐ |
| Live run date (light Terra-load day) | Steven | ☐ |

**Follow-ups**

- **Mandatory:** re-run §6 after Session L2 and re-evaluate G3 decisively. Zero
  tokens — the replay already covers all 1,047 conditions.
- If the gates pass, the *next* session designs production wiring (shadow key,
  flag, policy) with its own decision record. It is not authorized here.
- If G2 passes but the post-L2 dirB loss population shows the verifier revives
  conditions Terra was right to reject, that is a specificity finding that
  outranks G2.

**Do NOT**

- Wire anything into the pipeline: no flag, no envelope field, no shadow key,
  no `decide_disposition` change.
- Add a fourth verdict or touch `REVIEW_VERDICTS` / `DISPOSITION_REASON_CODES`
  / any policy version. QP2 rejected that lane on stated evidence.
- Let the verifier name a replacement catalog item. Free-text description only.
- Decide severity or billing policy from these numbers.
- Loosen the fingerprint refusal, or reuse the production `.checkpoints` store
  (same-fingerprint reuse silently republishes stored verdicts and measures
  nothing — `redecide…py:14-18`).
- Write anything into `artifacts_canary/renovation_session9_20260818`,
  `reports/review_queue.json`, `reports/review_verdicts.jsonl`, or
  `reports/session9_*`. All frozen.
- Modify `scripts/replay_renovation_architecture.py` — it is structurally
  provider-free and stays that way.

---

## Appendix A — verified numbers (all confirmed 2026-08-31)

**Canary replica 1** (`artifacts_canary/renovation_session9_20260818/run_1/candidate`),
18 listings, all envelopes `state == "complete"`:

| quantity | value |
|---|---|
| conditions | 1,047 |
| review units (Terra calls) | 148 |
| distinct images attached | 538 |
| input / output / total tokens | 842,895 / 75,685 / **918,580** |
| stored verdicts | `supported` 882 (84.2%) · `unsupported` 136 (13.0%) · `cannot_assess` 29 (2.8%) |
| dispositions | `accepted_for_work` 743 · `excluded` 190 · `no_action` 78 · `inspection` 31 · `withheld` 5 |

**Labels** (`reports/labels_v1_1.json`, `label_version` `v1.1`, 125 cards; 88
canary / 37 production). Canary classes:

| class | cards | note |
|---|---|---|
| `supported_billed` | 57 | G1 population |
| `dirB_recovery` | 16 | G2 population |
| `misnamed_billed` | 7 | G3a — exploratory |
| `trivial_billed` | 5 | G3b — exploratory |
| `hard_false_billed` | 3 | G3c — directional |

Weighted canary billed-error mass: 82.100 / 743 accepted = **11.05%**, split
misnamed 40.550 (49.4%) · trivial 21.775 (26.5%) · absent 19.775 (24.1%).
Production adds only +1 misnamed and +1 trivial (its error mass is ~90%
absent-class) — the reason this is canary-only.

Labels carry no `run_id`; join `rc_*` → `reports/review_queue.json` for
`(property_key, condition_id, run_id)`. All canary labels reference **run_1**
runs, across 17 listings — `redfin_10803207` has zero labeled cards. No labeled
card is touched by the `MIN_EVIDENCE_PX = 500` thumbnail exclusion (canary
`low_res_excluded` is empty) or by the 11 orphaned verdicts (all from
production `redfin_10965375`).

**Fingerprints / pins**

| item | value |
|---|---|
| working-tree projection fingerprint | `6baacaf6254861409e1c6c4495e73d57c573523f581a138cc703a302f87b9308` |
| stored canary projection fingerprint | `ff62acaeb9a71cb9652687bd62ec2c1117c8f46ceb71f41a355c46b0b86bfacf` |
| stored canary `catalog_sha256` (canonicalized) | `d1cef743f379918fa17f0684c245779180d7d692ebc8fd3e66b7c69bebf68347` |
| catalog raw file sha, `07ee112`→`c833da4` | `20adb6e342f2d3a5304e90ae6dae7d8bf1724c53feab22358130081582abfd90` |
| canary freeze commit | `07ee112` |
| **pin for the replay** | **`c833da4`** (= `a9ed7ad^`), fallback `09e28d1` |
| Catalog 3.2 (the drift) | `a9ed7ad` |
| model | `gpt-5.6-terra`, `max_output_tokens` 8192, reasoning effort `medium` |

**Replica noise floor:** Terra `run_1` vs `run_2` verdict flips 6.4% (16/250);
humans matched the two replicas 8:6.
