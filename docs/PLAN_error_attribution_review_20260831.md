# Plan: error attribution review — execution spec

Written 2026-08-31. Companion to `docs/HANDOFF_error_attribution_review.md`
(the handoff carries the background, cohort, rubric and truth-basis rules; this
doc carries **how to run the review** as an ultracode workflow). Where the two
disagree, this doc wins — every deviation is listed in §2 with its reason.

Verified against the frozen artifacts on 2026-08-31 by a 4-agent recon pass
(report-script rules, queue statistics, builder/tests semantics, photo
readability). Facts below marked *(verified)* come from that pass, with
file:line references into the actual code.

Offline only, unchanged from the handoff: no provider calls, no pipeline
reruns, no prompt or production changes. One tooling change is required (§3);
it touches only the offline report script and its tests.

---

## 1. What recon adds to the handoff (load-bearing deltas)

1. **Gold verdicts cannot enter the ledger as-is.** Any verdict whose
   `case_id` is not a key of the queue's `cases[]` is a hard
   `ReconciliationError` — exit 1, nothing written, **not** relaxed by
   `--allow-incomplete` (`scripts/error_attribution_report.py:106-108`).
   Handoff §10's `ga_`/`gx_` ids therefore need the §3 extension. The report
   was clearly built expecting this: lanes `miss_gold` and `halluc_gold_extra`
   already sit in its `MISS_LANES`/`HALLUC_LANES` constants (lines 42-43) and
   would pool into the headline denominators — the merge path is the only
   missing piece. *(verified)*
2. **Evidence images live in the FE repo image store**, not under either
   artifact root: all 83 case photos + all 15 gold photos are
   `C:\Users\Steven\IntelliJProjects\renointel-prod\public\images\properties\<listing>\photo_NNN.jpg`.
   All exist on disk, all `image_exists` flags agree, and the Read tool
   renders them (215–240KB JPGs, spot-checked with content descriptions).
   Agents use the queue's `image_path` verbatim — no path derivation needed.
   *(verified)*
3. **Latest-wins is pure file line order; `ts` is never consulted.** Two
   silent footguns when writing the ledger (`report.py:56-80`): a malformed
   JSON line **vanishes silently**, and a line whose `attribution` key is
   *missing* (not just null) acts as an **undo**. Consequence: the ledger is
   only trustworthy after a report run — rerun `--allow-incomplete` after
   every append batch as the integrity check. *(verified)*
4. **`ts`, `reviewer`, `p2a_excerpt`, `gold_match` are carried-only** — never
   validated, never copied into `error_attribution.{md,json}`. They survive
   only in the jsonl, so the RESULT doc's gold section and any excerpt quotes
   must read the jsonl / gold files directly. Extra keys on verdict lines are
   legal. *(verified)*
5. All 55 attributable cases: `status` pending, `join_methods == ["exact"]`
   (the join-rationale rule can never fire), 0 untraceable. Truth-basis split:
   **45 v1_1 / 10 v1_only** (miss_v1only 3 + halluc_v1only 3 +
   appendix_inconclusive 4). *(verified)*
6. Case record sizes: min 5,908 / median 8,194 / max 31,436 bytes
   (`rc_5063066c1a57`). Photos per case: 1→37, 2→12, 3→4, 4→1, 6→1 (83 total).
   *(verified)*
7. Gold: 15 photos on **two canary properties only** (redfin_10806500 ×7,
   redfin_11000447 ×8); findings per photo 2–10 (median 8), 112 total; 64
   conditions touch them, 52 accepted. `gold_id`s (`g1…gN`) are **scoped per
   photo** — only the composite `ga_<property>_<photo>_<gold_id>` is unique.
   Gold `lineage` is a flat dict (single photo), not `per_photo`. *(verified)*
8. **Duplicate-condition guard**: the same `(source, run_id, condition_id)`
   triple in two cases is a hard error (`report.py:94-104`, skipped when
   `condition_id` is empty). Gold cases anchored on real conditions must
   therefore be deduped against existing `rc_` cases first (§8.3). *(verified)*
9. **Provenance quirk**: the report hashes the *default* queue/verdicts paths
   into `inputs` regardless of `--queue`/`--verdicts` overrides
   (`report.py:211-215`). Final runs must run against the real default paths.
   *(verified)*
10. Queue is tracked in git, sha256
    `b58dcca7c3779f2b0539f988077f865c6c1ca040d23a44646d89b04a398940df`,
    `generated_at 2026-08-31T19:55:06Z`. **Do not regenerate mid-review** —
    `generated_at` alone changes the sha and breaks provenance. *(verified)*
11. `confidence` has no enforced enum; the only rule is basis `v1_only` +
    literal `"high"` fails (`report.py:134-136`). We enforce the cap at the
    **schema level** (§7) so the violation is unrepresentable.
12. The verdict token `excluded` is overloaded in the record: it is also a
    v1.1 label class (→ lane appendix_inconclusive) and a condition
    disposition (Terra rejected). The rubric text (§6) disambiguates
    explicitly so agents don't pattern-match the wrong sense. *(verified)*

## 1b. Corrections found during execution (2026-08-31)

**C1 — the handoff's 2e claim is imprecise, though its conclusion holds.**
Handoff §3 says "2e removed nothing at all" and infers that a 2e attribution
is structurally impossible. The first half is only true of one mechanism.
Measured over the frozen queue:

| signal | value |
|---|---|
| `p2e.removed_count` | 0 on all 143 photos ✓ (what the handoff measured) |
| `p2e_status` per issue | kept 102, **absent 27**, unknown 4 |
| photos carrying `suppressed_reason_counts` | **119 of 143** |
| suppression reasons | `tier_optional_suppressed` 225, `drop_if_generic` 39 |
| **issues that still projected a condition** | **133 of 133 — including all 27 absent and all 4 unknown** |

So 2e *does* suppress, on 18 attributable cases; it simply never costs a
condition, because `condition_projection` emits the condition regardless.
The conclusion (2e is never the first responsible stage) survives — for a
stronger reason than the handoff gave. The prompt states these mechanics
explicitly, because a reviewer who is told "2e removed nothing" and then reads
`p2e_status: "absent"` has been handed an apparent contradiction, which
discredits the rest of the rubric. D4 stands, now on measured grounds.
The RESULT doc must state the refined version, not the handoff's.

**C2 — the ladder has no rung after Terra.** A condition that reached Terra,
was *supported*, and still went unbilled (disposition `no_action` /
tier-optional routing) fits no stage. The prompt now tells reviewers to answer
`unclear` and say so, rather than forcing a wrong rung. Surfaced by the gold
pilot agent unprompted.

**C3 — the worked example's reference answer is contestable.** See §5.1.

## 2. Deviations from the handoff (Steven reviews these)

| # | Deviation | Why | Fallback if rejected |
|---|---|---|---|
| D1 | **Report extension `--gold-cases`** (§3) so gold verdicts flow through the validated report | Handoff §10's ga_/gx_ ledger entries hard-fail the orphan check (§1.1). The gold lanes are already in the report's constants; this completes the designed seam | Zero-touch fallback: separate `reports/error_attribution_gold_verdicts.jsonl` + hand-assembled gold numbers in the RESULT doc only. Works, but gold never passes validation and the miss/halluc headlines exclude it |
| D2 | **Pilot gate**: 5-agent calibration run before the main fan-out, gated on reproducing the worked example (§5.2) | `rc_128caa6212b8` has a validated answer (downstream/2d). One cheap run proves the template before ~70 agents consume it | Skip; first drift is then discovered in the consistency sweep after full spend |
| D3 | Skeptic set = `pass_2a`, `unclear`, **and `excluded`** (handoff: first two) | `excluded` overrules an adjudicated human verdict — a stronger claim than `downstream`; it deserves the same adversarial check. Expected volume ~0–3 | Handoff scope: pass_2a + unclear only |
| D4 | `2e` and `untraceable` **removed from agent-facing enums** (§7) | 2e removed nothing in this corpus (structurally impossible — handoff §3); untraceable is the builder's determination, not a reviewer judgment, and 0 attributable cases carry it. Schema-level removal beats prompt-level warnings | Keep full enum + prompt warnings |
| D5 | Gold dedupe guards (§8.3): no new gold case for a condition already carried by an `rc_` case; `matched` requires an **accepted** condition | Forced by the duplicate-condition guard (§1.8) and by not double-counting errors already in the miss/halluc lanes | None — required for the report to pass |
| D6 | Gold reverse-pass confirm step: an accepted condition with no gold match that the photo **does** support is gold-incompleteness (recorded, no case), not a hallucination | Same trap as the forward direction — gold is photo-observation truth, not exhaustive truth. Consistent with §10's own out-of-catalog logic | Treat every unmatched extra as halluc_gold_extra (inflates the lane with false positives) |

Everything else is handoff-verbatim: one agent per case, rubric §5 applied as
written, no pooling of v1 and v1.1, mechanical_hints never substitutes for
reading, `unclear` is a real answer, consistency sweep done inline by the
orchestrating session.

## 3. Pre-review tooling task: `--gold-cases` (D1)

Small extension to `scripts/error_attribution_report.py` + tests. Do this
**first**; it gates the finish line. Spec:

- New optional arg `--gold-cases reports/error_attribution_gold_cases.json`.
  File shape: `{schema_version, generated_at, source_note, matching_table,
  cases: [...]}` where each case carries exactly the fields
  reconcile/tally touch: `case_id` (`ga_`/`gx_` prefix), `lane`
  (`miss_gold` | `halluc_gold_extra`), `attribute: true`, `status: "pending"`,
  `human_truth: {basis: "gold", ...gold finding text/slug...}`, `v5_claim`
  (real `condition_id` when anchored on one, `""` otherwise — keeps the dupe
  guard active exactly where it should be), `run_ref`,
  `mechanical_hints: {join_methods: []}`.
- Merge into the case dict **after** the queue-header `lane_counts` check
  (that check stays frozen-queue-only). New checks: gold ids unique, prefixed
  `ga_`/`gx_`, no collision with `rc_` ids, gold lanes only, `basis == "gold"`.
- Per-verdict rules, completeness, and tally then apply to merged cases
  automatically (gold lanes already pool into the miss/halluc headlines via
  `MISS_LANES`/`HALLUC_LANES`). Verify `by_basis` picks up `"gold"` as a third
  row dynamically; adjust if hardcoded.
- Stamp the gold file's path+sha256 into `inputs` (do it properly for the new
  arg; the existing default-path quirk (§1.9) is out of scope — note it, don't
  fix it).
- Extend `report["gold"]` pass-through with the matching counts
  `{matched, miss_candidates, out_of_catalog, extras, gold_incomplete,
  already_cased}` read from `matching_table`.
- Tests to add (in `tests/test_error_attribution.py` style): merged gold case
  validated + counted in the misses headline; `gx_` duplicating an `rc_`
  condition triple fails; unknown lane fails; without `--gold-cases` a `ga_`
  verdict still fails as an orphan (pins that the old behavior is unchanged).
- Suite must stay green: `.venv\Scripts\python.exe -m pytest tests/test_error_attribution.py -q`

Budget ~60–90 lines of code + tests. No behavior change when the flag is
absent.

## 4. Prep (orchestrating session, inline)

1. Preflight: pytest green (18+new); queue sha256 equals §1.10's value; Read
   one evidence photo and one gold photo to confirm vision; confirm
   `reports/error_attribution_verdicts.jsonl` does not exist (or note prior
   content).
2. Split the queue into per-agent files in the session scratchpad (PowerShell
   5.1 mangles inline `python -c` quotes — write a script file):
   - `<scratch>/attribution_cases/<case_id>.json` — the 55 attributable case
     records, verbatim.
   - `<scratch>/gold_photos/<property>__<photo_key>.json` — the 15 gold
     records, verbatim.
3. Compute the **rc_ exclusion set** for the gold reverse pass: every
   `v5_claim.condition_id` of any case (all 104, both flags) whose
   `run_ref.property_key` is redfin_10806500 or redfin_11000447, mapped to its
   `case_id` + lane. Injected into gold prompts and used by §8.3.
4. Pick pilot cases (deterministic): `rc_128caa6212b8` (worked example,
   miss_label) + lowest case_id in `halluc_label`, `appendix_misnamed`, and
   `miss_v1only` (exercises the v1_only schema variant) + `gold_photos[0]`
   (redfin_10806500/photo_001).

## 5. Workflow shape

One workflow **script**, two invocations with different `args` — this is what
guarantees the pilot and the main run use byte-identical templates. Agents
return structured output only; **no agent writes any file**. The orchestrating
session owns the ledger.

```
args = { cases: [{case_id, lane, basis, file}...],
         gold:  [{photo_id, file, excluded_condition_ids: [...]}] }

phase Review:  pipeline(args.cases,
                 c => agent(casePrompt(c), {schema: c.basis=='v1_only' ? V_V1ONLY : V_V11}),
                 (r,c) => needsSkeptic(r) ? agent(skepticPrompt(c,r), {schema: SKEPTIC, phase:'Verify'})
                                            .then(s => ({c, r, s}))
                                          : {c, r, s: null})
phase Gold:    pipeline(args.gold,
                 g => agent(goldPrompt(g), {schema: GOLD}),
                 (r,g) => parallel(r.proto_cases.filter(needsSkeptic).map(p => () =>
                            agent(goldSkepticPrompt(g,p), {schema: SKEPTIC, phase:'Verify'})))
                          .then(sk => ({g, r, sk})))
return everything
```

`needsSkeptic` = attribution ∈ {pass_2a, unclear, excluded} (D3). No barriers:
each case's skeptic runs as soon as its review lands. Timestamps: workflow
scripts cannot call `Date.now()` — the session stamps real `ts` values at
ledger-write time (§8.2).

### 5.1 Invocation 1 — pilot (5 agents)

The 4 pilot cases + 1 gold photo from §4.4. **Gate** (all must hold before
invocation 2):

- `rc_128caa6212b8` returns `downstream` / `2d`. This answer was validated
  end-to-end by the session that built the tooling; a mismatch is a template
  failure — fix the template, re-run the pilot, never rationalize it away.

  **Outcome 2026-08-31: the stage did not match, and the evidence favours the
  agent.** Pilot run 1 returned `downstream` / **`terra`**, high confidence.
  Adjudicated on the record rather than accepted: the case's frozen
  `human_truth` carries `claim: "exact"`, `work: "warranted"`, slug
  `exact_and_warranted`, and the note *"It might actually be vinyl so ill mark
  it as true this time"*. If the human adjudicated the billed claim text
  (`vinyl or linoleum flooring torn, worn through, or lifting material`) as
  **exact**, then 2d did not resolve to a wrong item, and the first stage that
  actually lost the condition is Terra's rejection
  (`reason_code: verdict_unsupported`). The handoff's §5 narrative — "2d
  resolved it to a material-specific catalog item" — treats the material
  specificity as 2d's error, but the human's own note files that as a
  *forward-looking catalog suggestion* ("we might want to change the catalog to
  'hard flooring'"), not as a mis-resolution.

  This is recorded as a disagreement with the reference answer, not as a passed
  gate. It does not move the headline: both readings are `downstream`, so only
  the `downstream_stages` distribution is affected. The gate's purpose —
  evidence-grounded reasoning with the photos actually read — was met. The
  RESULT doc must carry this disagreement explicitly so it stays disputable.
- All outputs pass the §8.1 validation mirror (including: `p2a_excerpt` is a
  verbatim substring of some `per_photo` `p2a_prose`).
- Every photo of every pilot case appears in `per_photo_reading` (proves the
  agent opened them).
- Rationales attribute a stage; they do not re-litigate the human verdict.
- Gold pilot: all of photo_001's findings have exactly one decision; every
  accepted condition is either matched or handled in the reverse pass.

### 5.2 Invocation 2 — main run

Remaining 51 cases + 14 gold photos, same script. Pilot verdicts are kept
(they were produced by the identical template). Concurrency is capped ~16 by
the runtime; expect roughly 60–120 minutes wall-clock.

### 5.3 Budget

| Stage | agents | est. |
|---|---|---|
| Pilot (cases + gold + any skeptics) | 5–7 | ~0.15M |
| Case reviews | 51 | 1.0–1.5M |
| Case skeptics (est. 20–40% trigger) | ~10–20 | 0.15–0.35M |
| Gold photos | 14 | 0.45–0.7M |
| Gold skeptics | ~3–8 | 0.05–0.15M |
| Assembly, sweep, report, RESULT (inline) | — | ~0.1M |
| **Total** | **~85–100** | **~2.0M expected, 2.8M worst** |

Handoff budgeted 1.5–2.5M; D2+D3 add the delta. Levers if it runs hot, in
order: drop `excluded` from the skeptic set (back to handoff scope), then drop
skeptics for `unclear` (keep them for `pass_2a` — the headline claim keeps its
adversarial check no matter what).

## 6. Frozen prompt templates

Rules: the rubric block is byte-identical in every case-agent prompt — only
the marked injection slots vary. It deliberately **excludes the handoff's
worked example** (`rc_128caa6212b8` is a pilot case; feeding agents its
answer, or anchoring every flooring case on it, contaminates the run). Agents
are restricted to the Read tool: their case file and its image paths, nothing
else.

### 6.1 Case agent

```
You are reviewing one frozen error case from a real-estate photo-analysis
pipeline, offline. Read the case record at: {case_file}
Then use the Read tool to open EVERY evidence photo listed in
lineage.per_photo (field image_path). Use only the Read tool, only on the
case file and its photos.

The pipeline stages, in order:
  2a  VLM prose reading of each photo (lineage.per_photo[*].p2a_prose)
  2b  structured bullets from the prose (p2b_bullets)
  2c  bullet filter (survivors in p2c_surviving; drops in p2c_dropped —
      no rationale is ever recorded for a drop)
  2d  catalog item resolution (lineage.per_issue[*].p2d.resolved_item_id)
  2e  post-filter — it removed NOTHING in this entire corpus; it is not an
      available attribution
  condition_projection  issue -> billable condition
      (per_issue[*].projected_condition_id)
  terra  final accept/reject verdict on the condition (v5_claim.terra_verdict,
      terra_rationale)

This case's lane: {lane}. Premise: {lane_premise}
Truth basis: {basis}. {basis_note}

STEP 1 — read the photo(s) yourself and confirm the premise holds. If after
honest inspection it does not, set lane_confirmed=false and
attribution="excluded" with a rationale saying exactly what you saw. Expect
this to be rare: the premise comes from an adjudicated human review, and
disagreeing with it is itself a strong claim. (Vocabulary warning: "excluded"
in YOUR output is a verdict word. The same token appears in the record as a
v1.1 label class and as a condition disposition — unrelated meanings.)

STEP 2 — attribute the error to the FIRST stage that went wrong.
MISS (a real condition the pipeline failed to bill):
  pass_2a  when NO evidence photo's p2a_prose contains the real condition.
           With multiple photos this means absence from ALL of them — check
           every per_photo entry.
  downstream + first_responsible_stage otherwise:
           no matching 2b bullet -> 2b; bullet present but dropped -> 2c;
           resolved to the wrong catalog item -> 2d; surviving issue never
           became a condition -> condition_projection; condition existed and
           Terra rejected it -> terra.
HALLUCINATION (a billed claim the photo does not support):
  pass_2a  when the unsupported material claim ALREADY APPEARS in the 2a
           prose.
  downstream when 2a contains only a true or weaker observation and a later
           stage introduced the false meaning — same ladder, first stage that
           distorted it.
"unclear" is a real answer. Use it instead of guessing.

Rules:
- p2a_excerpt: copy VERBATIM (an exact substring) from one per_photo
  p2a_prose, or null when nothing relevant exists there. It is your chosen
  evidence sentence, not provenance.
- mechanical_hints is deterministic plumbing. Never let it stand in for
  reading the prose and the photos.
- human_truth.retag_answer is validation context only — never derive the
  truth axes from it. v1_verdict / v1_class ride along for context only.
- context.in_factorized_disagreements is model output. A column, not
  evidence.
- Your rationale must name what the 2a prose did and did not contain, and why
  the stage you chose is the first one that went wrong.
```

Injection slots: `{case_file}`; `{lane}`; `{lane_premise}` from the fixed
table below; `{basis}`; `{basis_note}` = for v1_only: *"single collapsed v1
verdict, never re-adjudicated — weaker. Confidence high is not available to
you (your schema excludes it)."*, for v1_1: *"adjudicated v1.1 verdict (claim
and work axes) — the authority."*

| lane | premise line |
|---|---|
| miss_label / miss_v1only | The human adjudicated this condition as REAL; the pipeline failed to bill it. Attribute as a MISS. |
| halluc_label / halluc_v1only | The pipeline BILLED this claim; the human adjudicated it as not supported by the photo. Attribute as a HALLUCINATION. |
| appendix_misnamed | A real problem was billed under the WRONG NAME. Your question: which stage chose the wrong name — did 2a read the photo wrong, or did 2b/2c/2d re-label a correct reading? Use the miss ladder with "the real condition" = the correctly-named problem. |
| appendix_trivial | The condition is TRUE but too trivial to bill. Your question: which stage inflated it — did 2a overstate the photo, or did a later stage escalate a proportionate observation? Use the hallucination ladder with "the unsupported material claim" = the billable-severity framing. |
| appendix_inconclusive | The human could not reach a verdict. Decide whether the record still lets you attribute cleanly; when it does not, unclear is the honest answer. |

### 6.2 Skeptic agent

```
You are an adversarial verifier. Another reviewer attributed a pipeline error;
your job is to REFUTE the attribution if the evidence allows it. Read the case
record at {case_file} and open every evidence photo in lineage.per_photo with
the Read tool. Use only the Read tool.

[same stage table + vocabulary warning as 6.1]

The verdict under attack:
  attribution: {attribution}   stage: {stage_or_dash}
  rationale: {rationale}
  p2a_excerpt: {excerpt_or_null}

Argue the strongest case AGAINST it from the same frozen evidence: quote the
specific p2a_prose text, bullets, drops, resolved item, or Terra rationale
that contradicts the verdict. If the strongest counter-argument is weak, say
the verdict stands and say why. You are not asked to be balanced — you are
asked to attack, and then to report honestly whether the attack succeeded.
Do not dispute the human truth verdict itself; dispute only the stage
attribution built on it.
```

### 6.3 Gold photo agent

```
You are reviewing one gold-annotated photo, offline. Read the gold record at
{gold_file}; open lineage.image_path with the Read tool. Use only the Read
tool. The record carries: gold_findings (photo-observation truth for THIS
photo), conditions_on_photo (every v5 condition touching it, with accepted
flags and Terra verdicts), and the full single-photo pipeline lineage
(p2a_prose, p2b_bullets, p2c_surviving, p2c_dropped).

Gold is PHOTO-OBSERVATION truth, not billable-condition truth. Some findings
are true but normal-for-context (the gold notes flag redfin_10806500
photo_003/007 basement shots specifically). And gold is not guaranteed
exhaustive.

[same stage table as 6.1]

PART 1 — for EVERY gold finding, exactly one decision:
  matched         an ACCEPTED v5 condition covers it -> record that
                  condition_id. No error.
  miss_candidate  the v5 catalog could have carried it but no accepted
                  condition did. If a REJECTED condition covered it, record
                  that condition_id as covering_rejected_condition_id.
  out_of_catalog  nothing in the v5 catalog could bill this (including
                  normal-for-context observations). NOT a miss.
PART 2 — for every ACCEPTED condition not used in a match: does the photo
  itself support the claim? If YES: gold is incomplete here — record it, no
  error case. If NO: it is a hallucination candidate.
  Skip conditions in this list entirely (they already have review cases):
  {excluded_condition_ids}
PART 3 — for each miss_candidate and each unsupported extra, attribute it
  with the same ladder as 6.1 (miss ladder for candidates, walking this
  photo's lineage forward from p2a_prose; hallucination ladder for extras,
  anchored on the condition). unclear is a real answer.
```

## 7. Output schemas (StructuredOutput, enforced at the tool layer)

Case verdict — two variants differing ONLY in the confidence enum
(`V_V11`: high|medium|low; `V_V1ONLY`: medium|low — makes the v1_only cap
unrepresentable):

```json
{ "type": "object", "additionalProperties": false,
  "required": ["lane_confirmed", "attribution", "first_responsible_stage",
               "confidence", "rationale", "p2a_excerpt", "per_photo_reading"],
  "properties": {
    "lane_confirmed": {"type": "boolean"},
    "attribution": {"enum": ["pass_2a", "downstream", "unclear", "excluded"]},
    "first_responsible_stage": {"enum": ["2b", "2c", "2d",
                                          "condition_projection", "terra", null]},
    "confidence": {"enum": ["high", "medium", "low"]},
    "rationale": {"type": "string", "minLength": 40},
    "p2a_excerpt": {"type": ["string", "null"]},
    "per_photo_reading": {"type": "array", "items": {"type": "object",
      "required": ["photo_key", "what_the_photo_shows", "p2a_covers_condition"],
      "properties": {"photo_key": {"type": "string"},
                      "what_the_photo_shows": {"type": "string"},
                      "p2a_covers_condition": {"enum": ["yes", "no", "partial"]}}}}}}
```

Note what is absent by design (D4): no `2e`, no `untraceable`, no `case_id`
(the script attaches it from the assignment — an agent can never mislabel its
own case), no `ts`/`reviewer` (stamped at write time). The stage-iff-downstream
rule is validated by the orchestrator (§8.1) since JSON Schema conditionals
aren't worth the retry risk.

Skeptic:

```json
{ "required": ["verdict_stands", "strongest_counter", "alternative_attribution",
               "alternative_stage", "refutation_confidence"],
  "properties": {
    "verdict_stands": {"type": "boolean"},
    "strongest_counter": {"type": "string", "minLength": 40},
    "alternative_attribution": {"enum": ["pass_2a", "downstream", "unclear",
                                          "excluded", null]},
    "alternative_stage": {"enum": ["2b", "2c", "2d", "condition_projection",
                                    "terra", null]},
    "refutation_confidence": {"enum": ["high", "medium", "low"]}}}
```

Gold photo:

```json
{ "required": ["photo_reading", "finding_decisions", "extra_conditions",
               "proto_cases"],
  "properties": {
    "photo_reading": {"type": "string"},
    "finding_decisions": {"type": "array", "items": {"type": "object",
      "required": ["gold_id", "decision", "matched_condition_id",
                   "covering_rejected_condition_id", "note"],
      "properties": {"gold_id": {"type": "string"},
        "decision": {"enum": ["matched", "miss_candidate", "out_of_catalog"]},
        "matched_condition_id": {"type": ["string", "null"]},
        "covering_rejected_condition_id": {"type": ["string", "null"]},
        "note": {"type": "string"}}}},
    "extra_conditions": {"type": "array", "items": {"type": "object",
      "required": ["condition_id", "photo_supports_claim", "note"],
      "properties": {"condition_id": {"type": "string"},
        "photo_supports_claim": {"enum": ["yes", "no", "partial"]},
        "note": {"type": "string"}}}},
    "proto_cases": {"type": "array", "items": {"type": "object",
      "required": ["kind", "anchor", "attribution", "first_responsible_stage",
                   "confidence", "rationale", "p2a_excerpt"],
      "properties": {"kind": {"enum": ["miss_gold", "halluc_gold_extra"]},
        "anchor": {"type": "string"},
        "attribution": {"enum": ["pass_2a", "downstream", "unclear", "excluded"]},
        "first_responsible_stage": {"enum": ["2b", "2c", "2d",
                                              "condition_projection", "terra", null]},
        "confidence": {"enum": ["high", "medium", "low"]},
        "rationale": {"type": "string", "minLength": 40},
        "p2a_excerpt": {"type": ["string", "null"]}}}}}}
```

## 8. Assembly (orchestrating session, inline)

### 8.1 Validation mirror (script it; run on pilot and on every batch)

Per case verdict: attribution in the 4-value agent enum;
`first_responsible_stage` non-null **iff** downstream; rationale non-empty;
`p2a_excerpt` null or an exact substring of some `per_photo[*].p2a_prose` for
that case; v1_only ⇒ confidence ≠ high (schema makes this unreachable —
check anyway); `per_photo_reading` covers every photo key; a `pass_2a` MISS
additionally requires every `p2a_covers_condition` = no. Per gold result:
every `gold_id` decided exactly once; `matched` ⇒ `matched_condition_id`
present and accepted; every accepted, non-excluded condition either matched
or in `extra_conditions`; proto_cases exactly consistent with decisions
(one `miss_gold` per miss_candidate; one `halluc_gold_extra` per
`photo_supports_claim: "no"` extra; none otherwise). Violations: re-run that
one agent once with the violation named; if it fails again, the orchestrator
reviews the case itself during the sweep — never auto-rewrite an agent's
judgment, and never let a hole ride silently.

### 8.2 Ledger write protocol

- Build all initial lines in memory, validate, then write
  `reports/error_attribution_verdicts.jsonl` in one pass. Line fields:
  `case_id`, `ts` (real UTC at write time), `reviewer`
  (`claude_case_agent` / `claude_gold_agent` / `claude_orchestrator`),
  `attribution` (**always present** — a missing key is an undo, §1.3),
  `first_responsible_stage` (only when downstream), `confidence`,
  `rationale`, `p2a_excerpt`, `lane_confirmed`, `skeptic_checked` (bool),
  and `gold_match` on gold-lane lines. `json.dumps` with `ensure_ascii=True`
  (known lone-surrogate hazard in this codebase's artifact text).
- After every append batch:
  `.venv\Scripts\python.exe scripts\error_attribution_report.py --allow-incomplete`
  (plus `--gold-cases` once §8.3 exists) — this is the only integrity check
  that catches the silent-line footguns.
- Revisions (from skeptic reconciliation or the sweep) are **appended** lines,
  reviewer `claude_orchestrator`, rationale stating both views and why the
  revision won. Never edit existing lines; the ledger is the audit trail.

### 8.3 Gold materialization

From the gold agents' outputs, the session writes
`reports/error_attribution_gold_cases.json` (§3 shape):

- `matching_table`: every finding decision and every extra, verbatim, plus
  the pairings — this is the §10 "re-derivable and disputable" record
  (precedent: `NOTE_THEMES` in `scripts/review_analysis.py`).
- Guards, applied in order: (1) any anchor `condition_id` in the §4.3
  rc_ exclusion set ⇒ **no new case**; record `already_cased: rc_…` in the
  matching_table (if that rc_ case sits in a counted lane while gold says the
  finding is real, flag it as a gold-vs-human conflict for the RESULT — do
  not create a case). (2) `matched` requires accepted; a covering-rejected
  condition makes the finding a miss_candidate anchored on that condition.
  (3) Case ids: `ga_<property>_<photo>_<gold_id>` / `gx_<property>_<condition_id>`.
- Verdict lines for materialized cases go into the same ledger with
  `gold_match` carrying the pairing, and pass through the same skeptic rule
  (D3) — their skeptics already ran inside the workflow.

### 8.4 Consistency sweep (inline — the orchestrator reads everything)

1. Re-run the §8.1 mirror over the complete ledger.
2. Stage-boundary drift, the real fan-out risk: sample rationales per stage;
   the confusable pairs are 2b-vs-2c (no bullet vs dropped bullet — the
   record answers this mechanically; rationales must match the record) and
   2d-vs-terra on rejected conditions (wrong-item-then-refuted vs
   right-item-wrongly-refuted). Check that like fact patterns got like
   stages; do NOT invent new policy while reconciling.
3. Every `pass_2a` and `excluded` verdict: confirm its skeptic ran; read both
   sides; decide; append revisions where the skeptic wins.
4. `unclear` rate: if it exceeds ~25–30% of judged, say so prominently — the
   factorized verifier's 28.9% unclear was a finding about question quality
   (RESULT_factorized_replay §, memory), and this review must not quietly
   repeat it.
5. `lane_confirmed: false` list: each is an `excluded` with a photo-grounded
   rationale, reported by name in the RESULT.
6. Any `2e` or enum violation = agent/template bug, not data (structurally
   impossible; the schema should have prevented it — investigate).

## 9. Finish

1. `.venv\Scripts\python.exe -m pytest tests/test_error_attribution.py -q` — green.
2. Final strict run, real default paths (§1.9), no `--allow-incomplete`:
   `.venv\Scripts\python.exe scripts\error_attribution_report.py --gold-cases reports/error_attribution_gold_cases.json`
   Exit 0 required; it rewrites `reports/error_attribution.{md,json}`.
3. Write `docs/RESULT_error_attribution_20260831.md` citing
   `reports/error_attribution.json`. Required content:
   - Headline splits for misses / hallucinations / appendix: judged
     denominators, pass_2a vs downstream shares, `downstream_stages`
     distribution; gold lanes shown both pooled (the report does this) and
     separately (`by_lane`).
   - Gold lane table: matched / miss_candidates / out_of_catalog / extras /
     gold-incomplete / already_cased (+ any gold-vs-human conflicts). Note
     that the md report has no gold section — this table reads from the gold
     file and the ledger (§1.4).
   - `by_basis`: v1_only reported on its own row, never pooled with v1_1.
   - Skeptic outcomes: challenged / upheld / overturned counts.
   - The 2e finding stated as the handoff demands: removal was structurally
     impossible in this corpus, not merely unused.
   - Limitations, verbatim from handoff §11 (2c presence/absence only;
     excerpts are interpretation; cohort-not-system-rates + `not_shown`
     caveats; 80% human self-consistency and the 6.4% Terra replica floor —
     an attribution split inside that noise is not a finding), plus: gold
     covers only two canary properties (§1.7), and the D-series deviations.
4. Commit: ledger, gold cases file, `reports/error_attribution.{md,json}`,
   RESULT doc, the §3 report extension + tests. The queue file must show
   **no diff**.

## 10. Failure modes

- **Agent returns null** (skipped/died): pipeline drops it to null — collect
  the case ids and re-run just those (same script, args = the stragglers).
- **Workflow interrupted**: resume with `resumeFromRunId` — unchanged
  (prompt, opts) pairs replay from cache; read `journal.jsonl` before assuming
  a cached result is non-empty.
- **Budget overrun**: §5.3 levers, in order. Never economize by batching
  multiple cases into one agent — independence is what makes the sweep's
  drift check meaningful.
- **Do not**: regenerate the queue (§1.10); call any provider; substitute a
  run for a missing condition (`condition_id` is run-scoped — handoff §12);
  write a ledger line without an `attribution` key (§1.3); trust an append
  without a report run after it (§1.3); let a skeptic auto-override (the
  orchestrator decides, §8.4.3).

## 11. Kickoff prompt for the execution session

> ultracode. Run the error attribution review per
> `docs/PLAN_error_attribution_review_20260831.md` (execution spec; its §2
> deviations are approved) with `docs/HANDOFF_error_attribution_review.md`
> as background. Scale is pre-approved: ~85–100 agents across the pilot and
> main workflows, budget ~2.0–2.8M tokens, levers in plan §5.3. Order: §3
> tooling extension first, then §4 prep, §5.1 pilot (gate on the worked
> example), §5.2 main run, §8 assembly + sweep, §9 finish. Offline only — no
> provider calls, no queue regeneration. Deliverables: complete
> `reports/error_attribution_verdicts.jsonl`, gold cases file, a strict
> (no-flag) report run, `docs/RESULT_error_attribution_20260831.md`, commit.

Strike any §2 deviation before kickoff and the plan degrades gracefully to
the listed fallback.
