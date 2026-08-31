# Handoff: Pass 2a vs. downstream error attribution — the review session

Status as of 2026-08-31: **tooling built and run, review not started.**
This session built the extractor, ran it, and froze the case queue. The next
session does the review, runs the report, and writes the RESULT doc.

Self-contained by design: everything needed is in this doc plus the two scripts
and the frozen queue. You do not need the session that produced them.

---

## 1. The question

For known v5 misses and hallucinations, did the error originate in **Pass 2a**
(the VLM's raw prose reading of the photo) or **downstream** (2b/2c/2d/2e,
condition projection, or Terra)?

Every prior analysis stops at "Terra was wrong" or "the claim was false".
Nothing in the repo joined a v5 condition back to the 2a prose. That join is now
built; what remains is judging the semantics case by case.

Offline only: no provider calls, no pipeline reruns, no prompt or production
changes. This is an audit of frozen artifacts.

## 2. What is already done

| Artifact | State |
|---|---|
| `scripts/build_error_attribution_queue.py` | built, run, tested |
| `scripts/error_attribution_report.py` | built, tested (fixtures); runs clean on an empty ledger |
| `tests/test_error_attribution.py` | 18 tests, green |
| `reports/error_attribution_queue.json` | **frozen, 1.3 MB, 104 cases + 15 gold photos** |
| `reports/error_attribution_verdicts.jsonl` | **does not exist yet — you create it** |
| `reports/error_attribution.{md,json}` | placeholder from an empty ledger; regenerate at the end |
| `docs/RESULT_error_attribution_20260831.md` | **not written — you write it** |

Regenerate the queue only if an input changed:
`.venv\Scripts\python.exe scripts\build_error_attribution_queue.py`
It is deterministic apart from `generated_at`, and case ids are stable
(`case_id` == the original `rc_` card id).

## 3. What the extraction established (facts, not assumptions)

Verified across all 28 frozen v5 listings (18 canary run_1 + 10 production):

- **891/891 photos carry Pass 2a prose.** Nothing is lost to missing 2a.
- **2,919/2,919 evidence refs join to a 2b bullet by exact string.** Zero
  normalized, zero failures — including rejected (dirB) conditions. The join
  risk this design worried about does not exist in this corpus.
- **2c is the dominant filter**: 11,114 2b bullets → 4,627 surviving issues,
  **6,487 dropped at 2c** (58%), with no rationale recorded anywhere.
- **2e removed nothing at all** across the entire population. A `2e` attribution
  is therefore structurally impossible here — if you find yourself reaching for
  it, re-read the record. Say this in the RESULT doc; it is a finding, not an
  absence of data.
- All 83 evidence photos for the 55 attributable cases exist on disk, as do all
  15 gold photos.

## 4. The cohort

104 cases. **55 are attributable** (you review these); 49 are counted-only.

| Lane | n | What it is |
|---|---|---|
| `miss_label` | 20 | v1.1 `dirB_recovery` (19) + `dirB_wording_recovery` (1): human says the condition is real, the pipeline rejected it |
| `miss_v1only` | 3 | `terra_flip` cards, human said supported, condition excluded. Weaker basis (see §6) |
| `halluc_label` | 9 | v1.1 `hard_false_billed`; identical to the `claim == absent` set (zero divergence) |
| `halluc_v1only` | 3 | accepted + `terra_claim_unsupported`, never re-adjudicated. Weaker basis |
| `appendix_misnamed` | 8 | v1.1 `misnamed_billed` — real problem, wrong name. Feeds the future remap lane |
| `appendix_trivial` | 6 | v1.1 `trivial_billed` — true but not worth billing |
| `appendix_inconclusive` | 6 | 2 v1.1 `excluded` + 4 v1-only inconclusive |
| `counted_correct_rejection` | 31 | human agreed with Terra's rejection — no error to attribute |
| `counted_agreement` | 7 | human agreed with an accepted claim — no error |
| `counted_orphan` | 11 | orphan verdicts, mostly `excluded_low_res` thumbnails; untraceable by construction |

Denominators for the report: **misses 23, hallucinations 12, appendix 20**
(plus whatever the gold lane yields).

Appendix cases get full attribution too — Steven's call, because the misnamed
ones tell you whether 2a or 2c/2d chose the wrong name.

**Why `miss_v1only` exists:** these three are `terra_flip` cards where the human
called the claim supported, Terra called it unsupported, and the condition was
excluded. The original plan filed them under correct rejections, which is
backwards — they are real conditions the pipeline lost. Three cards:
`rc_e2ad788f0845`, `rc_a22a241b3bf0`, `rc_f227f639eb82`.

## 5. The rubric — apply it verbatim

Per case: open the evidence photo(s), confirm the case really is what its lane
says (strict miss / strict hallucination / appendix / not actually an error),
then attribute.

**Miss** (a real condition the pipeline failed to bill):
- `pass_2a` when **no** evidence photo's 2a prose contains the real condition.
- `downstream` otherwise, recording the first stage that lost it:
  - no matching 2b bullet → `2b`
  - bullet present but dropped from the surviving issues (`p2c_dropped`) → `2c`
  - resolved to the wrong catalog item (`p2d.resolved_item_id`) → `2d`
  - removed at 2e → `2e` *(structurally impossible in this corpus, see §3)*
  - issue never became a condition (`projected_condition_id` null) → `condition_projection`
  - condition reached Terra and Terra rejected it → `terra`

**Hallucination** (a claim billed that the photo does not support):
- `pass_2a` when the unsupported material claim **already appears** in the 2a prose.
- `downstream` when 2a contains only a true or weaker observation and a later
  stage introduced the false meaning — same stage ladder.

`unclear` is a real answer. Use it rather than guessing; the report counts it.

**Worked example** (validated end to end this session, deliberately *not*
recorded as a verdict — review it yourself):
`rc_128caa6212b8`, canary `redfin_11000447`, photo_003. Claim: "vinyl or
linoleum flooring torn, worn through, or lifting". Terra rejected it: "the
visible flooring is hard plank material". 2a said: "The dark plank flooring is
heavily worn/scuffed and appears to have lifting/open seams". The photo shows
worn plank flooring with visible seams. 2a read the photo correctly and
material-neutrally; **2d** resolved it to a material-specific catalog item, and
Terra then refuted the material rather than the condition. That reads as
`downstream` / `2d`, high confidence. Note the human's own note reaches the same
place from the other direction ("we might want to change the catalog to 'hard
flooring'").

## 6. Truth basis — do not pool

- `basis: "v1_1"` — the adjudicated cohort. Two axes (claim: exact/misnamed/
  absent/inconclusive; work: warranted/trivial/none/inconclusive) plus the slug.
  This is the authority.
- `basis: "v1_only"` — one collapsed v1 verdict, never re-adjudicated. Weaker.
  **The report refuses `confidence: "high"` on these** and reports them on their
  own row. 6 attributable cases (3 miss + 3 hallucination) plus 4 inconclusive.

`human_truth.retag_answer` is carried verbatim for context. It is **validation
only, never a translation** — the 46 retag cards were re-asked under v1.1 and
disagreement between the two is itself a recorded finding. Do not derive axes
from it.

v1 fields (`v1_verdict`, `v1_class`) ride along for side-by-side reading only.
Never pool v1 and v1.1: they are the same cards relabelled, not additive.

## 7. Reading a case record

```
case_id, lane, attribute            what this is, and whether you review it
source_ids                          rc_ card id, adj_ card id, gold ids
run_ref                             source/property/run_id + artifact path & sha256
human_truth                         basis, axes, slug, note, retag answer, v1 fields
v5_claim                            condition_id, catalog item, claim_text as Terra saw it,
                                    terra_verdict + terra_rationale, disposition, accepted
lineage.per_photo[<key>]            image_path (+image_exists), p2a_prose (+sha),
                                    p2b_bullets, p2c_surviving, p2c_dropped, p2e
lineage.per_issue[]                 issue_id, observation, p2b_join{method,bullet_index},
                                    p2c_present, p2d{resolved_item_id,resolution_path,...},
                                    p2e_status, projected_condition_id, flat_lane
context                             strata, direction, second_opinion,
                                    in_factorized_disagreements
mechanical_hints                    deterministic only; explicitly NOT a judgment
status                              pending | untraceable
```

`lineage.per_photo` covers **every** evidence photo. For a miss, `pass_2a`
requires the condition to be absent from *all* of them. 18 of 55 cases are
multi-photo (up to 6).

`mechanical_hints` is plumbing, not opinion. Never let it stand in for reading
the prose and the photo.

`context.in_factorized_disagreements` marks cases the factorized verifier
flagged. That is **model** output and was never a case selector — treat it as a
column, not evidence.

## 8. Recording verdicts

Append one JSON object per line to `reports/error_attribution_verdicts.jsonl`.
Latest-wins per `case_id`; an `attribution: null` line undoes a case.

```json
{"case_id": "rc_128caa6212b8", "ts": "2026-09-01T12:00:00Z", "reviewer": "claude_session",
 "attribution": "downstream", "first_responsible_stage": "2d",
 "p2a_excerpt": "The dark plank flooring is heavily worn/scuffed and appears to have lifting/open seams",
 "confidence": "high",
 "rationale": "2a read the floor correctly and material-neutrally; 2d chose a vinyl-specific item and Terra refuted the material, not the condition."}
```

Rules the report enforces (it fails the run, it does not warn):
- `attribution` ∈ `pass_2a | downstream | unclear | untraceable | excluded`
- `first_responsible_stage` present **iff** `attribution == "downstream"`
- non-empty `rationale` on every verdict
- `confidence: "high"` forbidden when `basis == "v1_only"`
- if the case's join methods include `none`, the rationale must mention the join
  (no case in the current queue trips this)

`p2a_excerpt` is your chosen sentence. It is **inference, not provenance** —
2a has no observation ids, so the excerpt is your reading of a blob whose sha is
pinned separately. Keep the two distinct.

## 9. Suggested method: an ultracode workflow

~55 photo-heavy reviews do not fit one context well, and the headline claims
deserve adversarial checking. Steven approved running this as a workflow.

1. **Fan out** one agent per attributable case. Give each: its full case record,
   the evidence photo path(s), and §5's rubric **verbatim**. Schema-constrain
   the output to the §8 verdict shape so nothing needs parsing.
2. **Adversarially verify** every case attributed `pass_2a` or `unclear` — a
   skeptic agent per case, prompted to *refute* the attribution from the same
   evidence. `pass_2a` is the headline claim; it should be the hardest to earn.
3. **Gold lane** (see §10) — one agent per gold photo, 15 agents.
4. **Consistency sweep**, inline, by the orchestrating session: read the whole
   assembled ledger and look for rubric drift between agents (the real risk of
   fanning out), then reconcile and report.

Rough budget 1.5–2.5M tokens. Keep the rubric text identical across every
prompt; drift between agents is the failure mode to watch.

## 10. The gold lane

`reports/error_attribution_queue.json → gold_photos[]`: 15 photos, **112 frozen
gold findings**, 64 v5 conditions touching them (52 accepted). Each record has
the photo path, the full per-photo lineage, the gold findings, and every
condition on that photo with its verdict and disposition.

**No matching has been done.** It needs the photo in hand, so it belongs in the
review workflow, not in extraction. Per photo, decide for each gold finding:

- **matched** — a v5 condition covers it. No error. Record the pairing.
- **miss candidate** — the v5 catalog *could* have carried it but nothing did.
  Becomes a `miss_gold` case: attribute it with the forward walk
  (2a → 2b → 2c → 2d → projection) using that photo's lineage.
- **out of catalog** — nothing in the v5 catalog could bill this. **Not a miss.**
  Appendix, counted separately.

That third bucket is load-bearing. Gold is *photo-observation* truth, not
billable-condition truth — its own note keeps "technically-true conditions that
are normal-for-context" (and flags `redfin_10806500` photo_003/007 basement
shots specifically). Treating every unmatched gold finding as a miss would
invent ~80 bogus misses and wreck the headline.

Then the reverse direction: an **accepted** condition on a gold photo that maps
to no gold finding is a `halluc_gold_extra` case (condition-anchored walk).

Record the mapping in the verdict's `gold_match` field so it is re-derivable and
disputable — the precedent is `NOTE_THEMES` in `scripts/review_analysis.py`.
Case ids: `ga_<property>_<photo>_<gold_id>` for misses,
`gx_<property>_<condition_id>` for extras.

## 11. Finishing

```bash
.venv\Scripts\python.exe -m pytest tests/test_error_attribution.py -q
```

```bash
.venv\Scripts\python.exe scripts\error_attribution_report.py
```

Use `--allow-incomplete` mid-review; the final run must pass **without** it.
Then write `docs/RESULT_error_attribution_20260831.md` citing
`reports/error_attribution.json`, and state these limitations:

- 2c keeps no rationale for a drop, so 2c attribution is presence/absence only.
  A meaning distorted at 2c rather than dropped may land `unclear`.
- Sentence-level 2a excerpts are interpretation; provenance stops at the photo blob.
- 2e removed nothing in this corpus (§3) — that attribution was unavailable, not
  merely unused.
- Results describe **the reviewed discrepancy cohort**, not system-wide miss or
  hallucination rates. The `not_shown` caveats in `reports/review_analysis.json`
  still apply.
- Human reviewer self-consistency was 80% on the v1.1 repeat arm and the Terra
  replica floor is 6.4%. An attribution split inside that noise is not a finding.

## 12. Where things live

- Loaders (reuse, do not rewrite): `tools/review_cards.py` — `v5_result`,
  `iter_runs`, `load_canary`, `index_result`, `latest_verdicts`, `claim_texts`,
  `CANARY_ROOT` / `PROD_ROOT`.
- Photo paths: `tools/pass_2f_artifact_inputs.photo_key_to_path`. Never the
  stored absolute paths in `resolved_items` — those drift.
- Label vocabulary: `tools/label_schema.py`.
- Canary root: `artifacts_canary/renovation_session9_20260818/run_1/candidate/`.
  Production: `C:/Users/Steven/IntelliJProjects/renointel-prod/artifacts/`
  (only 10 runs carry a complete v5 envelope; the rest of the ~1,266-run corpus
  is v4).
- `condition_id` is **run-scoped** — zero overlap across runs. Never substitute
  a run for a missing one; the extractor marks that `untraceable` and so should you.
  The cross-run key, if you ever need one, is
  `(property_key, catalog_item_id, estimate_unit_id)`.
