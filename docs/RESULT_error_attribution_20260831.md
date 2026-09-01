# Result: Pass 2a vs. downstream error attribution

Run 2026-08-31. Offline audit of frozen artifacts — no provider calls, no
pipeline reruns, no prompt or production changes. Numbers cited from
`reports/error_attribution.json` (regenerate with the command in §8).

Inputs: `reports/error_attribution_queue.json` (frozen, sha
`b58dcca7…40df`, unchanged by this run), `reports/error_attribution_verdicts.jsonl`
(101 lines), `reports/error_attribution_gold_cases.json` (40 materialised gold
cases). Method: `docs/PLAN_error_attribution_review_20260831.md`; background:
`docs/HANDOFF_error_attribution_review.md`.

---

## 1. The headline

**Misses and hallucinations have opposite origins.** This is the finding.

| cohort | judged | pass_2a | downstream | unclear | excluded |
|---|---:|---:|---:|---:|---:|
| **Misses** (63 cases) | 58 | 10 (**17.2%**) | 43 (**74.1%**) | 5 | 5 |
| **Hallucinations** (12 cases) | 12 | 10 (**83.3%**) | 2 (16.7%) | – | – |
| **Appendix** (20 cases) | 20 | 2 (10.0%) | 8 (40.0%) | 10 | – |

Sharper still on the human-adjudicated cohort alone, before the gold lane is
mixed in: **all 23 rc_ miss cases are downstream, none are Pass 2a** — 20 at
Terra, 3 at 2d. The VLM saw these conditions and said so; the pipeline lost
them afterwards.

The direction reverses for hallucinations: 10 of 12 originate in the 2a prose
itself. In the recurring shape, 2a converts a neutral description into a
billable judgement on its own — "basic trim" becomes a dated-finishes upgrade
opportunity, a just-installed vanity becomes "transitional/dated" — and every
later stage transmits that faithfully. Nothing downstream distorted anything;
the claim was already false when it left 2a.

So the two error classes need different fixes. Prompt work on 2a addresses
hallucinations and would barely touch misses. Misses are a Terra-and-2d
problem.

## 2. Where downstream misses actually die

`downstream_stages` over all 51 downstream attributions:

| stage | n | where |
|---|---:|---|
| terra | 21 | 19 miss_label, 1 miss_v1only, 1 gold |
| 2d | 20 | 6 appendix_misnamed, 8 gold, 3 miss lanes, 3 others |
| 2c | 5 | gold only |
| 2b | 4 | gold only |
| condition_projection | 3 | 2 gold, 1 appendix |
| **2e** | **0** | see §5 |

**Terra rejection is the single largest cause of a lost real condition** — 19
of the 20 adjudicated `miss_label` cases. The recurring pattern is Terra
refuting something adjacent to the claim rather than the claim itself: on
`rc_128caa6212b8` it rejected worn plank flooring because the material "is
hard plank material" rather than vinyl; on `rc_a22a241b3bf0` it answered "no
dated valance visible" against an item whose own text spans "valance **or
treatments**", leaving the observed dated blinds unruled-on.

**2d owns the naming problem.** The handoff asked whether the misnamed
appendix would implicate 2a or 2c/2d. The answer is 2d: 6 of 8
`appendix_misnamed` cases, with 2a reading the photo correctly and 2d
attaching a wrong-but-plausible catalog item — bare weathered wood decking
resolved to a "patio or porch surface wear / coating breakdown" item, a
material-neutral "wallcovering is dated" resolved to `dated_wallpaper_present`.
This is the evidence base for the remap lane.

**2b and 2c appear only in the gold lane, and that is structural, not a
finding.** Every rc_ case is anchored on a condition that exists, so by
construction its observation already survived 2b and 2c. Only the gold lane's
forward walk can see a bullet that never became a condition. Any read of "2c
drops 6,487 of 11,114 bullets" against this table has to account for that
asymmetry: this cohort cannot measure the 2c filter's cost.

## 3. The gold lane

15 photos, 112 frozen findings, on two canary properties only
(`redfin_10806500`, `redfin_11000447`).

| decision | n |
|---|---:|
| matched (an accepted condition covers it) | 44 |
| miss_candidate → became a `ga_` case | 40 |
| out_of_catalog (not billable — **not a miss**) | 27 |
| already_cased (an rc_ case owns the condition) | 1 |
| **gold findings total** | **112** |

**The out_of_catalog bucket is load-bearing, exactly as the handoff warned.**
27 findings are photo-true but not billable; a further 5 were reclassified to
`excluded` during adversarial verification (§4). Had every unmatched gold
finding been scored a miss, this report would have invented roughly 32 false
misses and the headline would have flipped toward "the pipeline misses
everything". Gold is photo-observation truth, not billable-condition truth.

**Gold is also not exhaustive.** 15 accepted conditions on gold photos are
supported by the photo but have no gold finding (`gold_incomplete`). The
reverse pass produced **zero** hallucination candidates: no `gx_` case exists,
because every accepted condition on a gold photo was either matched, already
cased, or genuinely supported. On this cohort, accepted conditions on
gold-annotated photos are not fabricated.

## 4. Adversarial verification

Every `pass_2a`, `unclear`, and `excluded` verdict was attacked by an
independent skeptic prompted to refute it (D3 widened this beyond the
handoff's `pass_2a`/`unclear`).

| | challenged | verdict stood | disputed | overturned by orchestrator |
|---|---:|---:|---:|---:|
| cases | 23 | 21 | 2 | 1 |
| gold protos | 21 | 16 | 5 | 5 |

All six overturns were adjudicated against the frozen record, not on rhetoric:

- **`rc_45b9d2f6923c`** pass_2a → **downstream/2d**. The case's only issue is
  `fe4392d2f287d083`, observation *"Wallcovering is dated."* — true and
  material-neutral. 2a's actual falsehood ("bamboo/raised-pattern wallpaper")
  lives in a *different* issue not in this case's chain, and its most
  material-specific sentence was dropped at 2c. The word "wallpaper" enters
  this condition only at 2d. Verified field-by-field before overturning.
- **Two electrical gold findings → `excluded`** (`10806500/photo_007#g3`,
  `11000447/photo_007#g5`). The electrical trade has been product-quarantined
  since 2026-07-12 with fail-closed product lanes, and wiring observations were
  parked as true-but-unactionable on 2026-08-11. v5 could not bill these by
  design, so they are out of catalog, not misses.
- **Two non-billable gold findings → `excluded`** (`11000447/photo_005#g8`
  loose panels leaning at the frame edge — movable contents, not an installed
  element; `11000447/photo_008#g10` missing base trim). The second is recorded
  at **low** confidence and is the weakest of the five: unlike the electrical
  pair there is no policy fact behind it, only an absence-of-evidence argument
  from five comparable observations 2c dropped with no catalog home. A
  reviewer who finds a base-trim item in the catalog should restore it as a
  `pass_2a` miss.
- **`11000447/photo_008#g2`** unclear → **pass_2a**. Stays a real miss (an open
  cut-through in a bathroom wall is a genuine defect), but 2a asserted
  "Unfinished electrical opening" where the photo shows no box, device, or
  conduit — gold describes it neutrally. The false electrical framing is born
  at 2a; 2d then correctly resolved a wrong input.

One dispute was **upheld against the skeptic**: `rc_4c62bf04b442` stays
`unclear`. Its skeptic's third pillar — that 2d "returned catalog_item_id null
three times, so declining was representable" — misreads the record:
`p2c_surviving[].catalogItemId` is `None` for *every* surviving issue,
including the two that 2d demonstrably did resolve. The field carries no
information about 2d's choices.

## 5. Corrections to the handoff

**The 2e claim was imprecise; its conclusion survives.** Handoff §3 states 2e
"removed nothing at all" and infers a 2e attribution is structurally
impossible. Measured:

| signal | value |
|---|---|
| `p2e.removed_count` | 0 on all 143 photos ✓ |
| `p2e_status` per issue | kept 102, **absent 27**, unknown 4 |
| photos with `suppressed_reason_counts` | **119 of 143** |
| reasons | `tier_optional_suppressed` 225, `drop_if_generic` 39 |
| **issues that still projected a condition** | **133 of 133** |

2e *does* suppress, on 18 attributable cases. It simply never costs a
condition, because `condition_projection` emits one regardless. So a 2e
attribution was unavailable — for a stronger reason than the handoff gave, and
0 of 95 verdicts used it. This matters operationally: a reviewer told "2e
removed nothing" who then reads `p2e_status: "absent"` has been handed an
apparent contradiction. The prompt was corrected to state the real mechanics
before the main run.

**The worked example's reference answer is contestable.** The handoff's §5
example (`rc_128caa6212b8`) is given as `downstream`/`2d`. Two independent
runs returned `downstream`/**`terra`** at high confidence. The case's frozen
`human_truth` carries `claim: "exact"`, `work: "warranted"`, slug
`exact_and_warranted` — the human adjudicated the billed text ("vinyl or
linoleum flooring torn, worn through, or lifting") as *exact*, which endorses
2d's resolution rather than faulting it; the note's catalog complaint ("we
might want to change the catalog to 'hard flooring'") is forward-looking. On
that record Terra's rejection is the first stage that lost the condition. The
headline is unaffected — both readings are `downstream` — but the
`downstream_stages` split moves by one. Recorded as a disagreement, not a
passed gate.

**A gap in the stage ladder.** Routing/disposition happens *after* Terra and
has no rung. A condition that reached Terra, was *supported*, and still went
unbilled (`no_action` / tier-optional routing) fits nowhere; reviewers were
told to answer `unclear` and say so. Surfaced unprompted by a gold agent.

## 6. What `unclear` is telling us

10 of 55 case verdicts (18.2%) are `unclear` — below the 25–30% level that
would make the question itself suspect, and well below the factorized
verifier's 28.9% (`docs/RESULT_factorized_replay_20260831.md`). But the
distribution is not random:

- **`appendix_trivial` is 6/6 unclear.** "True but not worth billing" does not
  decompose into a stage. The condition is real and every stage handled it
  correctly; what went wrong is a billing-threshold judgement no rung
  represents. This lane should not be re-run against this ladder — the answer
  will be `unclear` again.
- **`appendix_inconclusive` is 4/6 unclear**, which is the honest result when
  the human could not settle the claim either.
- The adjudicated miss and hallucination lanes carry **zero** unclear verdicts.
  Where the truth basis is firm, the ladder discriminates.

## 7. Limitations

- **2c keeps no rationale for a drop**, so 2c attribution is presence/absence
  only. A meaning distorted at 2c rather than dropped can land `unclear`.
- **Sentence-level 2a excerpts are interpretation.** 2a has no observation ids;
  provenance stops at the photo blob, whose sha is pinned separately. Every
  excerpt was machine-checked to be a verbatim substring of some `p2a_prose`,
  which proves quotation, not that it is the *causal* sentence.
- **2e was unavailable, not merely unused** (§5).
- **Results describe the reviewed discrepancy cohort**, not system-wide miss or
  hallucination rates. The `not_shown` caveats in `reports/review_analysis.json`
  still apply. Canary and production are not pooled into a rate.
- **Noise floor.** Human reviewer self-consistency was 80% on the v1.1 repeat
  arm and the Terra replica floor is 6.4%. An attribution split inside that
  noise is not a finding. The two headline splits (17% vs 74% for misses, 83%
  vs 17% for hallucinations) are far outside it; the `2d`-vs-`terra` boundary
  within downstream is not — a pilot re-run flipped one case across it, and the
  worked-example disagreement sits on the same boundary.
- **Gold covers two canary properties only**, both from the same run. Its
  stage distribution should not be read as representative.
- **Reviewer is a model, not a human.** Cross-run agreement on the pilot was
  4/4 on attribution and 3/4 on stage. This audits the record, and its
  judgements are recorded per case so they can be disputed.
- **Deviations from the handoff** (all approved, see plan §2): the report gained
  a `--gold-cases` merge path; a pilot gate ran first; skeptics covered
  `excluded` as well; `2e` and `untraceable` were removed from the agent-facing
  enums so structurally-impossible answers were unrepresentable.

## 8. Reproducing

```bash
.venv\Scripts\python.exe -m pytest tests/test_error_attribution.py -q
```

```bash
.venv\Scripts\python.exe scripts\error_attribution_report.py --gold-cases reports/error_attribution_gold_cases.json
```

The second command must exit 0 **without** `--allow-incomplete`: 144 cases, 95
attributable, 95 verdicts, 0 pending. Every verdict carries a rationale, and
every overturn carries both views.

## 9. What this suggests next

Not decisions — the option calls are Steven's.

1. **Terra is the miss engine** (19 of 20 adjudicated `miss_label` cases). The
   failure mode is answering a neighbouring question: refuting a material, a
   valance, a coating, when the claim was about a condition. That is a Terra
   prompt or claim-text problem, and it is the highest-yield target here.
2. **The catalog vocabulary drives both Terra misses and 2d misnames.** The
   `vinyl_linoleum_torn_or_lifted` case is one error surfacing twice: 2d must
   pick a material-specific item, then Terra refutes the material. Steven's own
   note proposed the fix — a "hard flooring" style item that does not force a
   material commitment. §2's 2d column is the candidate list.
3. **Hallucinations are a 2a prompt problem**, specifically 2a volunteering
   renovation judgements ("upgrade opportunity", "dated") about finishes it has
   just described as clean and undamaged.
4. **Retire `appendix_trivial` from stage attribution** (§6) — the ladder
   cannot answer it.
5. **The gold lane wants more properties** before its stage distribution is
   worth acting on.
