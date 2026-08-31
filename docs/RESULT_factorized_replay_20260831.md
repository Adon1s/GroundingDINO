# Result — factorized verifier canary replay (2026-08-31)

Run of `docs/HANDOFF_factorized_verifier_replay.md`. Arm root
`artifacts_canary/factorized_v1_20260831`, scorecard
`reports/factorized_review_scorecard.md` + `.json`. Code at `6730ebd`
(thresholds pre-committed there, before any result existed).

## 0. Verdict

**Decisive result: FAIL. Every gate failed.** Do not wire any of this into
production policy. The hypothesis — that splitting Terra's verdict into three
questions would surface the misnamed and trivial error classes without
suppressing real work — is **not supported by this evidence**.

The run itself was clean, so the failure is a finding about the *questions*,
not about the harness.

| gate | result | threshold | status |
|---|---|---|---|
| G1 false suppression | **10**/57 | ≤ 5 | FAIL (decisive) |
| G2 recovery | **6**/16 | ≥ 11 | FAIL (decisive) |
| G3a misnamed catch | **2**/7 | ≥ 4 | FAIL (exploratory) |
| G3b trivial catch | **0**/5 | ≥ 3 | FAIL (exploratory) |
| G3c absent catch | **1**/3 | ≥ 2 | FAIL (directional) |
| G4 degeneracy & health | 3 of 5 checks | all | FAIL (decisive) |

## 1. The machinery worked perfectly

Nothing here is an execution artifact. 148/148 units, 1,047/1,047 conditions,
**0** parse failures, **0** fingerprint refusals, **0** prompt-adherence
violations (every misnamed answer carried a description; every `visible = no`
answer set the other factors to `unclear` as instructed). 1,058,584 tokens
(901,651 in / 156,933 out), 1.15× stored Terra, against a 1.0–1.4M forecast.

The model held a harder format than Terra's without a single violation across
1,047 judgments. Format discipline is not the problem.

## 2. Why it failed, axis by axis

### `visible` reproduces Terra instead of improving on it

| comparison | divergence |
|---|---|
| factorized vs stored run_1 | **39/702 (5.6%)** |
| stored run_1 vs run_2 (noise floor) | **42/702 (6.0%)** |

Measured on the 702 conditions the replicas share. The factorized verifier
differs from Terra by *less than Terra differs from itself between replicas* —
it is **inside the noise floor**. Asking "is it visible" as a separate question
did not change what the model perceives; it only changed how the answer is
reported.

That is the direct, quantitative cause of **G2**: of the 16 conditions Terra
wrongly rejected, **9 came back `visible = no`** — agreeing with Terra against
the humans. Only 6 were recovered. The factorization inherits Terra's blind
spot because it is the same perception.

### `material_enough_for_work` does not measure what the humans measured

On conditions the model could see:

| human label | model `yes` | model `no` |
|---|---|---|
| `warranted` (n=69) | 55 | **14** |
| `trivial` (n=5) | **5** | **0** |

It called every single human-labelled trivial condition *warranted*, while
calling 14 human-labelled warranted conditions *trivial*.

**This is not a calibration offset.** A stricter threshold would catch
everything a looser one catches, so the 5 trivial cards would have come back
trivial too. Disagreeing in both directions at once means the axis is tracking
a different construct than the human judgment, not the same construct at a
different cut point. Re-anchoring the wording will not fix a correlation this
shape.

### `claim_accurate_as_written` punts, and misses

28.9% `unclear` against a 10% ceiling, and only **2 of 7** labelled misnamed
conditions caught. On the 78 exactly-worded conditions it answered `unclear`
16 times. The one axis with a plausible mechanism is also the one that most
often declines to answer.

## 3. What G1's suppressions actually were

All **10** share one pattern — `visible=yes, accurate=yes, material=no` →
`exact_but_trivial`. **Zero** were false `absent`.

The items: `wall_scuffs_marks_or_dents` ×2, `baseboard_wear_scuffs` ×2,
`patio_or_porch_surface_wear` ×2, `exterior_siding_discoloration_fading`,
`exterior_door_paint_failure`, `dated_interior_trim`,
`floor_dirty_or_heavily_soiled`.

Every one is a cosmetic-minor item type, and the model's "not its own line
item" call on them is arguably defensible on its face. It is still a gate
failure: the gate was pre-committed against these labels, and it fired at
twice its allowance. Worth stating both ways rather than choosing the
flattering reading — but note §2 forecloses the charitable interpretation,
because a model applying a consistently stricter bar would also have caught
the 5 the humans called trivial, and it caught none.

## 4. The one asset produced

**50 misnamed candidates across the full sweep, each with a description of
what the model actually sees** — and adherence was perfect, so none are empty.
Qualitatively several look real:

- `masonry wall discoloration and patching` (claimed: wall scuffs/dents)
- `Swollen, peeling, deteriorated vanity cabinet surfaces.`
- `moderate tub discoloration with aged caulk and dark grout`
- `weathered fence with leaning sections`

That is the remap lane's raw input, and it cost nothing extra. It does not
rescue the gates (G3a is 2/7 against labels) but it is the part worth keeping.

## 5. What this does not show

- The labelled slice was sampled disagreements-first, so catch rates sit on a
  hard, non-representative population.
- No dirB **loss** population exists (`dirB_terra_correct` = 0), so G2 has no
  paired over-revival check.
- G3a/b/c are n = 7/5/3 and remain exploratory.

## 6. What the L2 rescore can and cannot change

The rescore is still owed and still free. It will make **G3a/b/c** decisive.

It will **not** rescue this result. G4's failing checks (28.9% and 16.8%
`unclear`) are label-free — more labels cannot move them. The `visible`-inside-
noise-floor finding is measured on 702 conditions, not on the labelled 88. G1
fired at 10/57 on an adequate population.

## 7. Recommendation

**Do not proceed to production wiring, and do not re-run this design.**
Specifically:

1. **Drop the `visible` axis as a source of improvement.** It is Terra's
   judgment restated. Whatever it is worth, it is not new information.
2. **Do not simply re-word the materiality rubric.** §2 shows the disagreement
   is not a threshold offset. If materiality is worth pursuing, it needs
   grounding in actual human trivial exemplars — which means the severity
   follow-up the roadmap already carries as a standing item, not a prompt tweak.
3. **The accuracy axis is the only survivor worth another look**, and it is
   underpowered (2/7) and evasive (28.9% unclear). If anything continues, it
   is a narrow "is this claim's wording right for what is visible" probe with
   `unclear` disallowed — not a three-factor verifier.
4. **Keep the 50 misnamed candidates** as input to any future remap work.

Terra remains unchanged and authoritative, which is exactly what shadow mode
was for: this cost 1.06M tokens and about a day, and it changed nothing that
users see.
