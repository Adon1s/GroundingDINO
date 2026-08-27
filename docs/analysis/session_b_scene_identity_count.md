# Session B — scene-identity mismatch count (QP9 amendment)

Measured 2026-08-27 by `scripts/count_scene_identity_mismatches.py`
(one-off, read-only; NOT a pipeline stage). Question: how often does a
package candidate's evidence carry photos whose scene assignment conflicts
with the package room — the P05 "indoor photo in an exterior package" mode,
plus the utility-room bathroom surrogate variant.

## Method

Per non-display v5 package candidate: children → conditions →
representative photo keys (what Terra saw), each photo's normalized scene
group (`photos[key].scene.group`) checked against the candidate room
(kitchen→kitchen, bathroom→bathroom, bedroom→bedroom, living→living_areas,
exterior→exterior). `other`/unknown groups never count as conflicts.
Corpus: frozen canary run_1 (18 listings) + post-cutover production runs
≥ 20260821_230000, excluding the two thumbnail-based listings
(redfin_10965375, redfin_10922002). Scene extraction sanity-checked on
redfin_10803207 (real group distribution, not an all-`other` collapse).

## Result: zero

| corpus | listings | candidates checked | candidates with ≥1 mismatch |
|---|---|---|---|
| canary run_1 | 18 | 119 | **0** |
| production (post-cutover) | 8 | 40 | **0** |

No interior-in-exterior, exterior-in-interior, utility-in-bathroom, or
cross-room photo appears in any candidate's evidence in either corpus.

## Reading

The volume does **not** warrant a scene-identity rule (the QP9 amendment's
own bar); no pipeline stage is proposed. Caveat, stated honestly: this
measures scene-ASSIGNMENT conflicts. A photo whose scene assignment is
itself wrong in the same direction as the package room (e.g. an indoor
photo misclassified as `exterior` inside an exterior package) is invisible
to this check — that failure mode, if it exists, lives in scene
classification quality, not in package evidence wiring, and the P05 review
finding may have been exactly that or a pre-cutover run. Nothing further
this session.
