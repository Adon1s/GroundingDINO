# Handoff — manual output-quality review (UI + artifact consistency)

Created 2026-08-22 · branch `renovation_architecture_rework` (HEAD `df35f23`) · FE `renointel-prod` @ `kind_ontology_v2_compat` (`411312e6`)

## Why this exists

The Session 9 cutover is live and its 7-day observation window is running. The
mechanical gates (schema, publication, reconciliation, accepted-work,
comparator) all pass — but they only prove the artifact is *internally
consistent*. They cannot see whether what a person reads on the property page
is coherent, or whether a claim is true of the photo it points at. Steven found
the first such defect by eye within a day of the flip (§2). This handoff makes
that kind of looking a repeatable task and indexes the other review work
already queued (§5), so a new session can pick it up without re-deriving the
landscape.

**Scope of this task:** read real artifacts and real pages, log what does not
make sense, classify each as data / display / pre-existing / cutover-caused,
and recommend where it belongs. It is deliberately *not* an implementation
task; nothing here should change estimate behaviour during the observation
window.

## 1. Current state (correct as of 2026-08-22; two docs are stale)

- Production runs `RENOVATION_ARCHITECTURE_MODE=new` + `KIND_ONTOLOGY_VERSION=observation_kind_v2` (catalog 3.1). Verified live artifact: `redfin_10965375/20260821_233007_c5b989ee` — `verify_renovation_artifact.py --expect-mode new` → PASS 16/16.
- **7-day observation window is running**, day 1 = 2026-08-22 (first `new` artifact, `created_at 2026-08-22T04:31:48Z`). Rule: operational invariants only; roll back after three consecutive ontology/resolver/writer failures, or immediately on a published invalid/mixed artifact or a reproducible FE schema failure.
- The FE still renders `renovation_estimate_v4`; the root `renovation_estimate_v5` envelope is written but unread. The FE contract update is Steven's separate, later task.
- **Stale text to correct when next touched:** `docs/DECISION_renovation_architecture_session9_cutover_20260821.md:157` still says "production is staged, not flipped" (it was flipped 2026-08-21 ~23:20 CT); `docs/STATE_kind_ontology_program.md` still says Task 4A "sign-off not clean / open" (closed 2026-08-21 by written waiver — see the design-doc §9 log).

## 2. Finding #1 — "Flooring / Severe" card over a "Low" photo row (redfin_10965375)

**What was seen.** The *What we noticed* card reads **Flooring · Severe · "Top concerns: Worn or Stained Floori…" · 4 findings**, while the photo-23 detail for the same bucket reads **"Scratched or worn hard flooring finish · Low"**.

**Mechanism (verified by three independent passes; one corrected the first).**
The two words are different quantities about different findings:

| | card badge | photo row |
|---|---|---|
| field | `summary_v1.buckets[flooring].top_severity` = **5** | raw catalog `severity` = **1** |
| meaning | `max(display_severity)` over the bucket's blocks — `tools/property_summary_pass.py:344` | the item's unboosted base severity from the catalog |
| path | `lib/property/mobileNoticedItems.ts:71` → `DesktopNoticedList.tsx:157` → `SeverityBadge.tsx:3-9` (5 → "Severe") | `lib/analysis/enrichIssue.ts:65` → `lib/property/photoInspection.ts:169` → `InspectionIssueRow.tsx:28` (1 → "Low") |
| which finding | `worn_or_stained_flooring` on photo_007 + photo_022 | `hard_flooring_scratched_or_worn` on photo_023 |

`display_severity` is an escalation, not a severity: `base + evidence_boost + scope_boost + kind_boost + multi_scene_boost`, clamped 1–5 (`property_summary_pass.py:254-271`, constants `:33-41`). The 5 is a **base-1** item lifted by `scope_boost +2` (scope `replace`) and `multi_scene_boost +2` (it appears in `bedroom` and in the fallback `other` scene). The artifact contains **zero** severity *words* — every label is computed in the FE.

**The sharpest form of the problem:** click *View evidence* on that Severe card and you land on photo_007/photo_022, whose rows show `worn_or_stained_flooring` — catalog severity 1 — as **"Low"**. The exact item that produces the "Severe" badge renders as "Low" on its own evidence photo. Two incomparable numbers are painted on the same five-word scale with nothing on screen distinguishing them.

**Is it caused by our work? Mixed, but overwhelmingly pre-existing.**

- *Pre-existing and pervasive.* In a 250-artifact random sample of pre-cutover (catalog 2.1) artifacts: **91.0%** of buckets had `top_severity` greater than the max `base_severity` of their own blocks, and **14.6%** were `top_severity ≥ 5` while containing a `base_severity ≤ 1` member. For flooring specifically, **92 of 92** severity-5 cards contained a base ≤ 1 member. The named control artifact `redfin_11214304/20260803_224131_4dd6dbc1` (2026-08-03, 18 days pre-cutover) shows the identical "Flooring / Severe" card over base-1 members. The formula predates this whole program (commit `7936b77`, 2026-03-14).
- *What the cutover did change: exactly one notch on this card.* No severity integer changed in either catalog (`worn_or_stained_flooring` = 1 and `hard_flooring_scratched_or_worn` = 1 in both; the 3.1 split successors inherit their parent's severity). But catalog 2.1 classified `worn_or_stained_flooring` as kind `upgrade`, which drew `KIND_BOOST −1` and the `MAX_UPGRADE_SEVERITY = 4` cap; catalog 3.1 reclassifies it as `degradation`, which has `KIND_BOOST 0` and is not capped (only `modernization` is). Same evidence → **4 "Critical" before, 5 "Severe" now**.
- *The frontend contract is not stale here.* The version-aware adapter resolved catalog 3.1 correctly — proven by the fact that the photo-23 row renders at all under its 3.1-only name "Scratched or Worn Hard Flooring Finish"; under 2.1 that id does not exist, enrichment would yield a null name and `photoInspection.ts:320` would have dropped the row.

**Recommendation (mine).** Not a rollback trigger and not an observation-window failure — nothing is invalid, and the behaviour long predates the cutover. Log it and fold the fix into the planned FE contract work, where the natural remedies are: stop reusing one word-scale for both quantities (e.g. the card shows a concern *rank* or "4 findings, most severe: …" and only per-finding rows use severity words); or surface `severity_calc` so a boosted number is legible as boosted; or reconsider `multi_scene_boost` firing on the `other` fallback scene, which is what doubled this one. A backend-side alternative worth considering separately: whether a base-1 wear item should be able to reach 5 at all.

**Open judgment for Steven:** the photo shown for photo_023 looks like a *renovated* room with clean hardwood. Whether "visible wear, uneven sheen, possible staining" is true of that image is a P2-class question, not a display question — and `hard_flooring_scratched_or_worn` is already one of the disputed items in the P2 packet (direction A=1 / B=2; card C001).

## 3. How to run this review (repeatable)

1. Pick a recently analysed property; get its key from the worker log or `renointel-prod/artifacts/`.
2. `.venv\Scripts\python.exe scripts\verify_renovation_artifact.py --artifact <run dir> --expect-mode new --expect-terra-model gpt-5.6-terra --expect-sol-model gpt-5.6-sol` — mechanical gates first; if this fails, stop, it is an observation-window event.
3. Open `http://localhost:3000/property/<propertyKey>` and read the page as a user: the *What we noticed* cards, the per-photo findings, the estimate bands, Top Picks.
4. For anything that reads wrong, capture: the property + run id, the exact screen text, the artifact field(s) behind each number (use the paths in §2 as the worked example), and whether an older catalog-2.1 artifact shows the same shape.
5. Classify: **data** (the claim is untrue of the photo) / **display** (fields are fine, presentation misleads) / **contract** (FE reads a field that changed) / **pre-existing** vs **cutover-caused**.
6. Append to §4 below. Route: data → the P2/P1 packets; display + contract → the FE contract task; anything invalid/mixed → observation-window rollback rules.

Useful commands: `scripts/show_daily_token_spend.py --root C:/Users/Steven/IntelliJProjects/renointel-prod/artifacts` (ledger); the FE DB is SQLite at `renointel-prod/prisma/dev.db` (`Property` table maps `propertyKey` → address).

## 4. Findings log

| # | Date | Property | What reads wrong | Class | Verdict | Routed to |
|---|---|---|---|---|---|---|
| 1 | 2026-08-22 | redfin_10965375 | "Flooring / Severe" card over a "Low" photo row; the item driving "Severe" renders "Low" on its own evidence photo | display | pre-existing (91% of pre-cutover buckets); cutover moved this card 4→5 via the `upgrade`→`degradation` reclassification | FE contract task; base-1→5 escalation worth a separate backend look |
| 2 | 2026-08-26 | redfin_10965375 | Evidence photos are 390×260 thumbnails (32 of 34 photos); every model judgment on this listing except photo_001/photo_009 was made on a thumbnail | data (input) | scrape defect, **not** cutover-caused; ingest requested `bigphoto` URLs and the CDN returned small renditions; nothing validates the returned pixels | §7 below — future fix: minimum-resolution gate at ingest + `compute_photo_identity` |
| 3 | 2026-08-26 | 26-listing frozen cohort (212 cards; 18 canary + 8 production) | Manual review complete and analysed. Weighted broad-problem rate on billed conditions: canary 13.6% [6.8–26.6%], production 10.4% [5.1–32.5%] (hard-false 10.5% / 10.4%); error mass concentrates in `degradation`-kind claims (66% / 83%); v5 under-bills multi-bathroom work on 16 of 17 cards (unit gap +25); P1 packages split 11 warranted / 11 not; Terra replica flip rate 6.4% (16/250 pairs) | data | descriptive analysis, independently verified (6-agent recomputation, zero material findings): `reports/review_analysis.md` + `.json`, integrity OK, 11 orphaned verdicts excluded | P1/P3/P6(3) rows of the decision log (`DESIGN_renovation_architecture_decision_packets.md` §9); FE v5-adoption call |

## 5. Index of the other open review tasks

**2026-08-22 — the photo/judgment rows below are now served by one queue:**
`docs/DESIGN_review_method.md` (method, strata, phases) with
`scripts/build_review_queue.py` → `scripts/review_server.py` → `scripts/review_tally.py`.

**2026-08-26 — the queue is COMPLETE (212/212 reviewed) and analysed:**
`scripts/review_analysis.py` → `reports/review_analysis.md` / `.json` (frozen-cohort
descriptive analysis; finding #3 above). The P2/P1/P3/P6(3) rows below are done as
review work — the evidence is recorded in the decision log; the option calls remain open.
Note the correction there on the photo_023 anecdote: Terra rejected that claim
(v5 excluded it); Pass 2f confirmed it and v4 billed it — a direction-B case.

Photo/judgment work, roughly in the order the design doc recommends:

| Task | Asks | Time | Needs photos | Blocks |
|---|---|---|---|---|
| **P2** condition truth (68 cards) | on 85 disagreements where Terra and v4's 2f saw the same photo, who was right (`terra_correct` / `2f_correct` / `both_partly` / `cannot_tell`) | ~45 min | yes | P1's option choice, P6(3), any Terra rubric change |
| **P1** package warrant (P01–P13) | for 13 packages 2f rejected and Sol re-approved, `package_warranted` / `not_warranted` / `unsure` | 20–30 min | yes | FE adoption of v5; option (b)/(c) forces a fresh freeze + new canary |
| **P3** multi-bathroom (B01–B10) | how many distinct bathrooms you see; `bill: per_bathroom / once / unsure` | ~30 min | yes | same as P1; corpus effect −$58k/−$146k |
| **P6(3)** single-photo drivers | re-look at the 5 "v4 $0 → v5 priced" items once P2 is done (currently accepted *provisionally*) | ~5 min | yes | nothing |
| **Acceptance-rate audit** | are the 282 Terra rejections real hallucinations? (1746 supported / 282 unsupported / 65 cannot_assess; 71.4% accepted, 93.7% packages applied) | large | yes | nothing formally — but it *is* Steven's goal-1. **Do P2 first**; P2 is the sampled version of this question |
| **Terra batch-size quality** | do 20-condition / 12-image batches dilute verdict quality? No gate can see it | unknown | partly | nothing; tension: smaller batches re-send images and raise the binding token cost |
| **Price calibration** | real prices for the 42 split successors (`inherited_from_split_parent`) and the tier bands | large | no | any claim that the dollar figures are meaningful. Needs a renovator, not Steven alone |

Implementation-side, listed so they are not forgotten: **P4** Sol consistency rules + probe (offline-replayable, fail-safe); **P5** review-tooling gaps (do before the *next* canary — comparator keys v4 packages by `type|unit` and silently overwrites duplicates; package review rows carry no dollars; freeze should hash `vlm_client.py` / `analyzer_cli.py` / `failure_taxonomy.py`); **Task 4B** historical corpus migration (~1,264 artifacts still catalog 2.1) and **4C** legacy retirement (blocked on 4B); **Pass 2a prompt ablation** (STATE calls it "next task"; note its design predates Sessions 5–9 and a shipped 2a change invalidates the current canary deltas); the saturated `catalog-resolution-v2` benchmark. In the `rv-pass2a-bench` worktree: 5 pending package-benchmark review rows that block all verdicts, 3 auto-demoted gold targets, the Carroll `photo_005` scene decision, and the paused micro-observation questions (decide the scoring policy *before* touching the 2,650-row backlog).

Full detail: `docs/DESIGN_renovation_architecture_decision_packets.md` (P1–P6, §9 decision log), `docs/DECISION_renovation_architecture_session9_cutover_20260821.md` (§5 conditions, §6 not-conditions, §9 execution log), `docs/STATE_kind_ontology_program.md` (4B/4C, ablation).

## 6. Ground rules

- Do **not** re-open the 967 Session 9 review items; they were audited item-by-item and the comparator is reproducible byte-for-byte.
- Do **not** change estimate behaviour, prompts, or the catalog during the observation window. Logging a finding is the deliverable.
- Prices are provisional throughout — flag price-shaped findings as "needs a renovator" rather than treating them as defects.
- Do not `git checkout` another branch in `realtorvision-backend` while the window runs: `RV_ROOT` points at the working tree, so production runs whatever is checked out. Use a worktree.

## 7. Finding #2 — thumbnail evidence photos (future fix, do not change during the window)

> **Update, 2026-08-26 (later the same day):** scope is **two** production listings, not one —
> `redfin_10922002` (analysed 2026-08-25) has all 12 photos small (11× 640×480, 1× 325×389),
> same mechanism, files also dated 2026-08-02. Steven decided thumbnail evidence is
> **excluded from the manual review**: implemented in `tools/review_cards.py`
> (`MIN_EVIDENCE_PX = 500` short side; excluded cards leave the queue *and* the accepted
> denominators; 9 recorded verdicts on `redfin_10965375` orphaned). The pipeline fix is
> handed off: **`docs/HANDOFF_thumbnail_photo_ingest_fix.md`** — start after the window.

Raised 2026-08-26 from review card 119 (`exterior_siding_discoloration_fading @ exterior_primary`,
production `redfin_10965375`): the evidence photo was too small to judge.

### What the photo actually is

`photo_033.jpg` is **390×260, 28 KB**. Across that listing, **32 of 34 photos are 390×260**;
only `photo_001.jpg` (1280×853) and `photo_009.jpg` (984×656) are full size. Aspect ratio is
normal 3:2 — the defect is *resolution*, not proportion.

### Scope: one listing, not the corpus

| set | photos | ≤400 px wide |
|---|---|---|
| the 23 listings under review (18 canary + 5 production) | 765 | **32 (4.2 %) — all in `redfin_10965375`** |
| 60 randomly sampled corpus listings | 1,632 | **0** |
| distinct photos referenced by the review queue | 288 | 15 |

22 of the 23 reviewed listings are clean. So this is a per-listing ingest failure, not a
systemic downscale. It is **not** cutover-caused: the files were written 2026-08-02 21:36,
nineteen days before the flip.

### Mechanism (traced end to end)

1. `renointel-prod/scripts/python/redfin_single_scraper.py:_normalize_photo_url` rewrites any
   `midphoto` / `mbpaddedwide` / `islphoto` URL to the `bigphoto` variant
   (`/photo/235/bigphoto/410/MDBA2226410_8_0.jpg`). The batch scraper reuses this class, so the
   upgrade did run.
2. The saved scrape (`data/scraped/batches/cmsckusuc0awdv7ysom9obqeb/properties/redfin_10965375/scrape.json`)
   confirms **all 34 requested URLs were `bigphoto`** — the request side is correct.
3. Redfin's CDN nevertheless returned a small rendition for the non-primary photos of this MLS
   (`MDBA2226410_<n>_0.jpg`); the primary (`_0.jpg`) came back full size. The rewrite is a URL
   guess, and a wrong guess returns 200 with a thumbnail rather than 404.
4. **Nothing anywhere validates the returned pixels.** The batch scraper gates only on *count*
   (`downloaded_count <= 5` → skip); `lib/property/imageManifest.ts` hashes bytes; the backend
   pipeline never reads width/height. So the listing passed every gate and all analysis passes
   ran on thumbnails.

### Why it matters for the review

- All 35 v5 conditions on this listing rest on thumbnails; **8 condition cards in the queue have
  an entire evidence strip ≤400 px** (4 dirA, 3 dirB, 1 uniform — all production, all this listing).
- It touches the anecdote in `docs/HANDOFF_review_method_design.md` §5 / `DESIGN_review_method.md` §5:
  the photo_023 flooring dispute (Terra `unsupported` / 2f `confirmed`) was argued over a 390×260 image.
- Current signal from the verdicts so far is weak and not yet conclusive: on ≤400 px cards
  2 of 6 `terra_claim_unsupported` (33 %) vs 30 of 113 (27 %) on larger photos; `terra_evidence_inconclusive`
  is 0 in both groups.

### Proposed fix (after the observation window)

1. **Ingest gate (primary).** After download, read each image's dimensions; treat
   `min(width, height) < ~500 px` (or area below ~0.5 MP) as a failed fetch: retry the original
   pre-upgrade URL, else mark the photo — and, past a share threshold, the listing — unusable.
   This is the real fix: keep thumbnails out of the corpus.
2. **Backend backstop.** `tools/renovation_architecture/evidence.py:compute_photo_identity` already
   decodes every evidence photo and holds `rgb.width`/`rgb.height` — record them on `PhotoIdentity`
   and surface a per-condition `low_resolution_evidence` flag for zero extra I/O. Policy options:
   exclude such photos from evidence, or keep them but block `accepted_for_work` (route to
   `inspection`) when a condition's *only* support is low-resolution.
3. **Carry the number into the artifact** so review tooling and the FE can show "evidence photo is
   390×260" instead of leaving a reviewer to guess. `scripts/review_server.py` could label the strip.
4. **Backfill question.** How many of the ~1,266 corpus artifacts were analysed on thumbnails? The
   60-listing sample says ~0 %, so likely a handful; worth a full sweep before any re-analysis
   decision, and `redfin_10965375` should be re-scraped and re-analysed once the window closes.

Do not implement during the observation window: (1) is frontend/scraper work and (2)–(3) change
estimate behaviour.
