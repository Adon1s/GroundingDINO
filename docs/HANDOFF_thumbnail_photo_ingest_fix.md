# Handoff — thumbnail evidence photos: ingest validation + backend backstop

Written 2026-08-26 · backend `renovation_architecture_rework` (working tree `df35f23` + untracked review tooling) · frontend/scraper `C:\Users\Steven\IntelliJProjects\renointel-prod`

**Purpose.** Some listings' downloaded photos are CDN thumbnail renditions, and the whole
pipeline — scraper, manifest, analysis — accepted them silently, so every model judgment on
those listings was made on thumbnails. A session is to fix ingestion so this cannot recur,
add a backend backstop, and clean up the affected listings. **Do not start before the
observation window closes (7 calendar days from 2026-08-22, i.e. work begins 2026-08-29 at
the earliest): the ingest fix touches the production scraper and the backstop changes
estimate behaviour.**

## 1. Symptom and discovery

Found 2026-08-26 during the manual v5 review (card 119, `exterior_siding_discoloration_fading`
on `redfin_10965375`): the evidence photo `photo_033.jpg` is 390×260 / 28 KB. Steven's
call: photos like this must not be used as evidence. Full investigation write-up:
`docs/HANDOFF_output_quality_manual_review.md` §7.

## 2. Mechanism (traced end to end; verify before changing anything)

1. `renointel-prod/scripts/python/redfin_single_scraper.py:_normalize_photo_url` rewrites any
   `midphoto` / `mbpaddedwide` / `islphoto` photo URL to the `bigphoto` variant
   (e.g. `/photo/235/bigphoto/410/MDBA2226410_8_0.jpg`). The batch scraper
   (`redfin_batch_scraper.py`) imports `RedfinPropertyScraper`, so it uses the same rewrite.
2. The saved scrapes confirm the request side worked: **all requested URLs were `bigphoto`**
   (`data/scraped/batches/<batch>/properties/<key>/scrape.json`, key `photos`).
3. Redfin's CDN answered some of those `bigphoto` URLs with a **small rendition and HTTP 200**
   — not a 404. Observed renditions: 390×260 and 325×389 (`redfin_10965375`), 640×480
   (`redfin_10922002`). The rewrite is a URL guess; a wrong guess degrades silently.
4. **Nothing validates the returned pixels.** The batch scraper gates only on *count*
   (`downloaded_count <= 5` → skip, `redfin_batch_scraper.py:323`);
   `lib/property/imageManifest.ts` only hashes bytes; the backend pipeline never reads
   width/height. The listings passed every gate.

## 3. Measured scope (2026-08-26)

| set | result |
|---|---|
| 10 production new-mode listings | **2 affected**: `redfin_10965375` (32/34 photos small; 34/35 conditions thumbnail-only) and `redfin_10922002` (12/12 photos small — 11× 640×480 + 1× 325×389; 30/30 conditions) |
| 18 canary listings | 0 affected |
| 60 random corpus listings | 0 of 1,632 photos small |

Both affected listings' image files have mtime **2026-08-02** (separate scrape batches:
`cmsckusuc0awdv7ysom9obqeb`, `cmrpus3bk0000v7bggziln7qf`) — the defect predates the
2026-08-21 cutover and is unrelated to it. The 60-listing sample says the historical corpus
is mostly clean, but that is a sample, not a sweep (§5.4).

## 4. Already done (2026-08-26, review side only — no pipeline change)

Per Steven's decision, thumbnail evidence is excluded from the **manual review**:
`tools/review_cards.py` gains `photo_dims` / `all_low_res` / `MIN_EVIDENCE_PX = 500`
(short side); queue build drops any card whose every known-size photo is below it, removes
those conditions from the accepted denominators, and reports
`meta.<source>.low_res_excluded`. Photo dimensions travel on each card and the review page
labels/outlines low-res photos. 9 already-recorded verdicts on `redfin_10965375` became
orphans (the tally notes them). Tests: `tests/test_review_cards.py::test_low_res_evidence_excluded`.
This is triage for human review only — the artifacts themselves still contain estimates
computed from thumbnails.

## 5. The fix (in priority order)

1. **Ingest gate (the real fix; FE repo).** After download, decode each image and treat a
   short side below a threshold (~500 px; decide the number, see §6) as a **failed fetch**:
   retry the original pre-rewrite URL before the `bigphoto` upgrade; if still small, record
   the photo as unusable, and past a share threshold (e.g. >20% of photos, or fewer than 6
   usable) mark the listing failed the same way `downloaded_count <= 5` does today. Wire the
   result into `scrape.json` / the manifest so it is auditable. Add the check to whichever
   path writes `public/images/properties/<key>/` (the scraper's downloader).
2. **Backend backstop (estimate behaviour — post-window, replay-validated).**
   `tools/renovation_architecture/evidence.py:compute_photo_identity` already decodes every
   evidence photo (`Image.open` at line ~60) — record `width`/`height` on `PhotoIdentity`
   for free. Policy to decide: exclude low-res photos from evidence, or keep them but block
   `accepted_for_work` (route to `inspection`) when a condition's *only* support is low-res.
   Either alters dispositions ⇒ validate offline with
   `scripts/replay_renovation_architecture.py` before shipping; it does NOT require new
   provider calls, but it does shift v5 outputs, so treat it like any estimate change
   (fresh canary rules apply if combined with prompt/catalog work).
3. **Surface the numbers.** Carry photo width/height into the artifact (`photos[key].photo`
   or `evidence_facts`) so the FE and review tooling can display them instead of guessing.
4. **Corpus sweep + cleanup.** One pass over `renointel-prod/artifacts` image dirs
   (dimensions only, no JSON loads) to list every affected listing; then re-scrape and
   re-analyse `redfin_10965375` and `redfin_10922002` (their current artifacts are built on
   thumbnails end to end — 2a observations included, so offline recompute does not help;
   they need full re-analysis after re-scrape).

## 6. Open decisions for the fixing session (or Steven)

- **Threshold.** Review-side uses short side < 500 px, which also catches the 640×480
  rendition. For ingest, consider: known-good listing photos are ≥ ~650 px short side;
  observed defect renditions are ≤ 480. Anything in 500–640 is unobserved territory.
- **Retry strategy** when `bigphoto` returns small: original URL, then `mbphotoextra`/other
  variants, or accept the listing with fewer usable photos.
- **Backstop policy**: exclude vs route-to-inspection (see 5.2).

## 7. Constraints

- Observation window: no estimate-behaviour change and no scraper deploy before 2026-08-29.
- `RV_ROOT` is the live working tree — use a worktree for backend changes.
- Zero-provider-call rule applies to validation tooling; re-analysis of the two listings is
  normal production work (budget guard is ON; each listing ≈ 220k Terra + 2f Sol tokens).
- Do not touch the Session 9 builder or `reports/session9_*` outputs.

## 8. Verification checklist for the fix

1. Ingest: feed the two known-bad listings' URL lists through the new gate → both flagged;
   a known-good listing passes untouched.
2. Backstop: replay the 18 canary listings — zero disposition changes (no canary photo is
   low-res); construct a synthetic artifact with one low-res-only condition → policy applies.
3. Sweep: report count of affected corpus listings; expected ≈ the two known ones.
4. After re-scrape + re-analyse: `scripts/verify_renovation_artifact.py --expect-mode new`
   passes; review queue rebuild picks the new runs up automatically (newest run per property)
   and their cards return with full-size evidence.

## 9. Pointers

`docs/HANDOFF_output_quality_manual_review.md` §7 (full investigation, findings-log row 2) ·
`tools/review_cards.py` (`MIN_EVIDENCE_PX`, `photo_dims`, `all_low_res`) ·
`renointel-prod/scripts/python/redfin_single_scraper.py` (`_normalize_photo_url`) ·
`renointel-prod/scripts/python/redfin_batch_scraper.py` (count gate) ·
`renointel-prod/lib/property/imageManifest.ts` ·
`tools/renovation_architecture/evidence.py` (`compute_photo_identity`) ·
memory notes `thumbnail-evidence-photos-defect`, `review-program-tooling-built`.
