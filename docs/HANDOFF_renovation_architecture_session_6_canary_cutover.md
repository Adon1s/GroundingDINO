# Handoff — Session 6 canary findings (operational record)

Date: 2026-08-16. Branch: `renovation_architecture_rework`, HEAD `eff304f`.
Companion to `docs/HANDOFF_renovation_architecture_session_6.md` (code changes).

**Status: canary STOPPED BY DECISION after replica 1 (18/18 complete).**
Steven judged one replica sufficient to require adjustments — principally API
usage — before further benchmarking. Replica 2, the comparator run, the
written review, the `new` cutover smoke, and the rollback smoke were **not
executed**. The backend remains on `RENOVATION_ARCHITECTURE_MODE=current`.
No Session 6 release gate has been passed or waived; the unmet gates are
listed at the end.

This document reports findings only. Interpretation and decisions are
deliberately left to the follow-up session.

## Environment and freeze

| Item | Value |
|---|---|
| Output root | `artifacts_canary\renovation_session6_20260816_02` (root `…_01` aborted, deleted — see Events) |
| Freeze | `input_freeze.json`, freeze_sha256 `36fbba3e9a4d5441c423226b36b2986fd0baa6bbd7e156e52eb9e8240d498d52` |
| Manifest | `configs/kind_ontology_canary_manifest.json`, frozen, 18 properties (6 strata × 3) |
| Catalog | `tools/issue_catalog_kind_v2.json`, SHA-256 `5BE11ABB…49E1EC`, byte-identical throughout |
| Schema / policy | contracts v5 / envelope v5 / 11 policy versions unchanged |
| Images | 572, each SHA-256-pinned; frozen paths verified before every property run |
| Property metadata | stored corpus metadata for 16/18; `redfin_125970550`, `redfin_80925528` have none in any stored run |
| Terra ledger | shared via `RENOVATION_TERRA_USAGE_ROOT=<output root>` (one UTC-day ledger) |

Model routing (recorded in freeze): 2a/2b/2c → `gpt-5.6-terra` (low), 2f →
`gpt-5.6-terra` (medium), condition review → `gpt-5.6-terra` cap 8192, Sol →
`gpt-5.6-sol` cap 8192 (medium), 1a + 2d resolution → local
`unsloth/qwen3.6-27b@q6_k` @ `http://169.254.83.107:1234`, 2d retrieval →
local jina v5 embeddings @ `127.0.0.1:8081`. `OPENAI_MODEL` unset; legacy
pass routing untouched.

## Replica 1 result

18/18 properties completed; **every private envelope
(`analysis_debug.renovation_estimate_v5`) is schema-valid `state="complete"`**;
`renovation_estimate_v4` present and valid beside all 18; zero publication
failures; zero reconciliation-audit or arithmetic failures (each envelope
self-validated by the recompute-equality gate at build time and again at the
publication gate).

### Per-property table

| Property | Cond | Work | Ledger | v5 low–high | v4 low–high | Terra tok | Sol tok |
|---|---|---|---|---|---|---|---|
| redfin_10803207 | 31 | 22 | 18 | 32,725–90,868 | 15,632–44,529 | 37,748 | 4,992 |
| redfin_10806500 | 21 | 18 | 13 | 33,752–98,081 | 12,468–36,711 | 15,085 | 3,763 |
| redfin_10952874 | 81 | 60 | 48 | 46,083–146,536 | 39,349–126,687 | 44,322 | 11,843 |
| redfin_11000447 | 36 | 37 | 24 | 34,161–101,268 | 23,578–69,212 | 16,781 | 7,295 |
| redfin_11077450 | 63 | 49 | 40 | 50,124–187,310 | 16,952–82,725 | 55,229 | 7,058 |
| redfin_11079485 | 32 | 31 | 25 | 20,390–131,667 | 21,500–74,759 | 16,340 | 6,230 |
| redfin_11185681 | 75 | 56 | 43 | 71,437–233,102 | 57,284–193,986 | 82,754 | 13,906 |
| redfin_125779232 | 54 | 46 | 33 | 57,431–175,338 | 32,729–99,786 | 38,181 | 8,588 |
| redfin_125970550 | 97 | 84 | 62 | 50,509–191,566 | 50,121–162,277 | 55,458 | 12,750 |
| redfin_126224899 | 113 | 92 | 70 | 70,613–248,585 | 62,471–210,699 | 75,701 | 17,757 |
| redfin_126418713 | 56 | 39 | 37 | 36,395–126,260 | 29,270–91,784 | 77,872 | 6,244 |
| redfin_127468088 | 59 | 44 | 37 | 60,837–189,547 | 36,405–100,638 | 67,700 | 9,537 |
| redfin_166147710 | 89 | 67 | 57 | 36,027–156,662 | 34,481–130,470 | 78,529 | 6,388 |
| redfin_25809814 | 41 | 26 | 22 | 34,767–141,711 | 29,086–85,674 | 44,075 | 4,155 |
| redfin_80877597 | 42 | 30 | 28 | 30,072–90,309 | 28,857–85,503 | 20,014 | 6,025 |
| redfin_80925528 | 65 | 55 | 40 | 49,580–178,902 | 45,200–131,000 | 80,203 | 10,999 |
| redfin_80990371 | 78 | 68 | 56 | 67,095–240,549 | 46,750–151,914 | 58,393 | 12,387 |
| redfin_81000709 | 47 | 27 | 25 | 17,150–65,789 | 13,258–58,092 | 55,093 | 4,444 |

### v4 → v5 headline deltas (replica 1; no review has been written)

| Property | Low delta | High delta |
|---|---|---|
| redfin_11077450 | +195.7% | +126.4% |
| redfin_10806500 | +170.7% | +167.2% |
| redfin_10803207 | +109.3% | +104.1% |
| redfin_125779232 | +75.5% | +75.7% |
| redfin_127468088 | +67.1% | +88.3% |
| redfin_11000447 | +44.9% | +46.3% |
| redfin_80990371 | +43.5% | +58.3% |
| redfin_81000709 | +29.4% | +13.2% |
| redfin_11185681 | +24.7% | +20.2% |
| redfin_126418713 | +24.3% | +37.6% |
| redfin_25809814 | +19.5% | +65.4% |
| redfin_10952874 | +17.1% | +15.7% |
| redfin_126224899 | +13.0% | +18.0% |
| redfin_80925528 | +9.7% | +36.6% |
| redfin_166147710 | +4.5% | +20.1% |
| redfin_80877597 | +4.2% | +5.6% |
| redfin_125970550 | +0.8% | +18.0% |
| redfin_11079485 | −5.2% | +76.1% |

- Corpus: v5 799,148–2,794,050 vs v4 595,391–1,936,446 → **+34.2% low /
  +44.3% high**.
- Median per-property delta: +24.5% low / +41.9% high.
- **17 of 18 properties exceed the 15% review threshold** on at least one
  endpoint. One property (`redfin_11079485`) is lower on one endpoint.
- v5 totals split: standalone 75,148–769,317 vs packaged 724,000–2,024,733 —
  packages contribute 91% of corpus low and 72% of corpus high.
- Unverified observation, recorded as a lead only: applied packages price at
  `max(unfloored tier spec, Σ owned child allowances)`
  (`tools/renovation_architecture/reconciliation.py`, per the S5 handoff).
  With 771 accepted conditions feeding 678 active work items and 118 of 126
  candidates applied, child sums may be exceeding the tier floors that
  historically set package prices. **No per-property trace was performed**;
  the two largest deltas (`redfin_11077450`, `redfin_10806500`) are the
  natural trace targets.

### Funnel (corpus, 18 listings)

observations 2,270 → conditions 1,080 → condition_reviews 1,080 →
accepted_for_work 771 (71.4%) / excluded 201 / inspection 43 / withheld 4 /
no_action 61 → work_items_active 678 (+173 suppressed) → package_candidates
126 → decisions 126 → applied 118 / not_applied 8 → ledger 678 entries
(318 standalone / 360 absorbed, 53%).

## Token economics (replica 1, complete; includes the wasted partial attempt on redfin_11185681)

From run logs (per-pass tables) and artifact observability (which reconcile
exactly: "unattributed" 1,073,839 = architecture Terra 919,478 + Sol 154,361):

| Lane | Tokens | Per listing (÷18) | Where billed |
|---|---|---|---|
| 1a scene | 721,149 | 40,064 | local (free) |
| 2d resolution | 2,158,662 | 119,926 | local (free) |
| 2a | 974,926 | 54,162 | OpenAI |
| 2b | 602,304 | 33,461 | OpenAI |
| 2c | 804,708 | 44,706 | OpenAI |
| 2f | 677,470 | 37,637 | OpenAI |
| Architecture Terra (condition review) | 919,478 | 51,082 | OpenAI |
| Sol (package review) | 154,361 | 8,575 | OpenAI |
| **Total OpenAI-billed** | **4,133,247** | **~229,625** | |

- Upstream cloud passes (2a/2b/2c/2f) cost **3,059,408** — 2.85× the new
  architecture's own Terra+Sol spend.
- The Terra usage guard meters **only** the architecture's condition review;
  the ledger recorded 919,478 debited (36.8% of the 2.5M ceiling) while
  actual OpenAI service usage was 4.13M — matching Steven's dashboard
  observation (~1.6M mid-run when the ledger read 455,940).
- Additional same-day spend not in the table: the aborted `…_01` attempt
  (18 of 49 images on `redfin_126224899`, upstream passes only) and the
  discarded first pilot property.
- Terra condition review per listing: min 15,085 (`redfin_10806500`), max
  82,754 (`redfin_11185681`), mean 51,082 — above the plan's ~25,000
  reference. Sol per listing: min 3,763, max 17,757, mean 8,575.
- Projected listings/day under the 2.5M ceiling depends entirely on scope:
  **~48/day** if the ceiling covers architecture condition review only
  (2.5M ÷ 51,082); **~10–11/day** if it covers all Terra-service usage
  (2.5M ÷ 229,625). The plan calls it a "hard **service** budget" while its
  25k/listing reference matches the architecture-only reading. Unresolved.
- Sol has a separate 250,000/day budget (Steven), enforced by nothing in
  code. One replica used 154,361 — two same-day replicas would exceed it.
- Local passes consumed ~2.88M tokens on the local GPU (not billed).

## Observations for review (flagged during the run)

### Terra batch size (flagged by Steven)

Condition review batches by estimate unit. Measured mid-run over 97 calls /
12 properties: conditions per call max 20 / mean 7.0 (14 calls ≥ 12);
images per call max 12 / mean 2.9 (41 calls single-image). Heaviest:
`redfin_166147710/exterior_primary` 14 conditions + 12 images + 18,281 input
tokens; `…/basement_primary` 20 conditions + 11 images. The plan permits
unit batching "without obscuring per-condition verdicts"; whether these
batch sizes obscure them is untested — verdicts are structurally valid and
reconcile, so no gate detects quality dilution. Task chip filed
("Evaluate Terra condition-review batch size"). Note the tension: splitting
batches re-sends image context, raising the per-listing tokens that are
already the binding constraint.

### High acceptance/approval rates (context for the deltas)

71.4% of reviewed conditions accepted for work; 93.7% of package candidates
applied (118/126); 8 not applied; zero display-only. Whether these rates are
correct judgment or over-acceptance was not evaluated.

## Events log (chronology, for reproducibility)

1. Root `…_01`: first replica-1 attempt on `qwen3.8-27b` (configured host
   `100.102.92.1:1234` unreachable). Property 1 aborted: LM Studio
   speculative-decoding fault under concurrent vision requests
   (`speculative batch index 4 is not inside the current sub-batch [0, 4)`),
   ~6% of images failing → analyzer abort. Root deleted; zero artifacts.
   Per Steven: cause was `unsloth/qwen3.6-27b@q6_k` being loaded while
   `qwen3.8-27b` was resident (VRAM bloat). With one model loaded, 6-way
   concurrency probes pass (6/6) for whichever model is resident.
2. Root `…_02` created on the reference model `unsloth/qwen3.6-27b@q6_k`
   (new freeze). This removed the prior model deviation — only the LM Studio
   host differs from the reference config.
3. Embeddings sidecar died once with the known false-healthy Vulkan fault
   (`vk::Queue::submit: ErrorDeviceLost`; `/health` still 200). Caught by
   the real-POST preflight before any spend; fixed by restart (Steven).
4. The replica-1 coordinator process was killed once by an environment
   restart at 1/18 complete. Resume with the identical command skipped the
   completed property and continued; Terra checkpoints prevented re-spend.
5. Run paused deliberately at 12/18 (Steven), resumed for 1 listing
   (13/18), paused for a usage-reset window, then resumed.
6. `redfin_11185681` failed once with a connect timeout to
   `169.254.83.107:1234` (39 images not attempted); server was reachable
   again minutes later; `--only` retry completed it. Note the host is a
   link-local (169.254.x.x) address.
7. Replica 1 reached 18/18. Replica 2 not started (decision).

## Session 6 release gates — status

| Gate | Status |
|---|---|
| 18/18 schema-valid new artifacts | **PASS (replica 1)** |
| Zero publication failures | **PASS (replica 1)** |
| Zero reconciliation/arithmetic failures or double counts | **PASS (replica 1)** |
| Zero unexplained accepted-condition loss | comparator not run — **NOT EVALUATED** |
| Written review for material scope/package changes | **NOT DONE** (17/18 properties over threshold await review) |
| Headline deltas >15% reviewed/approved | **NOT DONE** |
| Run-to-run stability measured | **NOT DONE** (needs replica 2) |
| Terra usage measured + 2.5M ceiling respected | measured; ledger enforced its scope; **denominator ambiguity open** |
| Listings/day reported | reported as a range (~48 vs ~10–11) pending the denominator decision |
| Selector rollback + worker restart smoke-tested | **NOT DONE** (unit-level tests pass; live smokes not run) |

**Cutover was therefore not attempted. Backend: `current`. No v5 artifact
exists at any root key anywhere.**

## Inputs for the follow-up session (facts, no recommendations)

- One replica of frozen canary data exists and is reusable: 18 complete v5
  envelopes + paired v4 results under
  `artifacts_canary/renovation_session6_20260816_02/run_1/candidate`, with
  per-call Terra/Sol records, funnels, timings, and checkpoints.
- The freeze is date-independent; replica 2 can run later against the same
  freeze **only if code/config stay byte-identical** (the freeze hashes the
  relevant files). Any change ⇒ new root and two fresh replicas.
- `tools/compare_renovation_architecture_cutover.py --review-template` will
  generate the review file from two replicas; it fails closed until every
  scope/package/headline/stability item has a written, freeze-bound decision.
- `scripts/run_worker_smoke.py` (untracked, new) drives a fresh
  `analyzer_server.py` in a chosen mode end-to-end for the cutover/rollback
  smokes: readiness gating, one listing via stdin JSON (`modelOverrides` /
  `reasoningEfforts` / `propertyMetadata` keys), artifact verification left
  to the caller.
- Open questions the data poses but does not answer:
  1. What the 2.5M/day Terra ceiling denominates (architecture-only vs all
     service usage), and whether production routes 2a/2b/2c/2f to Terra.
  2. Whether the +34%/+44% corpus increase is surfaced-scope (intended) or
     floor-vs-sum pricing lift (policy artifact) — untraced.
  3. Whether 71%/94% acceptance/approval rates are correct judgment.
  4. Whether 12-image/20-condition Terra batches dilute verdict quality.
  5. Where API usage can be reduced (candidate levers visible in the data:
     upstream 2a/2b/2c/2f routing at 170k/listing, Terra batch composition,
     cached-input reuse — `cached_input_tokens` was 0 on observed calls).
  6. Whether the ~25k/listing plan reference should be revised versus the
     measured 51k condition-review mean.
