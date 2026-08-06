# Runbook — Kind Ontology v2 Canary (Task 4A)

Same-input comparison of the pinned legacy build vs the v2 candidate on 18
frozen properties. Nothing here touches production artifacts: all output goes
to `artifacts_canary/` (gitignored working data).

## Prerequisites (both sides)

- [ ] Manifest `configs/kind_ontology_canary_manifest.json` has `status: "frozen"`
      with Steven's approval recorded (`approved_by`, `approved_at`).
- [ ] Pinned legacy worktree exists: `git worktree add ../rv-legacy-a1972cf a1972cf`.
- [ ] LM Studio serving the Qwen model (Pass 2d runs local on both sides).
- [ ] Embeddings sidecar (:8081) healthy — probe with a **real POST**, not
      `/health` (it returns 200 while Vulkan is dead; restart the process if a
      real call fails).
- [ ] `OPENAI_API_KEY` set; model routing comes from
      `benchmarks/configs/kind_canary_model_map.json` (Terra on 1a/2a/2b/2c/2f,
      2d deliberately Qwen) and is applied **identically to both sides** so only
      code/ontology differs.
- [ ] `KIND_ONTOLOGY_VERSION` is NOT set in the shell (the driver sets it per side).

## Run

```bash
.venv/Scripts/python.exe scripts/run_kind_canary.py --side baseline --build-root ../rv-legacy-a1972cf --out artifacts_canary
```

```bash
.venv/Scripts/python.exe scripts/run_kind_canary.py --side candidate --build-root . --out artifacts_canary
```

The driver reads images from
`renointel-prod/public/images/properties/<key>/` and property metadata from the
stored baseline artifact (same-inputs discipline). Completed properties are
skipped on re-run; use `--only <property_key>` to retry one.

## Compare and gate

```bash
.venv/Scripts/python.exe tools/compare_kind_cutover.py --baseline artifacts_canary/baseline --candidate artifacts_canary/candidate --config configs/kind_ontology_cutover.json --report reports/kind_cutover_canary_$(date +%Y%m%d).json
```

Automatic gates (`configs/kind_ontology_cutover.json`):

- ≥ 18 compared properties (runbook check: 3 per stratum — the comparator only
  enforces the total).
- Zero stale kinds and zero unresolved successor ids in candidate output.
- Unresolved-rate increase ≤ 2 pp; estimate-coverage drop ≤ 5 pp.
- Property-level `final_rehab` low/high delta ≤ 15 % unless recorded in
  `approved_headline_deltas` with written justification.

Manual review (required before approval):

- Every changed Pass 2c claim in the report (`pass_2c_classification.changes`):
  is the split atomic and the kind correct per the ontology contract
  (docs/HANDOFF_kind_ontology_task1.md)?
- Every package change and materially different display summary needs written
  approval in the report.
- Expected multiplier drift, diagnostic not failure: ex-upgrade→degradation
  items rise (0.6→1.0), ex-defect→modernization items fall (1.0→0.6).

## Deliverables

- `reports/kind_cutover_canary_<date>.json` committed.
- `approved_headline_deltas` entries (if any) committed with justifications.
- Canary approval recorded before production enablement
  (docs/RUNBOOK_kind_ontology_release_rollback.md).
