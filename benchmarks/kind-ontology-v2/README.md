# kind-ontology-v2 benchmark

Focused, catalog-independent benchmark for the observation-kind-v2 Pass 2b/2c
contract (`defect | degradation | modernization` + excluded lane). Separate
from the photo benchmark system in `benchmarks/datasets/` — this one is
text-level, needs no photos, no retrieval, and no embeddings sidecar.

## Layout

- `cases_dev.json` — 93 development cases. Prompt tuning happens ONLY here.
- `cases_holdout.json` — 39 holdout cases. Opened once per prompt SHA; decides
  the completion gates. Record `PASS_2C_PROMPT_SHA256` before running it.
- `v1_baseline_prompts.json` — frozen retired two-kind prompts for the
  baseline replay.
- `models/terra.json`, `models/qwen.json` — explicit model configs. The gates
  run on Terra; Qwen runs are reported for production-transfer visibility.
- `results/` — timestamped JSON + Markdown reports.

## Case composition (132 total, dev 93 / holdout 39, stratified)

- 90 atomic cases (30 per kind) — fed straight to Pass 2c in frozen
  scene-coherent batches (`batch_id`).
- 18 excluded cases (3 per exclusion reason).
- 24 mixed-note decomposition cases — run 2b → 2c end to end; gold is a list
  of claims (`component` + `anchors_any` + `kind`) matched deterministically
  (word-start anchored, no LLM judge).

Sources: real observation texts mined from the artifact corpus
(`photo_intel_debug.json`, renointel-prod) plus synthetic edge cases; every
case records `source`. All 132 gold labels were human-reviewed before freeze;
the case-set sha256 fingerprint is embedded in every report.

## Running

```bash
# v1 baseline (replays the retired prompts)
.venv/Scripts/python.exe scripts/benchmark_kind_ontology.py --contract v1-baseline --cases dev --model-config benchmarks/kind-ontology-v2/models/terra.json --repeats 5

# v2 development
.venv/Scripts/python.exe scripts/benchmark_kind_ontology.py --contract v2 --cases dev --model-config benchmarks/kind-ontology-v2/models/terra.json --repeats 5

# v2 holdout with gates (the completion decision)
.venv/Scripts/python.exe scripts/benchmark_kind_ontology.py --contract v2 --cases holdout --model-config benchmarks/kind-ontology-v2/models/terra.json --repeats 5 --gates
```

## Gates (holdout, Terra, 5 repeats)

1. 100% valid/complete response partitions (zero schema failures).
2. ≥95% atomic kind accuracy (mean per-repeat).
3. ≥90% recall for each kind.
4. ≥95% of atomic cases unanimous across the 5 repeats.
5. ≥98% pairwise agreement.
6. ≤5% of excluded-gold text classified with a kind.
7. ≥90% mixed-case full success.
8. ≤5% cross-kind bundling.

Small-n caveat: 39 holdout cases means gate 2 tolerates at most one miss. A
single-case failure is recorded and analyzed in the handoff, never silently
tuned against holdout.
