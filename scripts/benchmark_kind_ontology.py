"""
kind-ontology-v2 classification benchmark runner.

Measures the Pass 2b/2c observation-kind-v2 contract (defect | degradation |
modernization + excluded lane) against frozen, human-approved gold cases.
Catalog-independent: no retrieval, no embeddings sidecar, no property
analysis, and nothing here writes into the artifact corpus.

Case slices live in benchmarks/kind-ontology-v2/cases_dev.json and
cases_holdout.json. Tune prompts against dev only; the holdout run decides
the completion gates. The v1 baseline replays the retired two-kind prompts
(frozen in v1_baseline_prompts.json) over the same cases to quantify the
inconsistency being fixed — chiefly where degradation-gold cases land.

Usage:
  python scripts/benchmark_kind_ontology.py --contract v2 --cases dev \
      --model-config benchmarks/kind-ontology-v2/models/terra.json --repeats 5
  python scripts/benchmark_kind_ontology.py --contract v1-baseline --cases dev \
      --model-config benchmarks/kind-ontology-v2/models/terra.json --repeats 5

Reports (JSON + Markdown) land under benchmarks/kind-ontology-v2/results/.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.llm_json import extract_json_object  # noqa: E402
from tools.pipeline_common import term_matches  # noqa: E402
from tools.scene_classifier_passes import (  # noqa: E402
    EXCLUSION_REASONS,
    OBSERVATION_KINDS,
    ONTOLOGY_VERSION,
    PASS_2B_PROMPT_SHA256,
    PASS_2B_PROMPT_VERSION,
    PASS_2C_PROMPT_SHA256,
    PASS_2C_PROMPT_VERSION,
    PassExecutionError,
    run_pass_2b,
    run_pass_2c,
    safe_format_prompt,
)

BENCH_DIR = ROOT / "benchmarks" / "kind-ontology-v2"
RESULTS_DIR = BENCH_DIR / "results"
V1_PROMPTS_PATH = BENCH_DIR / "v1_baseline_prompts.json"

BENCHMARK_ID = "kind-ontology-v2"
BENCHMARK_SCHEMA_VERSION = 1

# v1 forwarded labels, for the baseline mapping.
V1_FORWARD_LABELS = {"defect_or_damage", "upgrade_candidate"}
V1_ALL_LABELS = {"defect_or_damage", "upgrade_candidate", "good_condition", "generic_presence", "other"}

# Gold kind → the v1 label the two-kind contract would have to produce to be
# "right". degradation has no v1 home — that split is the headline stat.
V1_EXPECTED_LABEL = {"defect": "defect_or_damage", "modernization": "upgrade_candidate"}

GATES = (
    ("valid_partitions", "100% valid/complete response partitions", lambda m: m["schema_failure_calls"] == 0),
    ("kind_accuracy", ">=95% kind accuracy (mean per-repeat, atomic cases)", lambda m: m["atomic"]["kind_accuracy_mean"] >= 0.95),
    ("per_kind_recall", ">=90% recall for every kind", lambda m: min(m["atomic"]["per_kind"][k]["recall"] for k in sorted(OBSERVATION_KINDS)) >= 0.90),
    ("atomic_unanimity", ">=95% of atomic cases unanimous across repeats", lambda m: m["repeatability"]["unanimous_fraction"] >= 0.95),
    ("pairwise_agreement", ">=98% pairwise agreement across repeats", lambda m: m["repeatability"]["pairwise_agreement"] >= 0.98),
    ("excluded_false_classification", "<=5% of excluded-gold text classified with a kind", lambda m: m["excluded"]["false_classification_rate"] <= 0.05),
    ("mixed_full_case", ">=90% mixed-case full success", lambda m: m["mixed"]["full_case_success_rate"] >= 0.90),
    ("cross_kind_bundling", "<=5% mixed-case cross-kind bundling", lambda m: m["mixed"]["cross_kind_bundling_rate"] <= 0.05),
)


# ─────────────────────────────────────────────────────────────────────────────
# Case loading
# ─────────────────────────────────────────────────────────────────────────────

def load_cases(slice_name: str, bench_dir: Path = BENCH_DIR) -> Tuple[List[Dict[str, Any]], str]:
    """Load one slice ('dev' | 'holdout') and return (cases, fingerprint)."""
    path = bench_dir / f"cases_{slice_name}.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("benchmark") != BENCHMARK_ID:
        raise ValueError(f"{path} is not a {BENCHMARK_ID} case file")
    cases = payload.get("cases") or []
    validate_cases(cases)
    return cases, case_fingerprint(cases)


def case_fingerprint(cases: List[Dict[str, Any]]) -> str:
    canonical = json.dumps(cases, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def validate_cases(cases: List[Dict[str, Any]]) -> None:
    seen_ids = set()
    for case in cases:
        cid = case.get("case_id")
        if not cid or cid in seen_ids:
            raise ValueError(f"missing or duplicate case_id: {cid!r}")
        seen_ids.add(cid)
        ctype = case.get("case_type")
        if ctype not in {"atomic", "excluded", "mixed"}:
            raise ValueError(f"{cid}: unknown case_type {ctype!r}")
        gold = case.get("gold") or {}
        if ctype == "atomic":
            if gold.get("kind") not in OBSERVATION_KINDS:
                raise ValueError(f"{cid}: atomic gold.kind {gold.get('kind')!r} invalid")
            if not case.get("input"):
                raise ValueError(f"{cid}: missing input")
        elif ctype == "excluded":
            if gold.get("reason") not in EXCLUSION_REASONS:
                raise ValueError(f"{cid}: excluded gold.reason {gold.get('reason')!r} invalid")
            if not case.get("input"):
                raise ValueError(f"{cid}: missing input")
        else:
            claims = gold.get("claims") or []
            if len(claims) < 2:
                raise ValueError(f"{cid}: mixed case needs >=2 gold claims")
            for claim in claims:
                if claim.get("kind") not in OBSERVATION_KINDS:
                    raise ValueError(f"{cid}: claim kind {claim.get('kind')!r} invalid")
                if not claim.get("component") or not claim.get("anchors_any"):
                    raise ValueError(f"{cid}: claim needs component and anchors_any")
            if not case.get("input"):
                raise ValueError(f"{cid}: missing input")


def group_observation_batches(cases: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
    """Group atomic+excluded cases into their frozen scene-coherent batches.

    Batches make the 2c calls production-faithful (a photo yields several
    observations per call) and are frozen in the case file via batch_id so
    every repeat and every model sees identical batching.
    """
    batches: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for case in cases:
        if case["case_type"] in {"atomic", "excluded"}:
            batches[str(case.get("batch_id") or case["case_id"])].append(case)
    return [batches[k] for k in sorted(batches)]


# ─────────────────────────────────────────────────────────────────────────────
# Model config
# ─────────────────────────────────────────────────────────────────────────────

def load_model_config(path: Path) -> Tuple[str, Dict[str, Any]]:
    """Load an explicit model config. Never relies on OPENAI_MODEL env resolution."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    label = raw.get("label") or raw.get("model") or "unnamed-model"
    keys = ("provider", "model", "url", "api_key", "reasoning_effort", "verbosity", "max_output_tokens", "timeout")
    config = {k: raw[k] for k in keys if raw.get(k) is not None}
    if not config.get("model"):
        raise ValueError(f"{path}: model config needs a 'model'")
    provider = config.get("provider")
    if provider == "openai" and not config.get("api_key"):
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            try:
                from tools import pipeline_config
                api_key = pipeline_config.OPENAI_API_KEY
            except Exception:
                api_key = None
        if not api_key:
            raise ValueError("openai provider needs an api_key (config or OPENAI_API_KEY env)")
        config["api_key"] = api_key
    if provider == "lmstudio" and not config.get("url"):
        from tools import pipeline_config
        config["url"] = pipeline_config.LM_STUDIO_URL
    return label, config


def redact_model_config(config: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(config)
    if out.get("api_key"):
        out["api_key"] = "<redacted>"
    return out


# ─────────────────────────────────────────────────────────────────────────────
# v2 execution
# ─────────────────────────────────────────────────────────────────────────────

async def run_v2_repeat(vlm_client: Any, model_config: Dict[str, Any],
                        cases: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """One repeat over all cases under the live v2 contract.

    Returns {case_id: outcome}. Atomic/excluded outcome: {"kind": k} or
    {"exclude": r} or {"error": msg}. Mixed outcome: {"observations": [...],
    "excluded": [...]} or {"error": msg}.
    """
    outcomes: Dict[str, Dict[str, Any]] = {}

    for batch in group_observation_batches(cases):
        observations = [{"description": c["input"]} for c in batch]
        scene = batch[0].get("scene") or "other"
        try:
            result = await run_pass_2c(
                vlm_client=vlm_client,
                model_config=model_config,
                observations=observations,
                scene=scene,
            )
        except PassExecutionError as err:
            for case in batch:
                outcomes[case["case_id"]] = {"error": f"{err.stage}: {err.message}"}
            continue
        by_desc: Dict[str, Dict[str, Any]] = {}
        for obs in result.observations:
            by_desc[obs["description"]] = {"kind": obs["kind"]}
        for obs in result.excluded:
            by_desc[obs["description"]] = {"exclude": obs["reason"]}
        for case in batch:
            outcomes[case["case_id"]] = by_desc.get(
                case["input"], {"error": "observation missing from response partition"}
            )

    for case in cases:
        if case["case_type"] != "mixed":
            continue
        scene = case.get("scene") or "other"
        try:
            r2b = await run_pass_2b(
                vlm_client=vlm_client,
                model_config=model_config,
                observations_freeform=case["input"],
            )
            r2c = await run_pass_2c(
                vlm_client=vlm_client,
                model_config=model_config,
                observations=r2b.observations,
                scene=scene,
            )
        except PassExecutionError as err:
            outcomes[case["case_id"]] = {"error": f"{err.stage}: {err.message}"}
            continue
        outcomes[case["case_id"]] = {
            "observations": r2c.observations,
            "excluded": r2c.excluded,
        }
    return outcomes


# ─────────────────────────────────────────────────────────────────────────────
# v1 baseline execution (frozen retired prompts, replayed locally)
# ─────────────────────────────────────────────────────────────────────────────

def load_v1_prompts(path: Path = V1_PROMPTS_PATH) -> Dict[str, str]:
    return json.loads(path.read_text(encoding="utf-8"))


def coerce_v1_labeled(payload: Any) -> List[Dict[str, str]]:
    """Replays v1's lenient coercion (unknown label → 'other')."""
    if not isinstance(payload, list):
        return []
    out = []
    for it in payload:
        if not isinstance(it, dict):
            continue
        desc = str(it.get("description") or "").strip()
        if not desc:
            continue
        label = str(it.get("label") or "").strip().lower()
        if label not in V1_ALL_LABELS:
            label = "other"
        out.append({"description": desc, "label": label})
    return out


async def _v1_pass_2c(vlm_client: Any, model_config: Dict[str, Any], prompts: Dict[str, str],
                      observations: List[Dict[str, str]], scene: str) -> List[Dict[str, str]]:
    observations_json = json.dumps(observations, ensure_ascii=False)
    user_prompt = f"Scene: {scene}\n\n" + safe_format_prompt(
        prompts["pass_2c_user_template"], observations_json=observations_json
    )
    response = await vlm_client.analyze_text(
        system_prompt=prompts["pass_2c_system"],
        user_prompt=user_prompt,
        **{**model_config, "analysis_pass": "kind-ontology v1 baseline 2c"},
    )
    return coerce_v1_labeled((extract_json_object(response) or {}).get("labeled"))


async def run_v1_repeat(vlm_client: Any, model_config: Dict[str, Any], prompts: Dict[str, str],
                        cases: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    outcomes: Dict[str, Dict[str, Any]] = {}

    for batch in group_observation_batches(cases):
        observations = [{"description": c["input"]} for c in batch]
        scene = batch[0].get("scene") or "other"
        try:
            labeled = await _v1_pass_2c(vlm_client, model_config, prompts, observations, scene)
        except Exception as err:  # baseline is observational; record and continue
            for case in batch:
                outcomes[case["case_id"]] = {"error": str(err)[:200]}
            continue
        by_desc = {row["description"]: row["label"] for row in labeled}
        for case in batch:
            label = by_desc.get(case["input"])
            outcomes[case["case_id"]] = {"label": label} if label else {"error": "not in response"}

    for case in cases:
        if case["case_type"] != "mixed":
            continue
        scene = case.get("scene") or "other"
        try:
            system_prompt = safe_format_prompt(prompts["pass_2b_system_template"], notes=case["input"])
            response = await vlm_client.analyze_text(
                system_prompt=system_prompt,
                user_prompt=prompts["pass_2b_user"],
                **{**model_config, "analysis_pass": "kind-ontology v1 baseline 2b"},
            )
            obs_rows = (extract_json_object(response) or {}).get("observations") or []
            observations = [
                {"description": str(o.get("description") or "").strip()}
                for o in obs_rows if isinstance(o, dict) and str(o.get("description") or "").strip()
            ]
            labeled = await _v1_pass_2c(vlm_client, model_config, prompts, observations, scene) if observations else []
        except Exception as err:
            outcomes[case["case_id"]] = {"error": str(err)[:200]}
            continue
        outcomes[case["case_id"]] = {"labeled": labeled}
    return outcomes


# ─────────────────────────────────────────────────────────────────────────────
# Mixed-case matching (deterministic, no LLM judge)
# ─────────────────────────────────────────────────────────────────────────────

def observation_matches_claim(description: str, claim: Dict[str, Any]) -> bool:
    """Word-start-anchored matching: the component term and at least one
    condition anchor must both appear in the produced description."""
    text = str(description or "").lower()
    if not term_matches(str(claim["component"]).lower(), text):
        return False
    return any(term_matches(str(anchor).lower(), text) for anchor in claim["anchors_any"])


def score_mixed_outcome(case: Dict[str, Any], outcome: Dict[str, Any]) -> Dict[str, Any]:
    """Score one mixed-case repeat. Full success = every gold claim matched by
    at least one classified observation with the right kind, and no produced
    observation bundles two different gold claims."""
    if "error" in outcome:
        return {"full_success": False, "cross_kind_bundled": False, "error": outcome["error"]}

    claims = case["gold"]["claims"]
    produced = outcome.get("observations") or []

    matched_kind_ok = []
    for claim in claims:
        hits = [obs for obs in produced if observation_matches_claim(obs["description"], claim)]
        matched_kind_ok.append(any(obs["kind"] == claim["kind"] for obs in hits))

    bundled = False
    cross_kind_bundled = False
    for obs in produced:
        hit_claims = [c for c in claims if observation_matches_claim(obs["description"], c)]
        if len(hit_claims) >= 2:
            bundled = True
            if len({c["kind"] for c in hit_claims}) >= 2:
                cross_kind_bundled = True

    return {
        "full_success": all(matched_kind_ok) and not bundled,
        "claims_matched": matched_kind_ok,
        "bundled": bundled,
        "cross_kind_bundled": cross_kind_bundled,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Scoring: v2
# ─────────────────────────────────────────────────────────────────────────────

def _decision_key(outcome: Dict[str, Any]) -> str:
    if "kind" in outcome:
        return f"kind:{outcome['kind']}"
    if "exclude" in outcome:
        return f"exclude:{outcome['exclude']}"
    return "schema_failure"


def score_v2(cases: List[Dict[str, Any]], repeats: List[Dict[str, Dict[str, Any]]]) -> Dict[str, Any]:
    atomic_cases = [c for c in cases if c["case_type"] == "atomic"]
    excluded_cases = [c for c in cases if c["case_type"] == "excluded"]
    mixed_cases = [c for c in cases if c["case_type"] == "mixed"]
    n_repeats = len(repeats)

    schema_failure_calls = 0
    for rep in repeats:
        for outcome in rep.values():
            if "error" in outcome:
                schema_failure_calls += 1

    # Confusion matrix over (atomic + excluded) × repeats.
    confusion: Dict[str, Counter] = defaultdict(Counter)
    for case in atomic_cases + excluded_cases:
        gold = case["gold"].get("kind") or "excluded"
        for rep in repeats:
            outcome = rep.get(case["case_id"], {"error": "missing"})
            if "error" in outcome:
                predicted = "schema_failure"
            elif "kind" in outcome:
                predicted = outcome["kind"]
            else:
                predicted = "excluded"
            confusion[gold][predicted] += 1

    # Atomic accuracy and per-kind precision/recall (pooled over repeats).
    per_repeat_acc = []
    for rep in repeats:
        correct = sum(
            1 for case in atomic_cases
            if rep.get(case["case_id"], {}).get("kind") == case["gold"]["kind"]
        )
        per_repeat_acc.append(correct / len(atomic_cases) if atomic_cases else 0.0)

    per_kind: Dict[str, Dict[str, float]] = {}
    for kind in sorted(OBSERVATION_KINDS):
        tp = confusion[kind][kind]
        gold_total = sum(confusion[kind].values())
        pred_total = sum(row[kind] for row in confusion.values())
        per_kind[kind] = {
            "recall": tp / gold_total if gold_total else 0.0,
            "precision": tp / pred_total if pred_total else 0.0,
            "gold_decisions": gold_total,
            "predicted_decisions": pred_total,
        }

    # Excluded lane.
    excl_total = len(excluded_cases) * n_repeats
    excl_false = 0
    excl_reason_correct = 0
    for case in excluded_cases:
        for rep in repeats:
            outcome = rep.get(case["case_id"], {"error": "missing"})
            if "kind" in outcome:
                excl_false += 1
            elif outcome.get("exclude") == case["gold"]["reason"]:
                excl_reason_correct += 1

    # Repeatability over atomic + excluded cases.
    unanimous = 0
    pair_agree_fractions = []
    repeat_cases = atomic_cases + excluded_cases
    for case in repeat_cases:
        keys = [_decision_key(rep.get(case["case_id"], {"error": "missing"})) for rep in repeats]
        if len(set(keys)) == 1 and keys[0] != "schema_failure":
            unanimous += 1
        pairs = list(combinations(keys, 2))
        if pairs:
            pair_agree_fractions.append(sum(1 for a, b in pairs if a == b) / len(pairs))

    # Mixed cases.
    mixed_total = len(mixed_cases) * n_repeats
    mixed_success = 0
    mixed_cross_bundled = 0
    mixed_rows = []
    for case in mixed_cases:
        for idx, rep in enumerate(repeats):
            score = score_mixed_outcome(case, rep.get(case["case_id"], {"error": "missing"}))
            mixed_rows.append({"case_id": case["case_id"], "repeat": idx, **score})
            if score["full_success"]:
                mixed_success += 1
            if score["cross_kind_bundled"]:
                mixed_cross_bundled += 1

    return {
        "n_cases": len(cases),
        "n_repeats": n_repeats,
        "schema_failure_calls": schema_failure_calls,
        "atomic": {
            "n": len(atomic_cases),
            "kind_accuracy_mean": sum(per_repeat_acc) / len(per_repeat_acc) if per_repeat_acc else 0.0,
            "kind_accuracy_per_repeat": per_repeat_acc,
            "per_kind": per_kind,
        },
        "excluded": {
            "n": len(excluded_cases),
            "false_classification_rate": excl_false / excl_total if excl_total else 0.0,
            "reason_accuracy": excl_reason_correct / excl_total if excl_total else 0.0,
        },
        "repeatability": {
            "unanimous_fraction": unanimous / len(repeat_cases) if repeat_cases else 0.0,
            "pairwise_agreement": sum(pair_agree_fractions) / len(pair_agree_fractions) if pair_agree_fractions else 0.0,
        },
        "mixed": {
            "n": len(mixed_cases),
            "full_case_success_rate": mixed_success / mixed_total if mixed_total else 0.0,
            "cross_kind_bundling_rate": mixed_cross_bundled / mixed_total if mixed_total else 0.0,
            "rows": mixed_rows,
        },
        "confusion_matrix": {gold: dict(row) for gold, row in confusion.items()},
    }


def evaluate_gates(metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
    results = []
    for gate_id, description, check in GATES:
        try:
            passed = bool(check(metrics))
        except Exception:
            passed = False
        results.append({"gate": gate_id, "description": description, "passed": passed})
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Scoring: v1 baseline
# ─────────────────────────────────────────────────────────────────────────────

def score_v1(cases: List[Dict[str, Any]], repeats: List[Dict[str, Dict[str, Any]]]) -> Dict[str, Any]:
    atomic_cases = [c for c in cases if c["case_type"] == "atomic"]
    excluded_cases = [c for c in cases if c["case_type"] == "excluded"]
    n_repeats = len(repeats)

    label_dist: Dict[str, Counter] = {k: Counter() for k in sorted(OBSERVATION_KINDS)}
    mapped_correct = {"defect": 0, "modernization": 0}
    for case in atomic_cases:
        gold = case["gold"]["kind"]
        for rep in repeats:
            outcome = rep.get(case["case_id"], {"error": "missing"})
            label = outcome.get("label") or ("schema_failure" if "error" in outcome else "missing")
            label_dist[gold][label] += 1
            expected = V1_EXPECTED_LABEL.get(gold)
            if expected and label == expected:
                mapped_correct[gold] += 1

    # The headline: what fraction of degradation-gold decisions landed on each
    # side of the v1 defect/upgrade line.
    deg = label_dist["degradation"]
    deg_total = sum(deg.values())
    deg_forwarded = sum(deg[l] for l in V1_FORWARD_LABELS)

    excl_false_forward = 0
    excl_total = len(excluded_cases) * n_repeats
    for case in excluded_cases:
        for rep in repeats:
            label = rep.get(case["case_id"], {}).get("label")
            if label in V1_FORWARD_LABELS:
                excl_false_forward += 1

    # Repeatability on the degradation band: how often the five repeats agreed.
    deg_unanimous = 0
    deg_cases = [c for c in atomic_cases if c["gold"]["kind"] == "degradation"]
    for case in deg_cases:
        labels = [rep.get(case["case_id"], {}).get("label") for rep in repeats]
        if len(set(labels)) == 1 and labels[0] is not None:
            deg_unanimous += 1

    n_atomic_dec = len(atomic_cases) * n_repeats
    return {
        "n_cases": len(cases),
        "n_repeats": n_repeats,
        "label_distribution_by_gold_kind": {k: dict(v) for k, v in label_dist.items()},
        "mapped_accuracy": {
            "defect_as_defect_or_damage": (
                mapped_correct["defect"] / (sum(label_dist["defect"].values()) or 1)
            ),
            "modernization_as_upgrade_candidate": (
                mapped_correct["modernization"] / (sum(label_dist["modernization"].values()) or 1)
            ),
        },
        "degradation_split": {
            "decisions": deg_total,
            "defect_or_damage": deg[
                "defect_or_damage"
            ],
            "upgrade_candidate": deg["upgrade_candidate"],
            "not_forwarded": deg_total - deg_forwarded,
            "unanimous_cases": deg_unanimous,
            "n_degradation_cases": len(deg_cases),
        },
        "excluded_forwarded_rate": excl_false_forward / excl_total if excl_total else 0.0,
        "n_atomic_decisions": n_atomic_dec,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Reports
# ─────────────────────────────────────────────────────────────────────────────

def render_markdown(report: Dict[str, Any]) -> str:
    meta = report["meta"]
    lines = [
        f"# {BENCHMARK_ID} — {meta['contract']} — {meta['model_label']} — {meta['cases_slice']}",
        "",
        f"- run at: {meta['timestamp']}",
        f"- model: `{json.dumps(redact_model_config(meta['model_config']))}`",
        f"- repeats: {meta['repeats']}",
        f"- case fingerprint: `{meta['case_fingerprint']}`",
        f"- ontology: {meta['ontology_version']}",
        f"- 2b prompt: {meta['pass_2b_prompt_version']} `{meta['pass_2b_prompt_sha256'][:12]}`",
        f"- 2c prompt: {meta['pass_2c_prompt_version']} `{meta['pass_2c_prompt_sha256'][:12]}`",
        "",
    ]
    metrics = report["metrics"]
    if meta["contract"] == "v2":
        lines += [
            "## Metrics",
            "",
            f"- schema-failure calls: **{metrics['schema_failure_calls']}**",
            f"- atomic kind accuracy (mean per-repeat): **{metrics['atomic']['kind_accuracy_mean']:.3f}**",
        ]
        for kind, row in metrics["atomic"]["per_kind"].items():
            lines.append(f"- {kind}: recall {row['recall']:.3f}, precision {row['precision']:.3f}")
        lines += [
            f"- excluded false-classification rate: **{metrics['excluded']['false_classification_rate']:.3f}**",
            f"- excluded reason accuracy: {metrics['excluded']['reason_accuracy']:.3f}",
            f"- unanimity: **{metrics['repeatability']['unanimous_fraction']:.3f}**, "
            f"pairwise agreement: **{metrics['repeatability']['pairwise_agreement']:.3f}**",
            f"- mixed full-case success: **{metrics['mixed']['full_case_success_rate']:.3f}**, "
            f"cross-kind bundling: **{metrics['mixed']['cross_kind_bundling_rate']:.3f}**",
            "",
            "## Confusion matrix (gold × predicted, decisions pooled over repeats)",
            "",
        ]
        preds = sorted({p for row in metrics["confusion_matrix"].values() for p in row})
        lines.append("| gold \\ predicted | " + " | ".join(preds) + " |")
        lines.append("|---" * (len(preds) + 1) + "|")
        for gold in sorted(metrics["confusion_matrix"]):
            row = metrics["confusion_matrix"][gold]
            lines.append(f"| {gold} | " + " | ".join(str(row.get(p, 0)) for p in preds) + " |")
        if report.get("gates"):
            lines += ["", "## Gates", ""]
            for gate in report["gates"]:
                status = "PASS" if gate["passed"] else "FAIL"
                lines.append(f"- [{status}] {gate['description']}")
    else:
        lines += [
            "## v1 baseline (two-kind contract replay)",
            "",
            f"- degradation-gold split: {json.dumps(metrics['degradation_split'])}",
            f"- mapped accuracy: {json.dumps(metrics['mapped_accuracy'])}",
            f"- excluded-gold forwarded rate: {metrics['excluded_forwarded_rate']:.3f}",
            "",
            "### Label distribution by gold kind",
            "",
        ]
        for gold, dist in metrics["label_distribution_by_gold_kind"].items():
            lines.append(f"- {gold}: {json.dumps(dist)}")
    lines.append("")
    return "\n".join(lines)


def write_report(report: Dict[str, Any], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (out_dir / "report.md").write_text(render_markdown(report), encoding="utf-8")
    return out_dir / "report.json"


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", choices=("v2", "v1-baseline"), default="v2")
    parser.add_argument("--cases", choices=("dev", "holdout", "boundary"), required=True)
    parser.add_argument("--model-config", required=True, help="Path to an explicit model config JSON")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--out-dir", default=None, help="Report directory (default: results/<auto>)")
    parser.add_argument("--gates", action="store_true", help="Evaluate completion gates (v2 only)")
    return parser.parse_args(argv)


async def run_benchmark(args: argparse.Namespace) -> Dict[str, Any]:
    from tools.vlm_client import create_vlm_client

    cases, fingerprint = load_cases(args.cases)
    model_label, model_config = load_model_config(Path(args.model_config))
    vlm_client = create_vlm_client(timeout=360)

    repeats: List[Dict[str, Dict[str, Any]]] = []
    v1_prompts = load_v1_prompts() if args.contract == "v1-baseline" else None
    for i in range(args.repeats):
        print(f"[{BENCHMARK_ID}] repeat {i + 1}/{args.repeats} ({args.contract}, {model_label})")
        if args.contract == "v2":
            repeats.append(await run_v2_repeat(vlm_client, model_config, cases))
        else:
            repeats.append(await run_v1_repeat(vlm_client, model_config, v1_prompts, cases))

    metrics = score_v2(cases, repeats) if args.contract == "v2" else score_v1(cases, repeats)

    report = {
        "benchmark": BENCHMARK_ID,
        "schema_version": BENCHMARK_SCHEMA_VERSION,
        "meta": {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "contract": args.contract,
            "cases_slice": args.cases,
            "case_fingerprint": fingerprint,
            "model_label": model_label,
            "model_config": redact_model_config(model_config),
            "repeats": args.repeats,
            "ontology_version": ONTOLOGY_VERSION,
            "pass_2b_prompt_version": PASS_2B_PROMPT_VERSION,
            "pass_2b_prompt_sha256": PASS_2B_PROMPT_SHA256,
            "pass_2c_prompt_version": PASS_2C_PROMPT_VERSION,
            "pass_2c_prompt_sha256": PASS_2C_PROMPT_SHA256,
        },
        "metrics": metrics,
        "raw_repeats": repeats,
    }
    if args.contract == "v2" and args.gates:
        report["gates"] = evaluate_gates(metrics)
    return report


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    report = asyncio.run(run_benchmark(args))

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    slug = report["meta"]["model_label"].lower().replace(" ", "-").replace(".", "")
    out_dir = Path(args.out_dir) if args.out_dir else RESULTS_DIR / f"{stamp}_{args.contract}_{slug}_{args.cases}"
    path = write_report(report, out_dir)
    print(f"[{BENCHMARK_ID}] report written: {path}")

    if report.get("gates"):
        failed = [g for g in report["gates"] if not g["passed"]]
        for gate in report["gates"]:
            print(f"  [{'PASS' if gate['passed'] else 'FAIL'}] {gate['description']}")
        return 1 if failed else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
