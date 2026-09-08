"""catalog-resolution-v2 benchmark runner.

Measures Pass 2d catalog resolution under the observation-kind-v2 contract:
does a three-kind observation retrieve the right candidates from the v2 catalog,
and does the resolver pick the right item?

Two lanes over the same frozen cases and the same encoder:

  v2      - tools/issue_catalog_kind_v2.json, strict exact-kind retrieval, via
            the production closure (catalog_embeddings.make_candidate_provider)
            and the production resolution step
            (scene_classifier_orchestrator.resolve_observation_against_catalog).
  legacy  - tools/issue_catalog.json with the v1 routing/filter semantics frozen
            in benchmarks/catalog-resolution-v2/legacy_baseline/routing_snapshot.py.

Only retrieval semantics differ between lanes; both resolve with the current
Pass 2d prompt, so the comparison isolates catalog structure + retrieval instead
of confounding it with a prompt change.

Nothing here writes into the artifact corpus, and results are non-publishable by
construction: the v2 catalog is publication_status=blocked_pending_pricing and
write_photo_intel refuses it.

Usage:
  python scripts/benchmark_catalog_resolution.py --lane v2 --cases dev \
      --model-config benchmarks/catalog-resolution-v2/models/terra.json --repeats 3
  python scripts/benchmark_catalog_resolution.py --lane legacy --cases dev \
      --model-config benchmarks/catalog-resolution-v2/models/terra.json --repeats 3
  python scripts/benchmark_catalog_resolution.py --lane v2 --cases holdout \
      --model-config benchmarks/catalog-resolution-v2/models/terra.json \
      --repeats 5 --compare-to <legacy-report.json> --gates
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.util
import json
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.catalog_embeddings import (  # noqa: E402
    CatalogEmbeddingsRetriever,
    build_guardrails_from_catalog,
    make_candidate_provider,
)
from tools.observation_kinds import OBSERVATION_KINDS, ONTOLOGY_VERSION  # noqa: E402
from tools.scene_classifier_orchestrator import (  # noqa: E402
    resolve_observation_against_catalog,
)
from tools.scene_classifier_passes import (  # noqa: E402
    PASS_2D_PROMPT_SHA256,
    PASS_2D_PROMPT_VERSION,
    PassExecutionError,
)

BENCHMARK_ID = "catalog-resolution-v2"
BENCHMARK_SCHEMA_VERSION = 1
BENCH_DIR = ROOT / "benchmarks" / BENCHMARK_ID
RESULTS_DIR = BENCH_DIR / "results"
EMBEDDINGS_CONFIG_PATH = BENCH_DIR / "embeddings.json"
LEGACY_SNAPSHOT_PATH = BENCH_DIR / "legacy_baseline" / "routing_snapshot.py"
V2_CATALOG_PATH = ROOT / "tools" / "issue_catalog_kind_v2.json"
V1_CATALOG_PATH = ROOT / "tools" / "issue_catalog.json"

RECALL_AT = 5

CASE_TYPES = ("resolution", "no_match", "invalid_kind", "empty_filter")

# Gates encode the approved holdout thresholds. Each takes the scored metrics
# plus an optional comparable legacy baseline.
GATES: Tuple[Tuple[str, str, Any], ...] = (
    ("no_failures", "Zero schema/provider failures",
     lambda m, b: m["failures"]["total"] == 0),
    ("kind_purity", "100% candidate-kind purity",
     lambda m, b: m["kind_purity"] == 1.0),
    ("recall_overall", f"Candidate recall@{RECALL_AT} >= 0.98 overall",
     lambda m, b: m["recall_at_k"]["overall"] >= 0.98),
    ("recall_per_kind", f"Candidate recall@{RECALL_AT} >= 0.95 for every kind",
     lambda m, b: all(v >= 0.95 for v in m["recall_at_k"]["by_kind"].values())),
    ("recall_no_regression", "Candidate recall >= comparable legacy baseline",
     lambda m, b: b is None or m["comparable"]["recall_at_k"] >= b["comparable"]["recall_at_k"]),
    ("final_accuracy_overall", "Final resolved-ID accuracy >= 0.95 overall",
     lambda m, b: m["final_accuracy"]["overall"] >= 0.95),
    ("final_accuracy_per_kind", "Final resolved-ID accuracy >= 0.90 for every kind",
     lambda m, b: all(v >= 0.90 for v in m["final_accuracy"]["by_kind"].values())),
    ("paired_recall", "Paired cases achieve 100% candidate recall",
     lambda m, b: m["paired"]["recall_at_k"] == 1.0),
    ("paired_accuracy", "Paired cases achieve >= 0.95 final accuracy",
     lambda m, b: m["paired"]["final_accuracy"] >= 0.95),
    ("filter_probes", "Unknown/empty kind filters never search the catalog",
     lambda m, b: m["probes"]["violations"] == 0 and m["probes"]["n"] > 0),
)


# ─────────────────────────────────────────────────────────────────────────────
# Cases
# ─────────────────────────────────────────────────────────────────────────────

def case_fingerprint(cases: List[Dict[str, Any]]) -> str:
    return hashlib.sha256(
        json.dumps(cases, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def file_fingerprint(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_cases(cases: List[Dict[str, Any]]) -> None:
    seen = set()
    for case in cases:
        cid = case.get("case_id")
        if not cid:
            raise ValueError("case missing case_id")
        if cid in seen:
            raise ValueError(f"{cid}: duplicate case_id")
        seen.add(cid)
        ctype = case.get("case_type")
        if ctype not in CASE_TYPES:
            raise ValueError(f"{cid}: case_type {ctype!r} not in {list(CASE_TYPES)}")
        if not case.get("input"):
            raise ValueError(f"{cid}: missing input")
        if not case.get("scene_group"):
            raise ValueError(f"{cid}: missing scene_group")
        gold = case.get("gold") or {}
        if ctype == "resolution":
            if case.get("kind") not in OBSERVATION_KINDS:
                raise ValueError(f"{cid}: kind {case.get('kind')!r} not in the ontology")
            if not gold.get("resolved_id"):
                raise ValueError(f"{cid}: resolution case needs gold.resolved_id")
            if not gold.get("acceptable_candidate_ids"):
                raise ValueError(f"{cid}: resolution case needs acceptable_candidate_ids")
        elif ctype == "no_match":
            if case.get("kind") not in OBSERVATION_KINDS:
                raise ValueError(f"{cid}: kind {case.get('kind')!r} not in the ontology")
            if gold.get("resolved_id") is not None:
                raise ValueError(f"{cid}: no_match case must have gold.resolved_id null")
        elif ctype == "invalid_kind":
            if case.get("kind") in OBSERVATION_KINDS:
                raise ValueError(f"{cid}: invalid_kind case must carry a kind outside the ontology")


def available_slices() -> List[str]:
    """Every shipped case slice, discovered from disk.

    Slices are additive: an approved change that owes gold lands a NEW frozen
    slice rather than editing a fingerprinted one, so the set cannot be a
    hardcoded pair. Sorted for a stable CLI and stable test ordering.
    """
    return sorted(p.stem[len("cases_"):] for p in BENCH_DIR.glob("cases_*.json"))


def load_cases(slice_name: str) -> Tuple[List[Dict[str, Any]], str, str]:
    payload = json.loads((BENCH_DIR / f"cases_{slice_name}.json").read_text(encoding="utf-8"))
    cases = payload.get("cases") or []
    validate_cases(cases)
    return cases, case_fingerprint(cases), str(payload.get("status") or "unknown")


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

def load_model_config(path: Path) -> Tuple[str, Dict[str, Any]]:
    """Load an explicit model config. Never relies on env model resolution."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    label = raw.get("label") or raw.get("model") or "unnamed-model"
    keys = ("provider", "model", "url", "api_key", "reasoning_effort",
            "verbosity", "max_output_tokens", "timeout")
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


def load_embeddings_config() -> Dict[str, Any]:
    return json.loads(EMBEDDINGS_CONFIG_PATH.read_text(encoding="utf-8"))


def build_retriever(catalog: Dict[str, Any], embeddings_config: Dict[str, Any],
                    encoder: Any = None) -> CatalogEmbeddingsRetriever:
    """Build a retriever over `catalog`.

    Constructing it embeds the whole catalog, which doubles as a real POST
    against the embeddings sidecar — the /health endpoint has been observed
    returning 200 while every embeddings call fails, so this is the probe.
    """
    return CatalogEmbeddingsRetriever(
        catalog_v2=catalog,
        model_name=embeddings_config["model_name"],
        device=embeddings_config.get("device", "cpu"),
        trust_remote_code=embeddings_config.get("trust_remote_code", True),
        default_topk=embeddings_config.get("topk", 5),
        guardrails=build_guardrails_from_catalog(catalog),
        backend=embeddings_config.get("backend", "openai_compatible"),
        base_url=embeddings_config.get("base_url"),
        st_model_name=embeddings_config.get("st_model_name"),
        embedding_dimension=embeddings_config.get("dimension", 1024),
        encoder=encoder,
    )


def load_legacy_snapshot() -> Any:
    spec = importlib.util.spec_from_file_location(
        "catalog_resolution_legacy_snapshot", LEGACY_SNAPSHOT_PATH
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# ─────────────────────────────────────────────────────────────────────────────
# Execution
# ─────────────────────────────────────────────────────────────────────────────

def _candidate_ids(candidates: List[Dict[str, Any]]) -> List[str]:
    return [
        str(c.get("item_id") or c.get("id") or "")
        for c in (candidates or [])
    ]


async def run_repeat(
    *,
    lane: str,
    cases: List[Dict[str, Any]],
    provider: Any,
    vlm_client: Any,
    model_config: Dict[str, Any],
    top_k: int,
) -> Dict[str, Dict[str, Any]]:
    """One repeat over all cases. Returns {case_id: outcome}."""
    outcomes: Dict[str, Dict[str, Any]] = {}
    legacy = load_legacy_snapshot() if lane == "legacy" else None

    for case in cases:
        cid = case["case_id"]
        ctype = case["case_type"]
        scene_group = case["scene_group"]

        # ── probes: exercise the filter contract directly, no LLM ───────────
        if ctype == "empty_filter":
            candidates = provider(case["input"], {
                "allowed_kinds": [], "scene_group": scene_group, "top_k_candidates": top_k,
            })
            unknown = provider(case["input"], {
                "allowed_kinds": ["upgrade"], "scene_group": scene_group, "top_k_candidates": top_k,
            })
            outcomes[cid] = {
                "candidate_ids": _candidate_ids(candidates),
                "unknown_kind_candidate_ids": _candidate_ids(unknown),
            }
            continue

        kind = case["kind"]
        observation = {
            "description": case["input"],
            "kind": kind,
            "issue_id": f"bench-{cid}",
            "scene_group": scene_group,
            "source_photo_key": f"{cid}.jpg",
        }
        base_context = {"scene": case.get("scene"), "scene_group": scene_group}

        if lane == "legacy":
            legacy_meta = case.get("legacy") or {}
            legacy_kind = legacy_meta.get("kind") or kind
            routing = legacy.legacy_evaluate_kind_routing(case["input"], legacy_kind)
            try:
                candidates = provider(case["input"], {
                    **base_context,
                    "kind": legacy_kind,
                    "allowed_kinds": list(routing.expanded_kinds),
                    "top_k_candidates": top_k,
                })
            except Exception as exc:  # noqa: BLE001 - recorded, not raised
                outcomes[cid] = {"error": f"{type(exc).__name__}: {exc}"}
                continue
            candidates = candidates[:top_k]
            outcomes[cid] = await _resolve_and_record(
                vlm_client=vlm_client, model_config=model_config,
                observation_text=case["input"], candidates=candidates, kind=legacy_kind,
                expanded_kinds=list(routing.expanded_kinds),
            )
            continue

        # ── v2 lane: the production resolution path ─────────────────────────
        try:
            resolved_row, debug_row, _ = await resolve_observation_against_catalog(
                vlm_client=vlm_client,
                model_config=model_config,
                candidate_provider=provider,
                observation=observation,
                base_context=base_context,
                top_k=top_k,
                source_image_path=f"{cid}.jpg",
            )
        except PassExecutionError as exc:
            outcomes[cid] = {
                "pass_error_code": exc.code,
                "pass_error_stage": exc.stage,
                "retrieved": False,
            }
            continue
        except Exception as exc:  # noqa: BLE001
            outcomes[cid] = {"error": f"{type(exc).__name__}: {exc}"}
            continue

        candidates = (resolved_row or {}).get("candidates") or []
        outcomes[cid] = {
            "resolved_id": (resolved_row or {}).get("resolved_item_id"),
            "candidate_ids": _candidate_ids(candidates),
            "candidate_kinds": sorted({str(c.get("kind") or "") for c in candidates}),
            "resolution_path": debug_row.get("resolution_path"),
            "skipped_reason": debug_row.get("skipped_reason"),
            "candidate_count": debug_row.get("candidate_count", 0),
            "retrieved": True,
        }
    return outcomes


async def _resolve_and_record(*, vlm_client, model_config, observation_text,
                              candidates, kind, expanded_kinds) -> Dict[str, Any]:
    """Legacy lane resolution: current 2d prompt over legacy candidates."""
    from tools.scene_classifier_passes import run_pass_2d

    if not candidates:
        return {
            "resolved_id": None, "candidate_ids": [], "candidate_kinds": [],
            "candidate_count": 0, "retrieved": True,
            "allowed_kinds": expanded_kinds,
        }
    try:
        result = await run_pass_2d(
            vlm_client=vlm_client, model_config=model_config,
            observation=observation_text, candidates=candidates, kind=kind,
        )
    except Exception as exc:  # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {exc}"}
    return {
        "resolved_id": result.resolved_item_id,
        "candidate_ids": _candidate_ids(candidates),
        "candidate_kinds": sorted({str(c.get("kind") or "") for c in candidates}),
        "candidate_count": len(candidates),
        "resolution_path": result.resolution_path,
        "retrieved": True,
        "allowed_kinds": expanded_kinds,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Scoring
# ─────────────────────────────────────────────────────────────────────────────

def _rate(hits: int, total: int) -> float:
    return hits / total if total else 0.0


def score(lane: str, cases: List[Dict[str, Any]],
          repeats: List[Dict[str, Dict[str, Any]]]) -> Dict[str, Any]:
    by_id = {c["case_id"]: c for c in cases}
    n_repeats = len(repeats) or 1

    failures = Counter()
    recall_hits: Counter = Counter()
    recall_total: Counter = Counter()
    rank_positions: List[int] = []
    acc_hits: Counter = Counter()
    acc_total: Counter = Counter()
    purity_ok = 0
    purity_total = 0
    nomatch_fp = 0
    nomatch_total = 0
    comparable_recall_hits = 0
    comparable_recall_total = 0
    comparable_acc_hits = 0
    comparable_acc_total = 0
    paired_recall_hits = 0
    paired_recall_total = 0
    paired_acc_hits = 0
    paired_acc_total = 0
    probe_violations = 0
    probe_n = 0
    confusion: Dict[str, Counter] = defaultdict(Counter)
    invalid_kind_ok = 0
    invalid_kind_total = 0
    rows: List[Dict[str, Any]] = []

    for case in cases:
        cid = case["case_id"]
        ctype = case["case_type"]
        gold = case.get("gold") or {}
        gold_id = gold.get("resolved_id")
        acceptable = set(gold.get("acceptable_candidate_ids") or [])
        kind = case.get("kind")
        legacy_meta = case.get("legacy") or {}
        if lane == "legacy" and legacy_meta:
            gold_id = legacy_meta.get("resolved_id")
            acceptable = set(legacy_meta.get("acceptable_candidate_ids") or [])
            kind = legacy_meta.get("kind") or kind

        for rep in repeats:
            outcome = rep.get(cid) or {"error": "missing outcome"}

            if ctype == "empty_filter":
                probe_n += 1
                if outcome.get("candidate_ids") or outcome.get("unknown_kind_candidate_ids"):
                    probe_violations += 1
                continue

            if ctype == "invalid_kind":
                probe_n += 1
                invalid_kind_total += 1
                if lane == "v2":
                    # Must fail closed before retrieval.
                    if outcome.get("pass_error_code") == "invalid_kind":
                        invalid_kind_ok += 1
                    else:
                        probe_violations += 1
                else:
                    # v1 semantics: an unknown kind produced an empty allowed_kinds
                    # set, which the legacy retriever read as "no filter".
                    if outcome.get("candidate_ids"):
                        probe_violations += 1
                    else:
                        invalid_kind_ok += 1
                continue

            if "error" in outcome:
                failures["error"] += 1
                continue

            candidate_ids = outcome.get("candidate_ids") or []
            top_k_ids = candidate_ids[:RECALL_AT]

            if ctype == "resolution":
                # candidate recall@k
                hit = bool(acceptable & set(top_k_ids))
                recall_total[kind] += 1
                comparable_recall_total += 1
                if hit:
                    recall_hits[kind] += 1
                    comparable_recall_hits += 1
                    rank_positions.append(
                        min(i for i, c in enumerate(top_k_ids, start=1) if c in acceptable)
                    )
                # kind purity (v2 only enforces one kind; legacy may widen)
                if lane == "v2":
                    purity_total += 1
                    kinds_seen = set(outcome.get("candidate_kinds") or [])
                    if not kinds_seen or kinds_seen == {kind}:
                        purity_ok += 1
                # final accuracy
                acc_total[kind] += 1
                comparable_acc_total += 1
                resolved = outcome.get("resolved_id")
                if resolved == gold_id:
                    acc_hits[kind] += 1
                    comparable_acc_hits += 1
                confusion[str(gold_id)][str(resolved)] += 1
                if case.get("paired_group"):
                    paired_recall_total += 1
                    paired_acc_total += 1
                    if hit:
                        paired_recall_hits += 1
                    if resolved == gold_id:
                        paired_acc_hits += 1
                rows.append({
                    "case_id": cid, "kind": kind, "gold": gold_id,
                    "resolved": resolved, "recall_hit": hit,
                    "candidates": top_k_ids,
                })
            elif ctype == "no_match":
                nomatch_total += 1
                if outcome.get("resolved_id"):
                    nomatch_fp += 1

    failures["total"] = sum(failures.values())

    # split-family confusion: resolved into a sibling successor of the same parent
    family_confusions = 0
    for gold_id, counts in confusion.items():
        for resolved, n in counts.items():
            if resolved != gold_id and resolved not in ("None", ""):
                family_confusions += n

    return {
        "lane": lane,
        "n_cases": len(cases),
        "n_repeats": n_repeats,
        "failures": {"total": failures.get("total", 0), "error": failures.get("error", 0)},
        "recall_at_k": {
            "k": RECALL_AT,
            "overall": _rate(sum(recall_hits.values()), sum(recall_total.values())),
            "by_kind": {
                k: _rate(recall_hits[k], recall_total[k])
                for k in sorted(recall_total)
            },
        },
        "rank": {
            "mean": sum(rank_positions) / len(rank_positions) if rank_positions else 0.0,
            "top1_rate": _rate(sum(1 for r in rank_positions if r == 1), len(rank_positions)),
            "n": len(rank_positions),
        },
        "final_accuracy": {
            "overall": _rate(sum(acc_hits.values()), sum(acc_total.values())),
            "by_kind": {
                k: _rate(acc_hits[k], acc_total[k])
                for k in sorted(acc_total)
            },
        },
        "kind_purity": _rate(purity_ok, purity_total) if purity_total else 1.0,
        "no_match": {
            "n": nomatch_total,
            "false_positive_rate": _rate(nomatch_fp, nomatch_total),
        },
        "paired": {
            "n": paired_recall_total,
            "recall_at_k": _rate(paired_recall_hits, paired_recall_total),
            "final_accuracy": _rate(paired_acc_hits, paired_acc_total),
        },
        "probes": {
            "n": probe_n,
            "violations": probe_violations,
            "invalid_kind_handled": _rate(invalid_kind_ok, invalid_kind_total),
        },
        "comparable": {
            "recall_at_k": _rate(comparable_recall_hits, comparable_recall_total),
            "final_accuracy": _rate(comparable_acc_hits, comparable_acc_total),
            "n": comparable_recall_total,
        },
        "split_family_confusions": family_confusions,
        "confusion_matrix": {g: dict(c) for g, c in confusion.items()},
        "rows": rows,
    }


def evaluate_gates(metrics: Dict[str, Any],
                   baseline: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    results = []
    for gate_id, description, check in GATES:
        try:
            passed = bool(check(metrics, baseline))
        except Exception:
            passed = False
        results.append({"gate": gate_id, "description": description, "passed": passed})
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Reporting
# ─────────────────────────────────────────────────────────────────────────────

def render_markdown(report: Dict[str, Any]) -> str:
    meta = report["meta"]
    m = report["metrics"]
    lines = [
        f"# {report['benchmark']} — {meta['lane']} / {meta['cases_slice']}",
        "",
        f"- model: `{json.dumps(meta['model_config'])}`",
        f"- embeddings: `{json.dumps(meta['embeddings_config'])}`",
        f"- repeats: {meta['repeats']}",
        f"- case fingerprint: `{meta['case_fingerprint']}`",
        f"- catalog fingerprint: `{meta['catalog_fingerprint']}`",
        f"- 2d prompt: {meta['pass_2d_prompt_version']} `{meta['pass_2d_prompt_sha256'][:16]}…`",
        "",
        "## Headline",
        "",
        f"- candidate recall@{m['recall_at_k']['k']}: **{m['recall_at_k']['overall']:.3f}** "
        f"(by kind: {json.dumps({k: round(v, 3) for k, v in m['recall_at_k']['by_kind'].items()})})",
        f"- final resolved-ID accuracy: **{m['final_accuracy']['overall']:.3f}** "
        f"(by kind: {json.dumps({k: round(v, 3) for k, v in m['final_accuracy']['by_kind'].items()})})",
        f"- candidate-kind purity: **{m['kind_purity']:.3f}**",
        f"- mean gold rank: {m['rank']['mean']:.2f} (top-1 {m['rank']['top1_rate']:.3f})",
        f"- no-match false positives: {m['no_match']['false_positive_rate']:.3f} "
        f"(n={m['no_match']['n']})",
        f"- paired cases: recall {m['paired']['recall_at_k']:.3f}, "
        f"accuracy {m['paired']['final_accuracy']:.3f} (n={m['paired']['n']})",
        f"- filter probes: {m['probes']['violations']} violations of {m['probes']['n']}",
        f"- split-family confusions: {m['split_family_confusions']}",
        f"- failures: {m['failures']['total']}",
        "",
    ]
    if report.get("comparison"):
        comp = report["comparison"]
        lines += [
            "## Legacy comparison (comparable population)",
            "",
            f"- recall@{RECALL_AT}: v2 {comp['v2_recall']:.3f} vs legacy {comp['legacy_recall']:.3f} "
            f"(delta {comp['recall_delta']:+.3f})",
            f"- final accuracy: v2 {comp['v2_accuracy']:.3f} vs legacy {comp['legacy_accuracy']:.3f} "
            f"(delta {comp['accuracy_delta']:+.3f})",
            "",
        ]
    if report.get("gates"):
        lines += ["## Gates", ""]
        for gate in report["gates"]:
            lines.append(f"- [{'PASS' if gate['passed'] else 'FAIL'}] {gate['description']}")
        lines.append("")
    misses = [r for r in m["rows"] if not r["recall_hit"] or r["resolved"] != r["gold"]]
    if misses:
        lines += ["## Misses", "", "| case | kind | gold | resolved | recall hit |", "|---|---|---|---|---|"]
        for r in misses[:60]:
            lines.append(
                f"| {r['case_id']} | {r['kind']} | {r['gold']} | {r['resolved']} | "
                f"{'yes' if r['recall_hit'] else 'NO'} |"
            )
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
    parser.add_argument("--lane", choices=("v2", "legacy"), default="v2")
    parser.add_argument("--cases", choices=tuple(available_slices()), required=True)
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--compare-to", default=None,
                        help="Path to a legacy-lane report.json for the no-regression gate")
    parser.add_argument("--gates", action="store_true")
    return parser.parse_args(argv)


async def run_benchmark(args: argparse.Namespace) -> Dict[str, Any]:
    from tools.vlm_client import create_vlm_client

    cases, fingerprint, status = load_cases(args.cases)
    if args.gates and status != "frozen" and not status.startswith("frozen"):
        raise SystemExit(
            f"refusing to run gates against a {status!r} case slice — freeze it first"
        )
    model_label, model_config = load_model_config(Path(args.model_config))
    embeddings_config = load_embeddings_config()

    catalog_path = V2_CATALOG_PATH if args.lane == "v2" else V1_CATALOG_PATH
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    retriever = build_retriever(catalog, embeddings_config)
    if args.lane == "v2":
        provider = make_candidate_provider(retriever)
    else:
        provider = load_legacy_snapshot().legacy_candidate_provider_factory(retriever)

    vlm_client = create_vlm_client(timeout=360)
    top_k = embeddings_config.get("top_k_candidates", 8)

    repeats: List[Dict[str, Dict[str, Any]]] = []
    for i in range(args.repeats):
        print(f"[{BENCHMARK_ID}] repeat {i + 1}/{args.repeats} ({args.lane}, {model_label})")
        repeats.append(await run_repeat(
            lane=args.lane, cases=cases, provider=provider, vlm_client=vlm_client,
            model_config=model_config, top_k=top_k,
        ))

    metrics = score(args.lane, cases, repeats)

    report = {
        "benchmark": BENCHMARK_ID,
        "schema_version": BENCHMARK_SCHEMA_VERSION,
        "meta": {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "lane": args.lane,
            "cases_slice": args.cases,
            "cases_status": status,
            "case_fingerprint": fingerprint,
            "catalog_path": str(catalog_path.relative_to(ROOT)),
            "catalog_fingerprint": file_fingerprint(catalog_path),
            "manifest_fingerprint": file_fingerprint(
                ROOT / "tools" / "catalog_migrations" / "2.1_to_3.0.json"
            ),
            "embeddings_config": embeddings_config,
            "model_label": model_label,
            "model_config": redact_model_config(model_config),
            "repeats": args.repeats,
            "ontology_version": ONTOLOGY_VERSION,
            "pass_2d_prompt_version": PASS_2D_PROMPT_VERSION,
            "pass_2d_prompt_sha256": PASS_2D_PROMPT_SHA256,
        },
        "metrics": metrics,
        "raw_repeats": repeats,
    }

    baseline = None
    if args.compare_to:
        baseline_report = json.loads(Path(args.compare_to).read_text(encoding="utf-8"))
        baseline = baseline_report.get("metrics")
        if baseline:
            report["comparison"] = {
                "baseline_report": str(args.compare_to),
                "v2_recall": metrics["comparable"]["recall_at_k"],
                "legacy_recall": baseline["comparable"]["recall_at_k"],
                "recall_delta": metrics["comparable"]["recall_at_k"] - baseline["comparable"]["recall_at_k"],
                "v2_accuracy": metrics["comparable"]["final_accuracy"],
                "legacy_accuracy": baseline["comparable"]["final_accuracy"],
                "accuracy_delta": metrics["comparable"]["final_accuracy"] - baseline["comparable"]["final_accuracy"],
            }

    if args.gates:
        report["gates"] = evaluate_gates(metrics, baseline)
    return report


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    report = asyncio.run(run_benchmark(args))

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    slug = report["meta"]["model_label"].lower().replace(" ", "-").replace(".", "")
    out_dir = Path(args.out_dir) if args.out_dir else RESULTS_DIR / f"{stamp}_{args.lane}_{slug}_{args.cases}"
    path = write_report(report, out_dir)
    print(f"[{BENCHMARK_ID}] report written: {path}")

    if report.get("gates"):
        for gate in report["gates"]:
            print(f"  [{'PASS' if gate['passed'] else 'FAIL'}] {gate['description']}")
        return 1 if any(not g["passed"] for g in report["gates"]) else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
