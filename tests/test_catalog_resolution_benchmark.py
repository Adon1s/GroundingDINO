"""Harness tests for the catalog-resolution-v2 benchmark.

No sidecar, no LLM, no network: a deterministic bag-of-words encoder and a fake
client stand in, so the scoring/gating logic and the shipped case files are
checked on every run. The benchmark's own accuracy numbers come from real runs;
what is pinned here is that the harness measures what it claims to.
"""
import asyncio
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from tools.catalog_embeddings import CatalogEmbeddingsRetriever, make_candidate_provider
from tools.observation_kinds import OBSERVATION_KINDS

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "benchmarks" / "catalog-resolution-v2"


def _load_runner():
    spec = importlib.util.spec_from_file_location(
        "benchmark_catalog_resolution", ROOT / "scripts" / "benchmark_catalog_resolution.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


bench = _load_runner()


class FakeEncoder:
    """Bag-of-words over a fixed vocab: deterministic, no model or server."""

    _VOCAB = ("cabinet", "worn", "broken", "dated", "roof", "shingle",
              "missing", "moss", "water", "stain")

    def __init__(self):
        self.dimension = len(self._VOCAB)

    def encode(self, texts):
        out = np.zeros((len(texts), self.dimension), dtype=np.float32)
        for i, t in enumerate(texts):
            tl = (t or "").lower()
            v = np.array([1.0 if tok in tl else 0.0 for tok in self._VOCAB], dtype=np.float32)
            n = np.linalg.norm(v)
            if n > 0:
                v = v / n
            out[i] = v
        return out


MINI_CATALOG = {
    "version": "3.0",
    "ontology_version": "observation-kind-v2",
    "publication_status": "blocked_pending_pricing",
    "trade_buckets": [{"id": "kitchen_cabinets_counters", "name": "Cabinets"}],
    "items": [
        {"id": "cabinets_damaged_or_water_stained", "name": "Damaged cabinets",
         "kind": "defect", "trade_bucket": "kitchen_cabinets_counters",
         "scene_groups": ["kitchen"], "embed_text": "cabinet broken water stain"},
        {"id": "cabinets_worn_finish", "name": "Worn cabinet finish",
         "kind": "degradation", "trade_bucket": "kitchen_cabinets_counters",
         "scene_groups": ["kitchen"], "embed_text": "cabinet worn"},
        {"id": "cabinets_dated_style", "name": "Dated cabinets",
         "kind": "modernization", "trade_bucket": "kitchen_cabinets_counters",
         "scene_groups": ["kitchen"], "embed_text": "cabinet dated"},
    ],
}


class FakeClient:
    """Resolves to the top candidate; records how often it was called."""

    def __init__(self):
        self.calls = 0

    async def analyze_text(self, system_prompt, user_prompt, **model_config):
        self.calls += 1
        for line in (user_prompt or "").splitlines():
            if line.startswith("- "):
                item_id = line[2:].split(" | ")[0].strip()
                return json.dumps({"resolved_item_id": item_id})
        return json.dumps({"resolved_item_id": None})


def _mini_provider():
    retriever = CatalogEmbeddingsRetriever(MINI_CATALOG, encoder=FakeEncoder())
    return make_candidate_provider(retriever)


def _case(case_id, kind, gold, text, **kw):
    case = {
        "case_id": case_id, "family": "cabinets", "paired_group": "cabinets-trio",
        "case_type": "resolution", "scene": "kitchen", "scene_group": "kitchen",
        "input": text, "kind": kind,
        "gold": {"resolved_id": gold, "acceptable_candidate_ids": [gold]},
        "legacy": {"kind": "defect", "resolved_id": "outdated_or_damaged_cabinets",
                   "acceptable_candidate_ids": ["outdated_or_damaged_cabinets"]},
    }
    case.update(kw)
    return case


# ── shipped case files ──────────────────────────────────────────────────────

@pytest.mark.parametrize("slice_name", bench.available_slices())
def test_shipped_case_slices_validate(slice_name):
    cases, fingerprint, status = bench.load_cases(slice_name)
    assert cases
    assert len(fingerprint) == 64
    assert status


def test_every_frozen_slice_is_actually_frozen():
    """--gates refuses a non-frozen slice, so an unfrozen one is a latent gate
    failure rather than a scoring difference."""
    for slice_name in bench.available_slices():
        _, _, status = bench.load_cases(slice_name)
        assert status.startswith("frozen"), f"{slice_name} is {status!r}"


def test_every_split_successor_is_gold_somewhere():
    manifest = json.loads(
        (ROOT / "tools" / "catalog_migrations" / "2.1_to_3.0.json").read_text(encoding="utf-8")
    )
    successors = {
        s["id"] for e in manifest["entries"] if e["change_type"] == "split"
        for s in e["successors"]
    }
    covered = set()
    for slice_name in bench.available_slices():
        cases, _, _ = bench.load_cases(slice_name)
        covered |= {c["gold"]["resolved_id"] for c in cases if c["gold"]["resolved_id"]}
    assert successors <= covered, f"never scored: {sorted(successors - covered)}"


def test_shipped_cases_reference_real_catalog_items():
    catalog = json.loads((ROOT / "tools" / "issue_catalog_kind_v2.json").read_text(encoding="utf-8"))
    by_id = {i["id"]: i for i in catalog["items"]}
    for slice_name in bench.available_slices():
        cases, _, _ = bench.load_cases(slice_name)
        for case in cases:
            gold = case["gold"]["resolved_id"]
            if gold is None:
                continue
            assert gold in by_id, f"{case['case_id']}: unknown gold {gold}"
            assert by_id[gold]["kind"] == case["kind"], f"{case['case_id']}: kind disagrees with catalog"
            assert case["scene_group"] in by_id[gold]["scene_groups"], (
                f"{case['case_id']}: gold is not retrievable in scene group {case['scene_group']}"
            )


def test_paired_groups_never_span_slices():
    """Paired metrics need the whole group in one run."""
    groups = {}
    for slice_name in bench.available_slices():
        cases, _, _ = bench.load_cases(slice_name)
        for case in cases:
            if case.get("paired_group"):
                groups.setdefault(case["paired_group"], set()).add(slice_name)
    split = {g: s for g, s in groups.items() if len(s) > 1}
    assert not split, f"paired groups split across slices: {split}"


def test_case_slices_are_disjoint():
    """No case id may appear in two slices: a case scored twice would be counted
    twice, and paired metrics would straddle runs."""
    seen: dict = {}
    for slice_name in bench.available_slices():
        cases, _, _ = bench.load_cases(slice_name)
        for case in cases:
            other = seen.get(case["case_id"])
            assert other is None, f"{case['case_id']} appears in both {other} and {slice_name}"
            seen[case["case_id"]] = slice_name


# ── case validation ─────────────────────────────────────────────────────────

def test_validate_rejects_duplicate_case_ids():
    with pytest.raises(ValueError, match="duplicate case_id"):
        bench.validate_cases([_case("a", "defect", "x", "t"), _case("a", "defect", "x", "t")])


def test_validate_rejects_a_kind_outside_the_ontology():
    with pytest.raises(ValueError, match="not in the ontology"):
        bench.validate_cases([_case("a", "upgrade", "x", "t")])


def test_validate_requires_invalid_kind_probes_to_be_actually_invalid():
    probe = _case("p", "defect", None, "t", case_type="invalid_kind")
    probe["gold"] = {"resolved_id": None, "acceptable_candidate_ids": []}
    with pytest.raises(ValueError, match="outside the ontology"):
        bench.validate_cases([probe])


def test_validate_requires_no_match_gold_to_be_null():
    bad = _case("n", "defect", None, "t", case_type="no_match")
    bad["gold"] = {"resolved_id": "something", "acceptable_candidate_ids": []}
    with pytest.raises(ValueError, match="must have gold.resolved_id null"):
        bench.validate_cases([bad])


def test_fingerprint_changes_when_gold_changes():
    a = [_case("a", "defect", "x", "t")]
    b = [_case("a", "defect", "y", "t")]
    assert bench.case_fingerprint(a) != bench.case_fingerprint(b)
    assert bench.case_fingerprint(a) == bench.case_fingerprint(list(a))


# ── scoring ─────────────────────────────────────────────────────────────────

def _perfect_outcome(case):
    gold = case["gold"]["resolved_id"]
    return {
        "resolved_id": gold,
        "candidate_ids": [gold, "other_item"],
        "candidate_kinds": [case["kind"]],
        "candidate_count": 2,
        "retrieved": True,
    }


def test_score_perfect_run():
    cases = [
        _case("c1", "defect", "cabinets_damaged_or_water_stained", "broken cabinet"),
        _case("c2", "degradation", "cabinets_worn_finish", "worn cabinet"),
        _case("c3", "modernization", "cabinets_dated_style", "dated cabinet"),
    ]
    repeats = [{c["case_id"]: _perfect_outcome(c) for c in cases}]
    m = bench.score("v2", cases, repeats)

    assert m["recall_at_k"]["overall"] == 1.0
    assert m["final_accuracy"]["overall"] == 1.0
    assert set(m["final_accuracy"]["by_kind"]) == OBSERVATION_KINDS
    assert m["kind_purity"] == 1.0
    assert m["paired"]["recall_at_k"] == 1.0
    assert m["rank"]["top1_rate"] == 1.0
    assert m["failures"]["total"] == 0


def test_score_counts_recall_miss_and_wrong_resolution():
    cases = [_case("c1", "degradation", "cabinets_worn_finish", "worn cabinet")]
    repeats = [{"c1": {
        "resolved_id": "cabinets_dated_style",
        "candidate_ids": ["cabinets_dated_style"],
        "candidate_kinds": ["degradation"],
        "candidate_count": 1, "retrieved": True,
    }}]
    m = bench.score("v2", cases, repeats)

    assert m["recall_at_k"]["overall"] == 0.0
    assert m["final_accuracy"]["overall"] == 0.0
    assert m["split_family_confusions"] == 1


def test_score_flags_impure_candidate_pool():
    cases = [_case("c1", "degradation", "cabinets_worn_finish", "worn cabinet")]
    repeats = [{"c1": {
        "resolved_id": "cabinets_worn_finish",
        "candidate_ids": ["cabinets_worn_finish"],
        "candidate_kinds": ["degradation", "defect"],
        "candidate_count": 2, "retrieved": True,
    }}]
    assert bench.score("v2", cases, repeats)["kind_purity"] == 0.0


def test_score_counts_no_match_false_positives():
    case = _case("n1", "defect", None, "unrelated", case_type="no_match")
    case["gold"] = {"resolved_id": None, "acceptable_candidate_ids": []}
    repeats = [{"n1": {"resolved_id": "cabinets_dated_style", "candidate_ids": [], "retrieved": True}}]
    m = bench.score("v2", [case], repeats)
    assert m["no_match"]["false_positive_rate"] == 1.0


def test_score_probe_violation_when_empty_filter_returns_candidates():
    case = _case("e1", "defect", None, "x", case_type="empty_filter")
    case["gold"] = {"resolved_id": None, "acceptable_candidate_ids": []}
    clean = bench.score("v2", [case], [{"e1": {"candidate_ids": [], "unknown_kind_candidate_ids": []}}])
    dirty = bench.score("v2", [case], [{"e1": {"candidate_ids": ["anything"], "unknown_kind_candidate_ids": []}}])
    assert clean["probes"]["violations"] == 0
    assert dirty["probes"]["violations"] == 1


def test_score_invalid_kind_probe_requires_fail_closed_in_v2():
    case = _case("i1", "upgrade", None, "x", case_type="invalid_kind")
    case["gold"] = {"resolved_id": None, "acceptable_candidate_ids": []}
    good = bench.score("v2", [case], [{"i1": {"pass_error_code": "invalid_kind", "retrieved": False}}])
    bad = bench.score("v2", [case], [{"i1": {"resolved_id": None, "candidate_ids": ["x"], "retrieved": True}}])
    assert good["probes"]["violations"] == 0
    assert bad["probes"]["violations"] == 1


def test_score_uses_legacy_gold_in_the_legacy_lane():
    cases = [_case("c1", "degradation", "cabinets_worn_finish", "worn cabinet")]
    repeats = [{"c1": {
        "resolved_id": "outdated_or_damaged_cabinets",
        "candidate_ids": ["outdated_or_damaged_cabinets"],
        "candidate_kinds": ["defect"], "candidate_count": 1, "retrieved": True,
    }}]
    assert bench.score("legacy", cases, repeats)["final_accuracy"]["overall"] == 1.0
    # the same outcome scores zero against the v2 gold
    assert bench.score("v2", cases, repeats)["final_accuracy"]["overall"] == 0.0


def test_score_records_errors_as_failures():
    cases = [_case("c1", "defect", "cabinets_damaged_or_water_stained", "broken cabinet")]
    m = bench.score("v2", cases, [{"c1": {"error": "RuntimeError: boom"}}])
    assert m["failures"]["total"] == 1


# ── gates ───────────────────────────────────────────────────────────────────

def _passing_metrics():
    return {
        "failures": {"total": 0},
        "kind_purity": 1.0,
        "recall_at_k": {"k": 5, "overall": 1.0,
                        "by_kind": {k: 1.0 for k in OBSERVATION_KINDS}},
        "final_accuracy": {"overall": 1.0, "by_kind": {k: 1.0 for k in OBSERVATION_KINDS}},
        "paired": {"recall_at_k": 1.0, "final_accuracy": 1.0, "n": 3},
        "probes": {"n": 2, "violations": 0},
        "comparable": {"recall_at_k": 1.0, "final_accuracy": 1.0, "n": 3},
    }


def test_gates_all_pass_on_perfect_metrics():
    gates = bench.evaluate_gates(_passing_metrics(), None)
    assert [g["gate"] for g in gates if not g["passed"]] == []


@pytest.mark.parametrize("mutate, failing_gate", [
    (lambda m: m["failures"].update(total=1), "no_failures"),
    (lambda m: m.update(kind_purity=0.99), "kind_purity"),
    (lambda m: m["recall_at_k"].update(overall=0.97), "recall_overall"),
    (lambda m: m["recall_at_k"]["by_kind"].update(degradation=0.94), "recall_per_kind"),
    (lambda m: m["final_accuracy"].update(overall=0.94), "final_accuracy_overall"),
    (lambda m: m["final_accuracy"]["by_kind"].update(modernization=0.89), "final_accuracy_per_kind"),
    (lambda m: m["paired"].update(recall_at_k=0.99), "paired_recall"),
    (lambda m: m["paired"].update(final_accuracy=0.94), "paired_accuracy"),
    (lambda m: m["probes"].update(violations=1), "filter_probes"),
    (lambda m: m["probes"].update(n=0), "filter_probes"),
])
def test_each_gate_fails_on_its_own_threshold(mutate, failing_gate):
    metrics = _passing_metrics()
    mutate(metrics)
    failed = {g["gate"] for g in bench.evaluate_gates(metrics, None) if not g["passed"]}
    assert failing_gate in failed


def test_no_regression_gate_compares_against_the_legacy_baseline():
    metrics = _passing_metrics()
    metrics["comparable"]["recall_at_k"] = 0.99
    better_legacy = {"comparable": {"recall_at_k": 1.0}}
    worse_legacy = {"comparable": {"recall_at_k": 0.90}}

    def gate_state(baseline):
        return {g["gate"]: g["passed"] for g in bench.evaluate_gates(metrics, baseline)}

    assert gate_state(better_legacy)["recall_no_regression"] is False
    assert gate_state(worse_legacy)["recall_no_regression"] is True
    assert gate_state(None)["recall_no_regression"] is True


# ── end-to-end over the mini catalog (fake encoder, fake client) ────────────

def test_end_to_end_v2_lane_resolves_within_its_kind():
    cases = [
        _case("c1", "defect", "cabinets_damaged_or_water_stained", "broken cabinet with water stain"),
        _case("c2", "degradation", "cabinets_worn_finish", "worn cabinet"),
        _case("c3", "modernization", "cabinets_dated_style", "dated cabinet"),
    ]
    client = FakeClient()
    outcomes = asyncio.run(bench.run_repeat(
        lane="v2", cases=cases, provider=_mini_provider(), vlm_client=client,
        model_config={}, top_k=5,
    ))

    m = bench.score("v2", cases, [outcomes])
    assert m["kind_purity"] == 1.0
    assert m["recall_at_k"]["overall"] == 1.0
    assert m["failures"]["total"] == 0
    for case in cases:
        assert outcomes[case["case_id"]]["candidate_kinds"] == [case["kind"]]


def test_end_to_end_invalid_kind_never_reaches_retrieval():
    probe = _case("i1", "upgrade", None, "worn cabinet", case_type="invalid_kind")
    probe["gold"] = {"resolved_id": None, "acceptable_candidate_ids": []}

    seen = []
    base = _mini_provider()

    def spy(description, context):
        seen.append(description)
        return base(description, context)

    client = FakeClient()
    outcomes = asyncio.run(bench.run_repeat(
        lane="v2", cases=[probe], provider=spy, vlm_client=client,
        model_config={}, top_k=5,
    ))

    assert outcomes["i1"]["pass_error_code"] == "invalid_kind"
    assert seen == []
    assert client.calls == 0
    assert bench.score("v2", [probe], [outcomes])["probes"]["violations"] == 0


def test_end_to_end_empty_filter_probe_returns_nothing():
    probe = _case("e1", "degradation", None, "worn cabinet", case_type="empty_filter")
    probe["gold"] = {"resolved_id": None, "acceptable_candidate_ids": []}
    outcomes = asyncio.run(bench.run_repeat(
        lane="v2", cases=[probe], provider=_mini_provider(), vlm_client=FakeClient(),
        model_config={}, top_k=5,
    ))
    assert outcomes["e1"]["candidate_ids"] == []
    assert outcomes["e1"]["unknown_kind_candidate_ids"] == []


# ── legacy snapshot ─────────────────────────────────────────────────────────

def test_legacy_snapshot_preserves_v1_routing_semantics():
    legacy = bench.load_legacy_snapshot()
    # asymmetric widening: upgrade + exterior component + condition widens
    assert legacy.legacy_evaluate_kind_routing(
        "Shingles appear aged and weathered.", "upgrade").expanded_kinds == ("upgrade", "defect")
    # a defect never widens
    assert legacy.legacy_evaluate_kind_routing(
        "Shingles are missing.", "defect").expanded_kinds == ("defect",)
    # the bug being measured: an unknown kind yields an EMPTY route
    assert legacy.legacy_evaluate_kind_routing(
        "Carpet is worn.", "degradation").expanded_kinds == ()


def test_legacy_provider_reproduces_the_whole_catalog_search_bug():
    """An unknown kind gave legacy an empty allowed_kinds, which its retriever
    read as 'no filter'. This is the baseline the v2 lane is measured against."""
    legacy = bench.load_legacy_snapshot()
    retriever = CatalogEmbeddingsRetriever(MINI_CATALOG, encoder=FakeEncoder())
    provider = legacy.legacy_candidate_provider_factory(retriever)

    hits = provider("worn cabinet", {
        "kind": "degradation", "allowed_kinds": [], "scene_group": "kitchen",
        "top_k_candidates": 5,
    })
    kinds = {c["kind"] for c in hits}
    assert len(kinds) > 1, "legacy should have searched across kinds here"

    # the v2 provider, same retriever, same input: zero candidates
    assert make_candidate_provider(retriever)("worn cabinet", {
        "kind": "degradation", "allowed_kinds": [], "scene_group": "kitchen",
        "top_k_candidates": 5,
    }) == []


# ── reporting ───────────────────────────────────────────────────────────────

def test_write_report_emits_json_and_markdown_without_leaking_keys(tmp_path):
    report = {
        "benchmark": bench.BENCHMARK_ID,
        "schema_version": 1,
        "meta": {
            "lane": "v2", "cases_slice": "dev", "repeats": 3,
            "case_fingerprint": "a" * 64, "catalog_fingerprint": "b" * 64,
            "model_config": {"model": "gpt-5.6-terra", "api_key": "<redacted>"},
            "embeddings_config": {"model_name": "jina"},
            "pass_2d_prompt_version": "pass_2d_exact_kind_v2",
            "pass_2d_prompt_sha256": "c" * 64,
        },
        "metrics": bench.score("v2", [], []),
        "gates": [{"gate": "kind_purity", "description": "100% purity", "passed": True}],
    }
    bench.write_report(report, tmp_path / "out")

    assert json.loads((tmp_path / "out" / "report.json").read_text(encoding="utf-8"))
    md = (tmp_path / "out" / "report.md").read_text(encoding="utf-8")
    assert "secret" not in md
    assert "<redacted>" in md
    assert "PASS" in md


def test_redact_model_config_hides_the_api_key():
    out = bench.redact_model_config({"model": "m", "api_key": "sk-super-secret"})
    assert out["api_key"] == "<redacted>"
    assert "sk-super-secret" not in json.dumps(out)


def test_model_config_is_file_driven():
    label, config = bench.load_model_config(BENCH / "models" / "qwen.json")
    assert label == "Qwen 3.6 27B"
    assert config["model"] == "unsloth/qwen3.6-27b@q6_k"
    assert config["url"]  # filled from pipeline_config, never from an env model name


def test_embeddings_config_is_explicit():
    cfg = bench.load_embeddings_config()
    assert cfg["model_name"]
    assert cfg["base_url"].startswith("http")
    assert cfg["dimension"] == 1024
