"""Scorer for the factorized replay: joins, gates, and the degeneracy guard.

The load-bearing test here is `test_degenerate_always_yes_arm_...`: G1 and G2
are both satisfied by a verifier that answers `yes` to everything, so the
label-free G4 checks are what make the decisive gates trustworthy. If that
property ever breaks, the gate design is wrong.

Run: .venv\\Scripts\\python.exe -m pytest tests/test_score_factorized_review.py -q
"""
import json
from pathlib import Path

import pytest

from scripts.score_factorized_review import (
    SUPPRESSING_CLASSES,
    class_confusion,
    evaluate_g4,
    evaluate_gates,
    load_arm,
    load_canary_labels,
    main,
)
from tools.renovation_architecture.factorized_review import derive_class

LABELS = Path("reports/labels_v1_1.json")
pytestmark = pytest.mark.skipif(
    not LABELS.is_file(), reason="frozen v1.1 labels not present"
)

# label slug -> the factors a perfect verifier would answer.
ORACLE = {
    "exact_and_warranted": ("yes", "yes", "yes"),
    "misnamed_but_warranted": ("yes", "no", "yes"),
    "exact_but_trivial": ("yes", "yes", "no"),
    "misnamed_and_trivial": ("yes", "no", "no"),
    "wrong_object_or_place": ("no", "unclear", "unclear"),
    "absent": ("no", "unclear", "unclear"),
    "inconclusive": ("unclear", "unclear", "unclear"),
}


def _answer(visible, accurate, material):
    factors = {
        "visible": visible,
        "claim_accurate_as_written": accurate,
        "material_enough_for_work": material,
    }
    return {
        **factors,
        "derived_class": derive_class(factors),
        "observed_description": "water staining" if accurate == "no" else "",
        "rationale": "synthetic",
    }


def _write_arm(root: Path, rows, factors_for, *, dry_run=False):
    """One unit file per listing, mirroring the harness output shape."""
    by_prop = {}
    for row in rows:
        by_prop.setdefault(row["property_key"], []).append(row)
    for prop, prop_rows in by_prop.items():
        unit_dir = root / prop / "units"
        unit_dir.mkdir(parents=True, exist_ok=True)
        (unit_dir / "terra_unit_0000000000000000.json").write_text(
            json.dumps({
                "property_key": prop,
                "estimate_unit_id": "unit_1",
                "condition_ids": [r["condition_id"] for r in prop_rows],
                "status": "redecided",
                "reviews": {
                    r["condition_id"]: _answer(*factors_for(r)) for r in prop_rows
                },
            }),
            encoding="utf-8",
        )
    (root / "manifest.json").write_text(
        json.dumps({
            "variant": {"label": "factorized_v1"},
            "dry_run": dry_run,
            "response_contract": "factorized_v1",
            "prompt_version": "terra_factorized_review_v1",
        }),
        encoding="utf-8",
    )
    return root


@pytest.fixture
def rows():
    loaded, meta = load_canary_labels(LABELS)
    assert meta["excluded_production_cards"] == 37  # excluded loudly, not silently
    assert len(loaded) == 88
    return loaded


class TestPopulations:
    def test_gate_populations_match_the_precommitted_counts(self, rows):
        gates = evaluate_gates(rows, {})
        for gate_id, expected in (
            ("G1", 57), ("G2", 16), ("G3a", 7), ("G3b", 5), ("G3c", 3)
        ):
            assert gates[gate_id]["n_population"] == expected, gate_id
            assert gates[gate_id]["n_expected"] == expected, gate_id

    def test_unjoined_arm_leaves_every_gate_unscored(self, rows):
        gates = evaluate_gates(rows, {})
        assert all(g["passed"] is None for g in gates.values())


class TestOracleArm:
    """A verifier that reproduces the human labels exactly must sweep."""

    def test_all_gates_pass(self, tmp_path, rows):
        root = _write_arm(tmp_path / "arm", rows, lambda r: ORACLE[r["slug"]])
        answers = load_arm(root)["answers"]
        gates = evaluate_gates(rows, answers)
        for gate_id, gate in gates.items():
            assert gate["passed"] is True, (gate_id, gate)
        assert gates["G1"]["fired"] == 0
        assert gates["G2"]["fired"] == 16

    def test_class_confusion_is_diagonal(self, tmp_path, rows):
        root = _write_arm(tmp_path / "arm", rows, lambda r: ORACLE[r["slug"]])
        answers = load_arm(root)["answers"]
        for label_class, counts in class_confusion(rows, answers).items():
            assert set(counts) == {label_class}, (label_class, counts)


class TestDegenerateArm:
    def test_always_yes_passes_the_decisive_label_gates(self, tmp_path, rows):
        """The hole G4 exists to close — assert it is real, not hypothetical."""
        root = _write_arm(tmp_path / "arm", rows, lambda r: ("yes", "yes", "yes"))
        answers = load_arm(root)["answers"]
        gates = evaluate_gates(rows, answers)
        assert gates["G1"]["passed"] is True   # suppresses nothing
        assert gates["G2"]["passed"] is True   # "recovers" everything
        # ...and catches none of the actual error classes.
        assert gates["G3a"]["fired"] == 0
        assert gates["G3b"]["fired"] == 0
        assert gates["G3c"]["fired"] == 0

    def test_g4_catches_it(self, tmp_path, rows):
        root = _write_arm(tmp_path / "arm", rows, lambda r: ("yes", "yes", "yes"))
        g4 = evaluate_g4(load_arm(root)["answers"])
        assert g4["passed"] is False
        assert g4["checks"]["exact_and_warranted_share"]["passed"] is False
        assert g4["checks"]["misnamed_share"]["passed"] is False

    def test_all_unclear_arm_is_caught_by_the_unclear_ceiling(self, tmp_path, rows):
        root = _write_arm(
            tmp_path / "arm", rows, lambda r: ("unclear", "unclear", "unclear")
        )
        g4 = evaluate_g4(load_arm(root)["answers"])
        assert g4["passed"] is False
        assert g4["checks"]["unclear_visible"]["passed"] is False


class TestSuppressionSemantics:
    def test_misnamed_is_not_counted_as_suppression(self, tmp_path, rows):
        """Misnamed routes to remapping, keeping the work — it is a label error
        but not a suppression, so G1 must not fire on it."""
        assert "misnamed_but_warranted" not in SUPPRESSING_CLASSES
        root = _write_arm(tmp_path / "arm", rows, lambda r: ("yes", "no", "yes"))
        gates = evaluate_gates(rows, load_arm(root)["answers"])
        assert gates["G1"]["fired"] == 0

    def test_trivial_and_absent_do_count(self, tmp_path, rows):
        root = _write_arm(tmp_path / "arm", rows, lambda r: ("yes", "yes", "no"))
        gates = evaluate_gates(rows, load_arm(root)["answers"])
        assert gates["G1"]["fired"] == gates["G1"]["n_judged"] == 57
        assert gates["G1"]["passed"] is False


class TestEndToEnd:
    def test_writes_the_pair_and_reports_dry_run_as_unscored(self, tmp_path, rows):
        root = _write_arm(
            tmp_path / "arm", rows, lambda r: ORACLE[r["slug"]], dry_run=True
        )
        out_md = tmp_path / "scorecard.md"
        out_json = tmp_path / "scorecard.json"
        rc = main([
            "--arm-root", str(root), "--labels", str(LABELS),
            "--canary-root", str(tmp_path / "no_canary"),
            "--out-md", str(out_md), "--out-json", str(out_json),
        ])
        assert rc == 1  # dry-run roots never claim integrity
        body = out_md.read_text(encoding="utf-8")
        assert "DRY RUN root" in body
        assert "What this does not show" in body
        assert "no dirB loss population" in body
        payload = json.loads(out_json.read_text(encoding="utf-8"))
        assert payload["label_version"] == "v1.1"
        assert payload["coverage"]["labels_joined"] == 88

    def test_live_oracle_arm_reports_integrity_ok(self, tmp_path, rows):
        root = _write_arm(tmp_path / "arm", rows, lambda r: ORACLE[r["slug"]])
        rc = main([
            "--arm-root", str(root), "--labels", str(LABELS),
            "--canary-root", str(tmp_path / "no_canary"),
            "--out-md", str(tmp_path / "s.md"),
            "--out-json", str(tmp_path / "s.json"),
        ])
        assert rc == 0
