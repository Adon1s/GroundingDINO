"""Tests for scripts/audit_pass2c_funnel.py.

Synthetic artifacts under tmp_path — the corpus walk is never pointed at real
data here. Modelled on tests/test_audit_exterior_estimate_coverage.py.

Run: `.venv\\Scripts\\python.exe -m pytest tests/test_audit_pass2c_funnel.py -q`
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
_SPEC = importlib.util.spec_from_file_location(
    "audit_pass2c_funnel",
    ROOT / "scripts" / "audit_pass2c_funnel.py",
)
audit = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = audit
_SPEC.loader.exec_module(audit)


# ─── fixtures ────────────────────────────────────────────────────────────────

def _photo(scene_group: str, observations, resolved=None):
    """One photo entry. `observations` is (description, label) pairs.

    Forwarding mirrors the live split at scene_classifier_passes.py:1031 —
    defect_or_damage and upgrade_candidate only.
    """
    labeled_debug = [{"description": d, "label": lb} for d, lb in observations]
    labeled_forward = [
        dict(row, scene_group=scene_group)
        for row in labeled_debug
        if row["label"] in audit.FORWARDED_LABELS
    ]
    return {
        "scene": {"id": f"{scene_group}_front", "group": scene_group},
        "debug": {
            "labeled_debug": labeled_debug,
            "labeled_forward": labeled_forward,
            "resolved_items": [
                {"description": d, "resolved_item_id": cid}
                for d, cid in (resolved or [])
            ],
        },
    }


def _write_artifact(
    root: Path,
    property_key: str,
    photos,
    *,
    run_id: str = "20260101_000000_abcdef12",
    created_at: str = "2026-01-01T00:00:00.000000Z",
    raw: str = None,
) -> Path:
    run_dir = root / property_key / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / audit.ARTIFACT_NAME
    if raw is not None:
        path.write_text(raw, encoding="utf-8")
        return path
    artifact = {
        "property": {"property_key": property_key},
        "run": {"run_id": run_id, "created_at": created_at},
        "photos": photos,
    }
    path.write_text(json.dumps(artifact), encoding="utf-8")
    return path


ABSENCE_DESC = "No visible gutters or downspouts direct water away from the foundation."
DAMAGE_DESC = "A downspout is disconnected or poorly routed near the porch steps."
DATED_DESC = "Wood lap siding appears weathered, with staining and aged paint."


@pytest.fixture
def corpus(tmp_path):
    """Two properties: one with an absence claim resolving onto a gutter item."""
    root = tmp_path / "artifacts"
    _write_artifact(root, "redfin_1", {
        "photo_001.jpg": _photo("exterior", [
            (ABSENCE_DESC, "defect_or_damage"),
            (DAMAGE_DESC, "defect_or_damage"),
            (DATED_DESC, "upgrade_candidate"),
            ("There is a front door.", "generic_presence"),
        ], resolved=[
            (ABSENCE_DESC, "clogged_or_damaged_gutters"),
            (DAMAGE_DESC, "clogged_or_damaged_gutters"),
        ]),
    })
    _write_artifact(root, "redfin_2", {
        "photo_001.jpg": _photo("kitchen", [
            ("Cabinets are dated oak.", "upgrade_candidate"),
            ("The counters look clean.", "good_condition"),
        ]),
    })
    return root


# ─── run selection ───────────────────────────────────────────────────────────

def test_selects_latest_run_by_created_at(tmp_path):
    root = tmp_path / "artifacts"
    # Dir names sort the other way round from created_at on purpose: whichever
    # field the selector actually uses, only one of these can win.
    _write_artifact(root, "p", {"a.jpg": _photo("exterior", [("old", "other")])},
                    run_id="20260301_000000_aaaaaaaa",
                    created_at="2026-01-01T00:00:00Z")
    _write_artifact(root, "p", {"a.jpg": _photo("exterior", [("new", "other")])},
                    run_id="20260101_000000_bbbbbbbb",
                    created_at="2026-03-01T00:00:00Z")

    selected = audit.select_artifacts(root)

    assert [p.parent.name for p in selected] == ["20260101_000000_bbbbbbbb"]


def test_falls_back_to_dir_name_when_created_at_missing(tmp_path):
    root = tmp_path / "artifacts"
    for run_id in ("20260101_000000_aaaaaaaa", "20260301_000000_bbbbbbbb"):
        run_dir = root / "p" / run_id
        run_dir.mkdir(parents=True)
        (run_dir / audit.ARTIFACT_NAME).write_text(
            json.dumps({"run": {}, "photos": {}}), encoding="utf-8",
        )

    selected = audit.select_artifacts(root)

    assert [p.parent.name for p in selected] == ["20260301_000000_bbbbbbbb"]


def test_skips_empty_run_dirs_and_dot_dirs(tmp_path):
    """`.checkpoints` is a sibling of the run dirs, and dead runs leave empty
    directories behind — neither may win latest-run selection."""
    root = tmp_path / "artifacts"
    _write_artifact(root, "p", {"a.jpg": _photo("exterior", [("x", "other")])},
                    run_id="20260101_000000_aaaaaaaa")
    (root / "p" / "20260501_000000_cccccccc").mkdir(parents=True)   # empty run
    (root / "p" / ".checkpoints" / "abc").mkdir(parents=True)

    selected = audit.select_artifacts(root)

    assert [p.parent.name for p in selected] == ["20260101_000000_aaaaaaaa"]


def test_all_runs_selects_every_run(tmp_path):
    root = tmp_path / "artifacts"
    for run_id in ("20260101_000000_aaaaaaaa", "20260301_000000_bbbbbbbb"):
        _write_artifact(root, "p", {"a.jpg": _photo("exterior", [("x", "other")])},
                        run_id=run_id)

    assert len(audit.select_artifacts(root, all_runs=True)) == 2
    assert len(audit.select_artifacts(root)) == 1


# ─── funnel counts ───────────────────────────────────────────────────────────

def test_scene_group_counts_come_from_the_photo_not_the_observation(corpus):
    """`scene_group` is stamped onto forwarded rows only, so grouping by the
    observation field would divide a full numerator by an empty denominator."""
    summary = audit.build_audit(corpus)["summary"]

    groups = summary["by_scene_group"]
    assert groups["exterior"] == {
        "observations": 4, "forwarded": 3, "forward_rate": 75.0,
    }
    assert groups["kitchen"] == {
        "observations": 2, "forwarded": 1, "forward_rate": 50.0,
    }
    assert summary["observations"] == 6
    assert summary["forwarded"] == 4


def test_label_distribution_splits_forwarded_from_dropped(corpus):
    summary = audit.build_audit(corpus)["summary"]

    forwarded = {e["label"]: e["occurrences"] for e in summary["labels_forwarded"]}
    dropped = {e["label"]: e["occurrences"] for e in summary["labels_dropped"]}

    assert forwarded == {"defect_or_damage": 2, "upgrade_candidate": 2}
    assert dropped == {"generic_presence": 1, "good_condition": 1}


def test_deprecated_safety_label_is_reported_not_swallowed(tmp_path):
    """`safety` is coerced to `other` in the live pass; historical artifacts
    still carry it and the audit must not hide it inside `other`."""
    root = tmp_path / "artifacts"
    _write_artifact(root, "p", {
        "a.jpg": _photo("exterior", [("Exposed wiring near the meter.", "safety")]),
    })

    summary = audit.build_audit(root)["summary"]

    assert {"label": "safety", "occurrences": 1} in summary["labels_dropped"]


# ─── absence cohort ──────────────────────────────────────────────────────────

def test_absence_cohort_counts_and_resolution(corpus):
    absence = audit.build_audit(corpus)["summary"]["absence_cohort"]

    assert absence["observations"] == 1
    assert absence["forwarded"] == 1
    assert absence["unresolved"] == 0
    assert absence["properties"] == 1
    assert absence["gutter_item_resolutions"] == 1
    assert {"catalog_item_id": "clogged_or_damaged_gutters", "occurrences": 1} \
        in absence["resolutions"]


# ─── the deny-list replay ────────────────────────────────────────────────────
# A stored resolution records the catalog as it was when the run was analysed,
# so the historical count cannot move without re-analysing the whole corpus.
# The replay is what makes the gate usable against a historical corpus.

def _catalog(tmp_path: Path, deny_any) -> Path:
    path = tmp_path / "catalog.json"
    path.write_text(json.dumps({
        "version": "test-1.0",
        "trade_buckets": [{"id": "roof_gutters"}],
        "items": [{
            "id": "clogged_or_damaged_gutters",
            "name": "Clogged or Damaged Gutters/Downspouts",
            "kind": "defect",
            "deny_any": deny_any,
            "support_any": ["gutter", "downspout"],
        }],
    }), encoding="utf-8")
    return path


def test_replay_marks_stored_resolution_as_blocked(corpus, tmp_path):
    catalog = _catalog(tmp_path, ["no visible"])

    absence = audit.build_audit(corpus, catalog_path=catalog)["summary"]["absence_cohort"]

    # The historical fact is unchanged; only the replayed number moves.
    assert absence["gutter_item_resolutions"] == 1
    assert absence["gutter_item_resolutions_under_current_catalog"] == 0


def test_replay_leaves_unblocked_resolution_visible(corpus, tmp_path):
    catalog = _catalog(tmp_path, ["something else entirely"])

    absence = audit.build_audit(corpus, catalog_path=catalog)["summary"]["absence_cohort"]

    assert absence["gutter_item_resolutions"] == 1
    assert absence["gutter_item_resolutions_under_current_catalog"] == 1


def test_shipped_catalog_blocks_the_fixture_absence_claim(corpus):
    """Guards the real deny list, not a synthetic one."""
    absence = audit.build_audit(corpus)["summary"]["absence_cohort"]

    assert absence["gutter_item_resolutions_under_current_catalog"] == 0


def test_damage_claim_is_not_counted_as_absence(corpus):
    """A visible disconnected downspout is a damage claim; it resolves onto a
    gutter item legitimately and must stay out of the cohort."""
    rows = audit.build_audit(corpus)["artifacts"]
    example_descs = [
        ex["description"]
        for row in rows for ex in (row["absence_cohort"]["examples"])
    ]

    assert DAMAGE_DESC not in example_descs
    assert example_descs == [ABSENCE_DESC]


def test_contradiction_proxy_needs_both_signals(tmp_path):
    root = tmp_path / "artifacts"
    # Absence claim alongside separate, non-absence gutter evidence.
    _write_artifact(root, "both", {
        "a.jpg": _photo("exterior", [(ABSENCE_DESC, "defect_or_damage")]),
        "b.jpg": _photo("exterior", [(DAMAGE_DESC, "defect_or_damage")]),
    })
    # Absence claim only.
    _write_artifact(root, "absence_only", {
        "a.jpg": _photo("exterior", [(ABSENCE_DESC, "defect_or_damage")]),
    })

    rows = {r["property_key"]: r for r in audit.build_audit(root)["artifacts"]}

    assert rows["both"]["contradiction"] is True
    assert rows["absence_only"]["contradiction"] is False


def test_unresolved_absence_is_counted_separately(tmp_path):
    """Pass 2d already refuses most absence claims; that is the good outcome and
    must be distinguishable from a claim that reached a catalog item."""
    root = tmp_path / "artifacts"
    _write_artifact(root, "p", {
        "a.jpg": _photo("exterior", [(ABSENCE_DESC, "defect_or_damage")]),
    })

    absence = audit.build_audit(root)["summary"]["absence_cohort"]

    assert absence["forwarded"] == 1
    assert absence["unresolved"] == 1
    assert absence["gutter_item_resolutions"] == 0


def test_absence_patterns_are_reported_by_name(corpus):
    """Cohort membership must be explainable, so a pattern that only fires on
    false positives is visible rather than buried in a total."""
    absence = audit.build_audit(corpus)["summary"]["absence_cohort"]

    assert {"pattern": "no_before_component", "occurrences": 1} \
        in absence["pattern_hits"]


def test_pattern_manifest_ships_with_the_snapshot(corpus):
    patterns = audit.build_audit(corpus)["patterns"]

    assert "review queue" in patterns["caveat"].lower()
    assert {p["name"] for p in patterns["absence_patterns"]} == {
        name for name, _ in audit.ABSENCE_PATTERN_SPECS
    }


# ─── exterior finish cohort ──────────────────────────────────────────────────

def test_exterior_finish_cohort_tracks_forwarding(corpus):
    ext = audit.build_audit(corpus)["summary"]["exterior_finish_cohort"]

    # "Wood lap siding..." (forwarded) and the porch-steps downspout (forwarded).
    assert ext["forwarded"] == 2
    assert ext["dropped"] == 0


# ─── malformed artifacts ─────────────────────────────────────────────────────

def test_malformed_artifact_is_reported_and_walk_continues(tmp_path):
    root = tmp_path / "artifacts"
    _write_artifact(root, "broken", {}, raw='{"photos": {,}')
    _write_artifact(root, "good", {
        "a.jpg": _photo("exterior", [(DATED_DESC, "upgrade_candidate")]),
    })

    summary = audit.build_audit(root)["summary"]

    assert summary["artifacts_audited"] == 1
    assert summary["artifacts_parse_error"] == 1
    assert summary["parse_errors"][0]["artifact"].startswith("broken/")
    assert "JSONDecodeError" in summary["parse_errors"][0]["reason"]
    assert summary["observations"] == 1


def test_main_fails_only_when_nothing_parses(tmp_path, capsys):
    root = tmp_path / "artifacts"
    _write_artifact(root, "broken", {}, raw="not json at all")

    assert audit.main(["--artifacts-root", str(root)]) == 1
    assert "no auditable artifacts" in capsys.readouterr().err


def test_main_succeeds_when_some_artifacts_survive(tmp_path):
    root = tmp_path / "artifacts"
    _write_artifact(root, "broken", {}, raw="not json at all")
    _write_artifact(root, "good", {
        "a.jpg": _photo("exterior", [(DATED_DESC, "upgrade_candidate")]),
    })

    assert audit.main(["--artifacts-root", str(root)]) == 0


def test_main_rejects_missing_root(tmp_path, capsys):
    assert audit.main(["--artifacts-root", str(tmp_path / "nope")]) == 2
    assert "artifacts root not found" in capsys.readouterr().err


# ─── the absence-resolution gate ─────────────────────────────────────────────

def test_fail_on_absence_resolution_trips_when_the_catalog_still_allows_it(
    corpus, tmp_path,
):
    catalog = _catalog(tmp_path, [])

    assert audit.main([
        "--artifacts-root", str(corpus), "--catalog", str(catalog),
        "--fail-on-absence-resolution",
    ]) == 1


def test_fail_on_absence_resolution_passes_once_the_deny_term_lands(
    corpus, tmp_path,
):
    catalog = _catalog(tmp_path, ["no visible"])

    assert audit.main([
        "--artifacts-root", str(corpus), "--catalog", str(catalog),
        "--fail-on-absence-resolution",
    ]) == 0


def test_fail_on_absence_resolution_passes_on_the_shipped_catalog(corpus):
    assert audit.main([
        "--artifacts-root", str(corpus), "--fail-on-absence-resolution",
    ]) == 0


def test_fail_on_absence_resolution_passes_when_unresolved(tmp_path):
    root = tmp_path / "artifacts"
    _write_artifact(root, "p", {
        "a.jpg": _photo("exterior", [(ABSENCE_DESC, "defect_or_damage")]),
    })

    assert audit.main([
        "--artifacts-root", str(root), "--fail-on-absence-resolution",
    ]) == 0


# ─── output stability ────────────────────────────────────────────────────────

def test_json_snapshot_is_stable_and_versioned(corpus, tmp_path):
    out = tmp_path / "nested" / "snapshot.json"

    assert audit.main([
        "--artifacts-root", str(corpus), "--json", str(out),
    ]) == 0

    first = out.read_text(encoding="utf-8")
    audit.main(["--artifacts-root", str(corpus), "--json", str(out)])

    assert out.read_text(encoding="utf-8") == first
    payload = json.loads(first)
    assert payload["audit"] == "pass2c_funnel_audit_v1"
    assert payload["selection"] == "latest_run_per_property"
    assert [r["artifact"] for r in payload["artifacts"]] == sorted(
        r["artifact"] for r in payload["artifacts"]
    )


def test_report_renders_and_flags_the_forbidden_mapping(corpus, tmp_path):
    report = tmp_path / "report.md"

    audit.main(["--artifacts-root", str(corpus), "--report", str(report)])

    text = report.read_text(encoding="utf-8")
    assert "# Pass 2c funnel audit" in text
    assert "Review queue, not human truth" in text
    assert "clogged_or_damaged_gutters" in text


def test_baseline_deltas_report_movement(corpus, tmp_path):
    before = tmp_path / "before.json"
    audit.main(["--artifacts-root", str(corpus), "--json", str(before)])

    data = audit.build_audit(corpus)
    deltas = audit.compute_deltas(data, audit.load_baseline(before))

    assert deltas["available"] is True
    assert deltas["corpus"]["absence_gutter_item_resolutions"]["delta"] == 0
    assert deltas["by_scene_group"]["exterior"]["after"] == 75.0


def test_audit_never_writes_to_the_corpus(corpus):
    before = {
        p: p.read_bytes() for p in corpus.rglob("*") if p.is_file()
    }

    audit.main(["--artifacts-root", str(corpus)])

    assert {p: p.read_bytes() for p in corpus.rglob("*") if p.is_file()} == before


def test_absence_allowance_permits_documented_residuals(corpus, tmp_path):
    """The historical corpus carries accepted mixed-claim residuals; the gate
    encodes that allowance explicitly rather than being unrunnable."""
    catalog = _catalog(tmp_path, [])

    argv = [
        "--artifacts-root", str(corpus), "--catalog", str(catalog),
        "--fail-on-absence-resolution",
    ]
    assert audit.main(argv) == 1
    assert audit.main(argv + ["--max-absence-resolutions", "1"]) == 0
