"""Session B (P5 item 5): multi-surrogate packages are a disagreement tier,
never Tier-1 auto-accepted key migrations. The cluster script targets the
frozen Session 9 report shape (type|unit keys, status="applied" v4 rows)."""
import json

from scripts.cluster_session9_reviews import main


def _package_item(review_id, prop, key, baseline=None, candidate=None):
    return {
        "review_id": review_id,
        "property_key": prop,
        "category": "package",
        "key": key,
        "baseline": baseline,
        "candidate": candidate,
    }


def _v4_row(ptype, tier="refresh"):
    return {"package_type": ptype, "estimate_unit_id": "",
            "pricing_tier": tier, "status": "applied"}


def _v5_row(ptype, unit, tier="refresh"):
    return {"package_type": ptype, "estimate_unit_id": unit,
            "pricing_tier": tier, "decision": "approve",
            "status": "applied", "reason_code": "approved_absorbs_children"}


def _run(tmp_path, items):
    report = tmp_path / "report.json"
    report.write_text(json.dumps({"review_items": items}), encoding="utf-8")
    template = tmp_path / "template.json"
    template.write_text(
        json.dumps({
            "schema_version": 1,
            "reviews": {
                item["review_id"]: {"decision": "", "explanation": ""}
                for item in items
            },
        }),
        encoding="utf-8",
    )
    out = tmp_path / "out.json"
    digest = tmp_path / "digest.md"
    assert main([
        "--report", str(report), "--template", str(template),
        "--canary-root", str(tmp_path / "canary"),
        "--digest", str(digest), "--out", str(out),
    ]) == 0
    return (
        json.loads(out.read_text(encoding="utf-8")),
        digest.read_text(encoding="utf-8"),
    )


def test_multi_surrogate_packages_land_in_disagreement_tier_unfilled(tmp_path):
    items = [
        _package_item("r_v4", "prop", "bathroom_modernization|",
                      baseline=_v4_row("bathroom_modernization")),
        _package_item("r_v5_a", "prop", "bathroom_modernization|bath_1",
                      candidate=_v5_row("bathroom_modernization", "bath_1")),
        _package_item("r_v5_b", "prop", "bathroom_modernization|bath_2",
                      candidate=_v5_row("bathroom_modernization", "bath_2")),
    ]
    prefilled, digest = _run(tmp_path, items)
    assert all(not row["decision"] for row in prefilled["reviews"].values())
    assert "package_multi_surrogate_disagreement - 3 items" in digest
    assert "package_key_migration" not in digest


def test_single_mate_still_pairs_as_key_migration(tmp_path):
    items = [
        _package_item("r_v4", "prop", "kitchen_modernization|",
                      baseline=_v4_row("kitchen_modernization")),
        _package_item("r_v5", "prop", "kitchen_modernization|kitchen_1",
                      candidate=_v5_row("kitchen_modernization", "kitchen_1")),
    ]
    prefilled, digest = _run(tmp_path, items)
    decisions = {
        rid: row["decision"] for rid, row in prefilled["reviews"].items()
    }
    assert decisions == {"r_v4": "accepted", "r_v5": "accepted"}
    assert "package_key_migration - 2 items" in digest
