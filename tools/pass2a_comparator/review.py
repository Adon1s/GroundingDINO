"""Blinding, verdict mapping and the directional summary.

The comparator is a human instrument, so the only guards that matter here are
the ones protecting the human: which side is which stays hidden until every
photo has a verdict, and the blind decisions are snapshotted at reveal so a
later revision cannot quietly rewrite what was decided blind.

Nothing in this module calls a model or reads an image.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tools.comparison_common import atomic_json
from tools.pass2a_comparator import config as cc
from tools.pass2a_comparator import storage as store

#: Value recorded per photo in the side map: which source is shown as side A.
CANDIDATE = "candidate"
BASELINE = "baseline"

#: Candidate-relative outcomes, after identities are known.
BETTER, SAME, WORSE, UNCLEAR = "better", "same", "worse", "unclear"

POSITIVE = "positive"
NEGATIVE = "negative"
NO_SIGNAL = "no clear signal"


# ---------------------------------------------------------------------------
# Blinding
# ---------------------------------------------------------------------------

def assign_sides(exp_id: str, photos: List[store.Photo]) -> Dict[str, str]:
    """Which source occupies side A, per photo: deterministic and balanced.

    Derived from the experiment id alone, so a resume - or a fresh process
    reopening the experiment - reproduces the same assignment without storing
    it. Balanced to within one photo so neither side is systematically first.
    All three repeats of a photo stay on that photo's side; the unit of
    assignment is the photo, never the repeat.
    """
    total = len(photos)
    order = sorted(
        range(total),
        key=lambda i: hashlib.sha256(f"{exp_id}|{i}".encode("utf-8")).hexdigest(),
    )
    candidate_first = set(order[: (total + 1) // 2])
    return {
        photo.key: (CANDIDATE if index in candidate_first else BASELINE)
        for index, photo in enumerate(photos)
    }


def side_outputs(
    photo: store.Photo,
    side_a_source: str,
    baseline_dir: Path,
    experiment_dir_path: Path,
    repeats: int = cc.REPEATS,
) -> Tuple[List[str], List[str]]:
    """(side_a_texts, side_b_texts) - three repeats each, grouped by side."""
    dirs = {BASELINE: baseline_dir, CANDIDATE: experiment_dir_path}
    side_b_source = BASELINE if side_a_source == CANDIDATE else CANDIDATE

    def _texts(source: str) -> List[str]:
        out: List[str] = []
        for rep in range(1, repeats + 1):
            record = store.load_call(store.call_path(dirs[source], photo, rep))
            out.append((record or {}).get("text") or "")
        return out

    return _texts(side_a_source), _texts(side_b_source)


# ---------------------------------------------------------------------------
# Review state
# ---------------------------------------------------------------------------

def review_path(experiment_dir_path: Path) -> Path:
    return experiment_dir_path / "review.json"


def empty_review() -> Dict[str, Any]:
    return {
        "decisions": {},
        "blind_decisions": None,
        "revealed_at": None,
        "final_verdict": None,
    }


def load_review(experiment_dir_path: Path) -> Dict[str, Any]:
    record = store.load_call(review_path(experiment_dir_path))
    if not record:
        return empty_review()
    base = empty_review()
    base.update(record)
    return base


def save_review(experiment_dir_path: Path, review: Dict[str, Any]) -> None:
    atomic_json(review_path(experiment_dir_path), review)


def record_decision(
    review: Dict[str, Any],
    photo_key: str,
    *,
    verdict: str,
    tags: Optional[List[str]] = None,
    note: str = "",
    critical_regression: bool = False,
) -> Dict[str, Any]:
    """Set one photo's verdict. Valid before and after reveal."""
    if verdict not in cc.BLIND_VERDICTS:
        raise ValueError(f"unknown verdict {verdict!r}; expected one of {cc.BLIND_VERDICTS}")
    unknown = sorted(set(tags or []) - set(cc.REASON_TAGS))
    if unknown:
        raise ValueError(f"unknown reason tags: {unknown}")
    review["decisions"][photo_key] = {
        "verdict": verdict,
        "tags": list(tags or []),
        "note": note,
        "critical_regression": bool(critical_regression),
        "decided_at": store.now_iso(),
    }
    return review


def is_complete(review: Dict[str, Any], photos: List[store.Photo]) -> bool:
    decisions = review.get("decisions") or {}
    return all(photo.key in decisions for photo in photos)


def is_revealed(review: Dict[str, Any]) -> bool:
    return bool(review.get("revealed_at"))


def reveal(review: Dict[str, Any], photos: List[store.Photo]) -> Dict[str, Any]:
    """Snapshot the blind decisions and unlock identities.

    Refuses while any photo is unreviewed - that refusal is the whole point of
    the blind phase. Re-revealing is a no-op, so the snapshot can never be
    overwritten by a later, informed decision.
    """
    if not is_complete(review, photos):
        missing = [p.key for p in photos if p.key not in (review.get("decisions") or {})]
        raise ValueError(f"cannot reveal: {len(missing)} photo(s) unreviewed: {missing[:3]}")
    if is_revealed(review):
        return review
    review["blind_decisions"] = {
        key: dict(value) for key, value in (review.get("decisions") or {}).items()
    }
    review["revealed_at"] = store.now_iso()
    return review


def record_final_verdict(
    review: Dict[str, Any], verdict: str, note: str = ""
) -> Dict[str, Any]:
    if verdict not in cc.FINAL_VERDICTS:
        raise ValueError(f"unknown final verdict {verdict!r}; expected {cc.FINAL_VERDICTS}")
    review["final_verdict"] = {
        "verdict": verdict,
        "note": note,
        "recorded_at": store.now_iso(),
    }
    return review


# ---------------------------------------------------------------------------
# Vote mapping + directional summary
# ---------------------------------------------------------------------------

def map_vote(verdict: str, side_a_source: str) -> str:
    """Blinded A/B verdict -> candidate-relative outcome."""
    if verdict == "same":
        return SAME
    if verdict == "unclear":
        return UNCLEAR
    chose_a = verdict == "A better"
    candidate_won = chose_a == (side_a_source == CANDIDATE)
    return BETTER if candidate_won else WORSE


def revisions(review: Dict[str, Any]) -> List[str]:
    """Photos whose verdict changed after the reveal. Empty before reveal."""
    blind = review.get("blind_decisions")
    if not blind:
        return []
    live = review.get("decisions") or {}
    return sorted(
        key for key in blind
        if (live.get(key) or {}).get("verdict") != blind[key].get("verdict")
    )


def summarize(
    review: Dict[str, Any], sides: Dict[str, str], photos: List[store.Photo]
) -> Dict[str, Any]:
    """Counts, per-photo outcomes and the directional indicator.

    The indicator is deliberately blunt: it reports direction, never magnitude
    or significance. 15 photos and one reviewer cannot support more than that.
    """
    decisions = review.get("decisions") or {}
    complete = is_complete(review, photos)
    counts = {BETTER: 0, SAME: 0, WORSE: 0, UNCLEAR: 0}
    per_photo: List[Dict[str, Any]] = []
    critical: List[str] = []

    for photo in photos:
        decision = decisions.get(photo.key)
        if not decision:
            continue
        outcome = map_vote(decision["verdict"], sides[photo.key])
        counts[outcome] += 1
        if decision.get("critical_regression"):
            critical.append(photo.key)
        per_photo.append(
            {
                "photo": photo.key,
                "scene": photo.scene,
                "side_a_source": sides[photo.key],
                "blind_verdict": decision["verdict"],
                "outcome": outcome,
                "tags": decision.get("tags") or [],
                "note": decision.get("note") or "",
                "critical_regression": bool(decision.get("critical_regression")),
            }
        )

    if not complete:
        directional, reason = NO_SIGNAL, "reviews incomplete"
    elif critical:
        directional, reason = NEGATIVE, f"critical regression on {len(critical)} photo(s)"
    elif counts[WORSE] > counts[BETTER]:
        directional, reason = NEGATIVE, f"worse {counts[WORSE]} > better {counts[BETTER]}"
    elif counts[BETTER] > counts[WORSE]:
        directional, reason = POSITIVE, f"better {counts[BETTER]} > worse {counts[WORSE]}"
    else:
        directional, reason = NO_SIGNAL, f"better and worse tied at {counts[BETTER]}"

    return {
        "complete": complete,
        "reviewed": len(per_photo),
        "photo_count": len(photos),
        "counts": counts,
        "critical_regressions": critical,
        "directional": directional,
        "directional_reason": reason,
        "per_photo": per_photo,
        "post_reveal_revisions": revisions(review),
    }


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def build_report(
    experiment: Dict[str, Any],
    review: Dict[str, Any],
    sides: Dict[str, str],
    photos: List[store.Photo],
) -> Dict[str, Any]:
    summary = summarize(review, sides, photos)
    return {
        "generated_at": store.now_iso(),
        "experiment_id": experiment["experiment_id"],
        "display_name": experiment.get("display_name"),
        "baseline_id": experiment.get("baseline_id"),
        "git_head": store.git_head(),
        "runtime": experiment.get("runtime"),
        "candidate_system_prompt": experiment.get("candidate_system_prompt"),
        "candidate_user_prompt": experiment.get("candidate_user_prompt"),
        "candidate_system_prompt_sha256": experiment.get("candidate_system_prompt_sha256"),
        "candidate_user_prompt_sha256": experiment.get("candidate_user_prompt_sha256"),
        "fingerprint": experiment.get("fingerprint"),
        "scene_conditional": bool(experiment.get("scene_conditional")),
        "revealed_at": review.get("revealed_at"),
        "summary": summary,
        "final_verdict": review.get("final_verdict"),
        "blind_decisions": review.get("blind_decisions"),
        # Always false: the 8000-token cap is not the production 2000-token cap.
        "production_equivalent": False,
        "production_equivalent_reason": cc.PRODUCTION_EQUIVALENT_REASON,
    }


def render_report_md(report: Dict[str, Any]) -> str:
    """Render the same dict the json carries. Never a separate computation."""
    summary = report["summary"]
    counts = summary["counts"]
    final = report.get("final_verdict") or {}
    runtime = report.get("runtime") or {}
    lines: List[str] = [
        f"# Pass 2a prompt comparison - {report.get('display_name')}",
        "",
        f"Experiment `{report['experiment_id']}` vs baseline `{report.get('baseline_id')}` "
        f"- generated {report['generated_at']} at `{report.get('git_head', '')[:12]}`",
        "",
        "> **Not production-equivalent.** " + report["production_equivalent_reason"],
        "",
    ]
    if report.get("scene_conditional"):
        lines += [
            "> **Scene-conditional prompt.** It renders `{scene}` from the frozen "
            "Pass 1a capture. Production `run_pass_2a` receives the scene in "
            "`context` but never reads it, so shipping this prompt needs that "
            "wiring too - a second change beyond the token cap.",
            "",
        ]
    lines += [
        "## Verdict",
        "",
        f"- Final verdict: **{final.get('verdict', 'not recorded')}**",
    ]
    if final.get("note"):
        lines.append(f"- Note: {final['note']}")
    lines += [
        f"- Directional indicator: **{summary['directional']}** ({summary['directional_reason']})",
        f"- Reviewed: {summary['reviewed']}/{summary['photo_count']}"
        + ("" if summary["complete"] else " - **incomplete**"),
        "",
        "## Candidate-relative outcomes",
        "",
        "| better | same | worse | unclear |",
        "|---:|---:|---:|---:|",
        f"| {counts[BETTER]} | {counts[SAME]} | {counts[WORSE]} | {counts[UNCLEAR]} |",
        "",
    ]
    if summary["critical_regressions"]:
        lines += [
            "**Critical regressions flagged:** "
            + ", ".join(f"`{key}`" for key in summary["critical_regressions"]),
            "",
        ]
    if summary["post_reveal_revisions"]:
        lines += [
            "**Verdicts revised after reveal:** "
            + ", ".join(f"`{key}`" for key in summary["post_reveal_revisions"])
            + ". The blind record is preserved in `report.json.blind_decisions`.",
            "",
        ]
    lines += [
        "## Runtime",
        "",
        f"- model `{runtime.get('model')}`, reasoning `{runtime.get('reasoning_effort')}`, "
        f"max output {runtime.get('max_output_tokens')}, image detail "
        f"`{runtime.get('image_detail')}`, {runtime.get('repeats')} repeats",
        "",
        "## Per photo",
        "",
        "| photo | scene | side A | blind verdict | outcome | tags |",
        "|---|---|---|---|---|---|",
    ]
    for row in summary["per_photo"]:
        flag = " **!**" if row["critical_regression"] else ""
        lines.append(
            f"| `{row['photo']}` | {row['scene']} | {row['side_a_source']} | "
            f"{row['blind_verdict']} | {row['outcome']}{flag} | "
            f"{', '.join(row['tags']) or '-'} |"
        )
    lines += [
        "",
        "## Candidate prompts",
        "",
        "### System",
        "",
        "```",
        report.get("candidate_system_prompt") or "",
        "```",
        "",
        "### User",
        "",
        "```",
        report.get("candidate_user_prompt") or "",
        "```",
        "",
    ]
    return "\n".join(lines)


def write_report(
    experiment_dir_path: Path,
    experiment: Dict[str, Any],
    review: Dict[str, Any],
    sides: Dict[str, str],
    photos: List[store.Photo],
) -> Dict[str, Any]:
    report = build_report(experiment, review, sides, photos)
    atomic_json(experiment_dir_path / "report.json", report)
    (experiment_dir_path / "report.md").write_text(
        render_report_md(report), encoding="utf-8"
    )
    return report
