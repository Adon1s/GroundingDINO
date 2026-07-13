"""
Bias Check — orchestrates two Phase D runs over the SAME cached cells to
quantify judge position bias.

What it does:
  Run 1: calls model_comparison.py with --ab-order model_a_first (full pipeline
         the first time: Phase 0 + A + B + D).
  Run 2: calls model_comparison.py with --resume --force-rejudge --ab-order
         model_b_first (reuses cells from Run 1; re-runs only Phase D).
  Diff:  loads both reports and prints per-skill flip rates + aggregate
         disagreement, then writes a bias_check_<property>_<timestamp>.json
         summary.

Both runs share Phase 0 + A + B cells, so any verdict delta is 100%
attributable to judge ordering — no contestant rerun nondeterminism in the signal.

Cost model:
  - Phase 0 (GPT fixtures): paid once (Run 2 resumes the checkpoint, Phase 0
    is fully cached).
  - Phases A + B (contestant cells): paid once. Run 2 skips them because cells
    are already checkpointed. You are NOT asked to swap LM Studio models
    a second time.
  - Phase D (judge): paid TWICE. This is the only doubled cost and it's
    exactly the point of the test.

Typical usage:
    python tools/bias_check.py --profile sol_vs_terra --property redfin_126224899 --comprehensive-judge

Pass-through flags:
    --property, --max-images, --judge-model, --comprehensive-judge,
    --concurrency, --confirm-every, --skip-skills,
    --artifacts-dir, --images-base, --model-a, --model-b, --provider-a, --provider-b

Interpretation guide (printed at the end of every run):
    flip_rate   <10%  = judge is effectively position-neutral; trust results
    flip_rate 10–20%  = mild bias; aggregate numbers are probably OK
    flip_rate 20–30%  = meaningful bias; treat per-image verdicts with caution
    flip_rate  >30%   = severe bias; aggregate numbers from random-order
                        runs may not be reliable either
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_THIS_DIR = Path(__file__).resolve().parent
_MAIN_SCRIPT = _THIS_DIR / "model_comparison.py"

# Skills we diff. Must match the main script's SKILLS tuple + comprehensive.
# Imported indirectly so a schema change in the main script doesn't silently
# break this tool.
sys.path.insert(0, str(_THIS_DIR))
from model_comparison import SKILLS, COMPREHENSIVE_SKILL  # noqa: E402


# ---------------------------------------------------------------------------
# Subprocess orchestration
# ---------------------------------------------------------------------------

def _build_main_cmd(
    args: argparse.Namespace,
    output_path: Path,
    ab_order: str,
    resume: bool,
    force_rejudge: bool,
) -> List[str]:
    """Build the argv for one invocation of model_comparison.py."""
    cmd: List[str] = [sys.executable, str(_MAIN_SCRIPT)]
    cmd += ["--profile", args.profile]

    # Pass-through flags (only include values the user actually set)
    if args.property:
        cmd += ["--property", args.property]
    if args.max_images is not None:
        cmd += ["--max-images", str(args.max_images)]
    if args.artifacts_dir:
        cmd += ["--artifacts-dir", args.artifacts_dir]
    if args.images_base:
        cmd += ["--images-base", args.images_base]
    if args.model_a:
        cmd += ["--model-a", args.model_a]
    if args.provider_a:
        cmd += ["--provider-a", args.provider_a]
    if args.model_b:
        cmd += ["--model-b", args.model_b]
    if args.provider_b:
        cmd += ["--provider-b", args.provider_b]
    if args.fixture_model:
        cmd += ["--fixture-model", args.fixture_model]
    if args.judge_model:
        cmd += ["--judge-model", args.judge_model]
    if args.judge_delay is not None:
        cmd += ["--judge-delay", str(args.judge_delay)]
    if args.skip_skills:
        cmd += ["--skip-skills", *args.skip_skills]
    if args.comprehensive_judge:
        cmd += ["--comprehensive-judge"]
    if args.confirm_every > 0:
        cmd += ["--confirm-every", str(args.confirm_every)]
    if args.concurrency is not None:
        cmd += ["--concurrency", str(args.concurrency)]

    # Bias-check-specific flags
    cmd += ["--output", str(output_path)]
    cmd += ["--ab-order", ab_order]
    if resume:
        cmd += ["--resume"]
    if force_rejudge:
        cmd += ["--force-rejudge"]
    return cmd


def _run_main(cmd: List[str], label: str) -> None:
    """Run the main script as a subprocess. Stream output live; abort on error."""
    print("\n" + "=" * 70)
    print(f"  {label}")
    print(f"  $ {' '.join(cmd)}")
    print("=" * 70 + "\n")
    # Inherit stdin so optional LM Studio load prompts work on Run 1, and
    # so --confirm-every safeguards behave normally. Inherit stdout/stderr for
    # live log streaming.
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"\n[bias_check] {label} exited with code {result.returncode}. Aborting.")
        sys.exit(result.returncode)


# ---------------------------------------------------------------------------
# Report loading + diff computation
# ---------------------------------------------------------------------------

def _load_report(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Expected report not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _winner_by_photo_and_skill(report: Dict[str, Any]) -> Dict[Tuple[str, str], Optional[str]]:
    """Return {(photo_key, skill): winner} for every image/skill present in the report.

    Missing or null verdicts map to None (not silently dropped, so the diff
    can report them accurately).
    """
    out: Dict[Tuple[str, str], Optional[str]] = {}
    for img in report.get("images") or []:
        photo_key = img.get("photo_key") or ""
        judge = img.get("judge") or {}
        for skill, verdict in judge.items():
            if isinstance(verdict, dict):
                out[(photo_key, skill)] = verdict.get("winner")
            else:
                out[(photo_key, skill)] = None
    return out


def _diff_reports(
    report_gf: Dict[str, Any],
    report_qf: Dict[str, Any],
    active_skills: List[str],
) -> Dict[str, Any]:
    """Compute per-skill flip rates and overall judge bias signal.

    A "flip" is an image where the winner differs between the two runs.
    Because the models were shown in opposite positions, a flip is evidence
    that position (not content) drove at least one of the two verdicts.

    Also tracks:
    - tie_in_both: image was called a tie regardless of position (content-driven)
    - agreed_non_tie: same non-tie winner both runs (strong content signal)
    - tie_flipped: one run called it a tie, the other picked a winner
      (weaker form of position sensitivity, counted separately from flips)
    """
    gf = _winner_by_photo_and_skill(report_gf)
    qf = _winner_by_photo_and_skill(report_qf)

    per_skill: Dict[str, Dict[str, Any]] = {}

    for skill in active_skills:
        # Images that have verdicts in BOTH reports (the only ones we can diff)
        photos_both: List[str] = []
        agreed_non_tie = 0
        tie_in_both = 0
        tie_flipped = 0    # one tie, one non-tie
        flipped = 0        # both non-tie but different winners
        missing_either = 0

        # Union of photo keys from both reports
        photo_keys = {pk for (pk, sk) in list(gf.keys()) + list(qf.keys()) if sk == skill}

        # Per-image records (so we can list out which ones flipped)
        flipped_photos: List[Dict[str, str]] = []

        for photo in sorted(photo_keys):
            w_gf = gf.get((photo, skill))
            w_qf = qf.get((photo, skill))
            if w_gf is None or w_qf is None:
                missing_either += 1
                continue
            photos_both.append(photo)

            if w_gf == "tie" and w_qf == "tie":
                tie_in_both += 1
            elif w_gf == "tie" or w_qf == "tie":
                tie_flipped += 1
            elif w_gf == w_qf:
                agreed_non_tie += 1
            else:
                flipped += 1
                flipped_photos.append({
                    "photo_key": photo,
                    "model_a_first_winner": w_gf,
                    "model_b_first_winner": w_qf,
                })

        n_both = len(photos_both)
        flip_rate = (flipped / n_both) if n_both else 0.0
        tie_flip_rate = (tie_flipped / n_both) if n_both else 0.0
        agreement_rate = ((agreed_non_tie + tie_in_both) / n_both) if n_both else 0.0

        # Severity band for the flip rate
        if flip_rate < 0.10:
            severity = "neutral"
        elif flip_rate < 0.20:
            severity = "mild"
        elif flip_rate < 0.30:
            severity = "meaningful"
        else:
            severity = "severe"

        per_skill[skill] = {
            "n_comparable": n_both,
            "agreed_non_tie": agreed_non_tie,
            "tie_in_both": tie_in_both,
            "tie_flipped": tie_flipped,
            "flipped": flipped,
            "missing_either": missing_either,
            "flip_rate": round(flip_rate, 3),
            "tie_flip_rate": round(tie_flip_rate, 3),
            "agreement_rate": round(agreement_rate, 3),
            "severity": severity,
            "flipped_photos": flipped_photos,
        }

    # Overall (union across skills)
    totals = {"n_comparable": 0, "flipped": 0, "tie_flipped": 0,
              "agreed_non_tie": 0, "tie_in_both": 0, "missing_either": 0}
    for s in per_skill.values():
        for k in totals:
            totals[k] += s[k]
    overall_flip_rate = (totals["flipped"] / totals["n_comparable"]) if totals["n_comparable"] else 0.0
    overall_agreement = (
        (totals["agreed_non_tie"] + totals["tie_in_both"]) / totals["n_comparable"]
    ) if totals["n_comparable"] else 0.0

    if overall_flip_rate < 0.10:
        overall_severity = "neutral"
    elif overall_flip_rate < 0.20:
        overall_severity = "mild"
    elif overall_flip_rate < 0.30:
        overall_severity = "meaningful"
    else:
        overall_severity = "severe"

    return {
        "per_skill": per_skill,
        "overall": {
            **totals,
            "flip_rate": round(overall_flip_rate, 3),
            "agreement_rate": round(overall_agreement, 3),
            "severity": overall_severity,
        },
    }


# ---------------------------------------------------------------------------
# Presentation
# ---------------------------------------------------------------------------

_SEVERITY_DESCRIPTIONS = {
    "neutral":    "judge is effectively position-neutral; trust verdicts",
    "mild":       "some position bias, but aggregate numbers are probably OK",
    "meaningful": "noticeable bias; treat per-image verdicts with caution",
    "severe":     "strong position bias; aggregate numbers are unreliable too",
}


def _print_diff_table(diff: Dict[str, Any]) -> None:
    print("\n" + "=" * 78)
    print("  BIAS CHECK RESULTS")
    print("=" * 78)
    print("  Each row: how did the judge's winner choice change when we flipped")
    print("  which model appeared first in the prompt?")
    print()
    hdr = (f"  {'skill':<18} {'n':>4} {'agreed':>8} {'flipped':>8} "
           f"{'tie-flip':>9} {'flip %':>8}  severity")
    print(hdr)
    print(f"  {'-'*18} {'-'*4} {'-'*8} {'-'*8} {'-'*9} {'-'*8}  {'-'*12}")
    for skill, s in diff["per_skill"].items():
        print(
            f"  {skill:<18} {s['n_comparable']:>4} "
            f"{s['agreed_non_tie'] + s['tie_in_both']:>8} "
            f"{s['flipped']:>8} {s['tie_flipped']:>9} "
            f"{s['flip_rate']*100:>7.1f}%  {s['severity']}"
        )
    o = diff["overall"]
    print(f"  {'-'*18} {'-'*4} {'-'*8} {'-'*8} {'-'*9} {'-'*8}  {'-'*12}")
    print(
        f"  {'OVERALL':<18} {o['n_comparable']:>4} "
        f"{o['agreed_non_tie'] + o['tie_in_both']:>8} "
        f"{o['flipped']:>8} {o['tie_flipped']:>9} "
        f"{o['flip_rate']*100:>7.1f}%  {o['severity']}"
    )
    print()
    print(f"  Interpretation: {_SEVERITY_DESCRIPTIONS[o['severity']]}")
    print()
    print("  Columns:")
    print("    agreed    = same winner both runs (incl. tie-in-both): content-driven")
    print("    flipped   = different non-tie winners across runs: position-driven")
    print("    tie-flip  = tie in one run, winner in the other: weaker bias signal")
    print("    flip %    = flipped / n_comparable")
    print("=" * 78)


def _print_flipped_photos(diff: Dict[str, Any], max_per_skill: int = 5) -> None:
    """List a few example flipped images per skill for manual inspection."""
    any_flips = any(s["flipped"] > 0 for s in diff["per_skill"].values())
    if not any_flips:
        return
    print("\n  Example flipped images (max {} per skill):".format(max_per_skill))
    for skill, s in diff["per_skill"].items():
        if not s["flipped_photos"]:
            continue
        print(f"    {skill}:")
        for fp in s["flipped_photos"][:max_per_skill]:
            print(f"      {fp['photo_key']}: "
                  f"model_a-first→{fp['model_a_first_winner']}, "
                  f"model_b-first→{fp['model_b_first_winner']}")
        remaining = len(s["flipped_photos"]) - max_per_skill
        if remaining > 0:
            print(f"      ... {remaining} more (see summary JSON)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run two Phase D passes over the same cells (model_a_first + model_b_first) "
            "and diff the results to measure judge position bias."
        ),
    )

    # All flags below are passed through to model_comparison.py verbatim.
    parser.add_argument("--profile", required=True, help="Comparison profile name or JSON path.")
    parser.add_argument("--property", default=None,
                        help="Filter to a single property ID.")
    parser.add_argument("--max-images", type=int, default=None,
                        help="Cap on images processed.")
    parser.add_argument("--artifacts-dir", default=None)
    parser.add_argument("--images-base", default=None)
    parser.add_argument("--model-a", default=None)
    parser.add_argument("--provider-a", choices=("openai", "lmstudio"), default=None)
    parser.add_argument("--model-b", default=None)
    parser.add_argument("--provider-b", choices=("openai", "lmstudio"), default=None)
    parser.add_argument("--fixture-model", default=None)
    parser.add_argument("--judge-model", default=None,
                        help="Judge model. Recommended: gpt-5.4-mini for bias check "
                             "(cheap enough to afford two Phase D runs).")
    parser.add_argument("--judge-delay", type=float, default=None,
                        help="Seconds between judge calls (default: main script's default).")
    parser.add_argument("--skip-skills", nargs="*", default=[],
                        help="Skills to skip entirely. Must match main script's skill names.")
    parser.add_argument("--comprehensive-judge", action="store_true",
                        help="Include the end-to-end comprehensive judge call.")
    parser.add_argument("--confirm-every", type=int, default=0,
                        help="Pause for confirmation every N images in API phases.")
    parser.add_argument("--concurrency", type=int, default=None,
                        help="Max concurrent images for API phases.")

    # Bias-check-specific flags
    parser.add_argument(
        "--output-dir",
        default=".",
        help="Directory for the two run reports and the bias_check summary JSON. "
             "Default: current directory.",
    )
    parser.add_argument(
        "--tag",
        default=None,
        help="Optional tag included in output filenames (e.g. 'mini_judge'). "
             "Default: just the timestamp.",
    )
    parser.add_argument(
        "--skip-run-1",
        action="store_true",
        help="Skip Run 1 (model_a_first) — assumes its output already exists at "
             "the expected path. Useful if Run 1 completed but Run 2 failed.",
    )
    return parser.parse_args()


def _build_output_paths(args: argparse.Namespace) -> Tuple[Path, Path, Path]:
    """Return (run1_path, run2_path, summary_path)."""
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.property or "all"
    tag = f"_{args.tag}" if args.tag else ""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = f"bias_check_{stem}{tag}_{ts}"
    run1 = out_dir / f"{base}__run1_model_a_first.json"
    run2 = out_dir / f"{base}__run2_model_b_first.json"
    summary = out_dir / f"{base}__summary.json"
    return run1, run2, summary


def main() -> None:
    args = parse_args()

    if not _MAIN_SCRIPT.exists():
        print(f"[bias_check] model_comparison.py not found at {_MAIN_SCRIPT}")
        sys.exit(1)

    run1_path, run2_path, summary_path = _build_output_paths(args)

    print("=" * 70)
    print("  BIAS CHECK")
    print("=" * 70)
    print(f"  Property:         {args.property or '(all)'}")
    print(f"  Run 1 output:     {run1_path}")
    print(f"  Run 2 output:     {run2_path}")
    print(f"  Summary output:   {summary_path}")
    if args.judge_model:
        print(f"  Judge override:   {args.judge_model}")
    if args.comprehensive_judge:
        print("  Comprehensive:    ENABLED")
    print("=" * 70)

    started = time.time()

    # ── Run 1: model_a_first (full pipeline; cells + judge) ────────────────
    if args.skip_run_1:
        if not run1_path.exists():
            print(f"[bias_check] --skip-run-1 set but {run1_path} does not exist.")
            sys.exit(1)
        print(f"\n[bias_check] Skipping Run 1 (reusing existing {run1_path})")
    else:
        cmd1 = _build_main_cmd(
            args,
            output_path=run1_path,
            ab_order="model_a_first",
            resume=False,
            force_rejudge=False,
        )
        _run_main(cmd1, "Run 1: --ab-order model_a_first (full pipeline)")

    # ── Run 2: model_b_first, resuming the Run 1 checkpoint, rejudge only ──
    # Copy the exact schema-versioned checkpoint so Run 2 can use a separate
    # report path while reusing identical fixture and contestant cells.
    run1_checkpoint = run1_path.with_suffix(".checkpoint.json")
    run2_checkpoint = run2_path.with_suffix(".checkpoint.json")
    if not run1_checkpoint.exists():
        print(f"[bias_check] Run 1 checkpoint not found: {run1_checkpoint}")
        sys.exit(1)
    shutil.copy2(run1_checkpoint, run2_checkpoint)
    cmd2 = _build_main_cmd(
        args,
        output_path=run2_path,
        ab_order="model_b_first",
        resume=True,
        force_rejudge=True,
    )
    _run_main(cmd2, "Run 2: --ab-order model_b_first (reuses Run 1 cells; rejudges only)")

    # ── Diff ─────────────────────────────────────────────────────────────
    report_gf = _load_report(run1_path)
    report_qf = _load_report(run2_path)

    # Determine active skills from the reports (respects --skip-skills).
    meta = report_gf.get("meta") or {}
    active_skills: List[str] = list(meta.get("skills_tested") or SKILLS)
    if meta.get("comprehensive_judge") and COMPREHENSIVE_SKILL not in active_skills:
        active_skills.append(COMPREHENSIVE_SKILL)

    diff = _diff_reports(report_gf, report_qf, active_skills)

    # ── Write summary JSON ───────────────────────────────────────────────
    summary = {
        "timestamp": datetime.now().isoformat(),
        "property": args.property,
        "run1_report": str(run1_path),
        "run2_report": str(run2_path),
        "run1_ab_order": "model_a_first",
        "run2_ab_order": "model_b_first",
        "run1_judge_model": ((report_gf.get("meta") or {}).get("judge") or {}).get("model"),
        "run2_judge_model": ((report_qf.get("meta") or {}).get("judge") or {}).get("model"),
        "n_images_run1": (report_gf.get("meta") or {}).get("n_images"),
        "n_images_run2": (report_qf.get("meta") or {}).get("n_images"),
        "active_skills": active_skills,
        "diff": diff,
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # ── Present ──────────────────────────────────────────────────────────
    _print_diff_table(diff)
    _print_flipped_photos(diff)

    elapsed = time.time() - started
    print(f"\n  Summary saved:    {summary_path}")
    print(f"  Run 1 report:     {run1_path}")
    print(f"  Run 2 report:     {run2_path}")
    print(f"  Total elapsed:    {elapsed:.1f}s")


if __name__ == "__main__":
    main()