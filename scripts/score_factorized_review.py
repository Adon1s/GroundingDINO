"""Score the factorized verifier replay against the v1.1 human labels.

Joins a --factorized harness output root (scripts/
redecide_renovation_architecture.py) to reports/labels_v1_1.json by
(property_key, condition_id), and to the stored canary envelopes for the
label-free full-population sweep.

The gates in GATES are pre-committed in
docs/HANDOFF_factorized_verifier_replay.md §7 and ratified before the live run.
Change a threshold only by changing that doc in the same commit.

Two things this scorer does deliberately, both fixing noted defects in
scripts/score_redecide_variants.py:
  - it excludes production LOUDLY (printed and recorded), rather than silently;
  - it reports G1/G2 next to the label-free degeneracy checks, because an
    always-`yes` verifier passes both of those gates on its own.

Usage:
  .venv\\Scripts\\python.exe scripts\\score_factorized_review.py \
      --arm-root artifacts_canary\\factorized_v1_20260831
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from tools import label_schema as ls  # noqa: E402
from tools.comparison_common import sha256_file  # noqa: E402
from tools.renovation_architecture.factorized_review import (  # noqa: E402
    DERIVED_CLASSES,
    FACTOR_KEYS,
)

ENVELOPE_KEY = "renovation_estimate_v5"
EXPLORATORY_MIN_JUDGED = 10
# Column order for the per-axis tables: worst-to-best reads left to right.
FACTOR_VALUES_ORDER = ("no", "unclear", "yes")
NOISE_FLOOR_NOTE = (
    "Noise floor: Terra replica flips ran 6.4% (16/250) and humans matched "
    "run_1/run_2 8:6 — read every factorized-vs-stored delta against the "
    "replica column, never against zero."
)

# The label vocabulary separates wrong_object_or_place from absent; the
# derivation emits one folded `absent` class (label_schema.CLAIM_AXIS maps
# both the same way).
LABEL_SLUG_TO_CLASS = {
    "exact_and_warranted": "exact_and_warranted",
    "misnamed_but_warranted": "misnamed_but_warranted",
    "exact_but_trivial": "exact_but_trivial",
    "misnamed_and_trivial": "misnamed_and_trivial",
    "wrong_object_or_place": "absent",
    "absent": "absent",
    "inconclusive": "inconclusive",
}

# Suppression = a derived class that would stop the work under any downstream
# policy. `misnamed_but_warranted` is NOT suppression: misnamed conditions
# route to remapping, keeping the work. It is still a label error and is
# reported separately.
SUPPRESSING_CLASSES = ("absent", "exact_but_trivial", "misnamed_and_trivial")

GATES = {
    "G1": {
        "title": "false suppression",
        "klass": "supported_billed",
        "n": 57,
        "rule": "derived class suppresses work",
        "cmp": "<=",
        "threshold": 5,
        "status": "decisive",
        "basis": "8.8% — just above the 6.4% replica noise floor, which a "
                 "different prompt inherits at minimum.",
    },
    "G2": {
        "title": "recovery of Terra's false rejections",
        "klass": "dirB_recovery",
        "n": 16,
        "rule": "visible == yes",
        "cmp": ">=",
        "threshold": 11,
        "status": "decisive",
        "basis": "humans confirmed these present and exactly named; the "
                 "hypothesis is that Terra's single verdict conflated "
                 "'not visible' with 'the wording does not fit'.",
    },
    "G3a": {
        "title": "misnamed catch",
        "klass": "misnamed_billed",
        "n": 7,
        "rule": "claim_accurate_as_written == no",
        "cmp": ">=",
        "threshold": 4,
        "status": "exploratory",
        "basis": "n=7 is below the 10-judgment exploratory floor; decisive "
                 "only after the L2 rescore.",
    },
    "G3b": {
        "title": "trivial catch",
        "klass": "trivial_billed",
        "n": 5,
        "rule": "material_enough_for_work == no",
        "cmp": ">=",
        "threshold": 3,
        "status": "exploratory",
        "basis": "n=5; the category no verdict-flip arm can measure at all.",
    },
    "G3c": {
        "title": "absent catch",
        "klass": "hard_false_billed",
        "n": 3,
        "rule": "visible == no",
        "cmp": ">=",
        "threshold": 2,
        "status": "directional",
        "basis": "n=3 is the population that blocked gate 2b for "
                 "verdict-flip arms; directional here by construction.",
    },
}

# Label-free, computed over every replayed condition. These exist because G1
# and G2 are BOTH passed by a degenerate verifier that answers `yes` to
# everything, and the category that would catch it (G3) is underpowered.
G4_CHECKS = {
    "exact_and_warranted_share_max": 0.90,
    "misnamed_share_min": 0.03,
    "unclear_share_max": 0.10,
}

NOT_SHOWN = [
    "Production listings. This scores the canary only (see §inputs for the "
    "excluded count): production contributes one misnamed and one trivial "
    "card, not enough to pay for a second input path.",
    "A production error rate. The labeled cards were sampled "
    "disagreements-first, so catch rates sit on a hard, non-representative "
    "slice, and the full-sweep class distribution is not a population rate.",
    "Specificity on Terra's correct rejections. There is no dirB loss "
    "population (`dirB_terra_correct` = 0 cards), so G2 measures recovery "
    "with no paired over-revival check until Session L2 adds one.",
    "Anything downstream: Sol, packages, pricing, dispositions, or whether a "
    "trivial condition should be absorbed by turnover rather than dropped.",
]


# ------------------------------------------------------------------ loading

def load_canary_labels(
    labels_path: Path, *, allow_partial: bool = False
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    doc = json.loads(labels_path.read_text(encoding="utf-8"))
    if doc.get("label_version") != ls.LABEL_VERSION:
        raise SystemExit(
            f"{labels_path} is label_version {doc.get('label_version')!r}, "
            f"this scorer speaks {ls.LABEL_VERSION!r}"
        )
    rows, pending, excluded = [], [], 0
    for origin, row in sorted(doc["labels"].items()):
        if row.get("source") != "canary":
            excluded += 1
            continue
        if not row.get("slug"):
            pending.append(origin)
            continue
        rows.append({
            "card_id": origin,
            "property_key": row["property_key"],
            "condition_id": row["condition_id"],
            "stored_verdict": row["stored_terra_verdict"],
            "slug": row["slug"],
            "claim": row["claim"],
            "work": row["work"],
            "klass": row["class_v1_1"],
        })
    if pending and not allow_partial:
        raise SystemExit(
            f"{len(pending)} canary cards are not adjudicated yet "
            f"(e.g. {pending[:3]}). Pass --allow-partial to score anyway and "
            "read the result as provisional."
        )
    meta = {"excluded_production_cards": excluded, "pending_cards": len(pending)}
    return rows, meta


def load_arm(arm_root: Path) -> Dict[str, Any]:
    manifest = json.loads((arm_root / "manifest.json").read_text(encoding="utf-8"))
    answers: Dict[Tuple[str, str], Dict[str, Any]] = {}
    statuses: Counter = Counter()
    for unit_file in sorted(arm_root.glob("*/units/terra_unit_*.json")):
        record = json.loads(unit_file.read_text(encoding="utf-8"))
        prop = str(record.get("property_key") or "")
        statuses[str(record.get("status") or "")] += 1
        for condition_id, answer in (record.get("reviews") or {}).items():
            answers[(prop, str(condition_id))] = answer
    return {
        "label": str((manifest.get("variant") or {}).get("label") or arm_root.name),
        "dry_run": bool(manifest.get("dry_run")),
        "response_contract": str(manifest.get("response_contract") or ""),
        "prompt_version": str(manifest.get("prompt_version") or ""),
        "answers": answers,
        "statuses": dict(statuses),
    }


def _envelope(run_dir: Path) -> Optional[Dict[str, Any]]:
    art = json.loads((run_dir / "photo_intel_debug.json").read_text(encoding="utf-8"))
    env = art.get(ENVELOPE_KEY) or (art.get("analysis_debug") or {}).get(ENVELOPE_KEY)
    return env if isinstance(env, dict) and env.get("state") == "complete" else None


def load_stored_verdicts(candidate_root: Path) -> Dict[str, Dict[Any, str]]:
    """Stored Terra verdicts under two keys.

    `by_cid` is (property_key, condition_id) — the join the arm uses, since the
    arm replayed run_1's own conditions.

    `by_key` is (property_key, catalog_item_id, estimate_unit_id) — the SEMANTIC
    identity, needed to compare replicas at all: condition_id derives from
    estimate_id, which includes source_run_id, so run_1 and run_2 share zero
    condition_ids for the same logical condition (verified: 0 overlap, vs 702
    under this key). `cid_to_key` carries the arm across to that space.
    """
    by_cid: Dict[Any, str] = {}
    by_key: Dict[Any, str] = {}
    cid_to_key: Dict[Any, Any] = {}
    if not candidate_root.is_dir():
        return {"by_cid": by_cid, "by_key": by_key, "cid_to_key": cid_to_key}
    for prop in sorted(p for p in candidate_root.iterdir() if p.is_dir()):
        runs = sorted(
            r for r in prop.iterdir()
            if r.is_dir() and (r / "photo_intel_debug.json").is_file()
        )
        if not runs:
            continue
        env = _envelope(runs[-1])
        if env is None:
            continue
        identity = {
            str(c["condition_id"]): (
                prop.name, str(c["catalog_item_id"]), str(c["estimate_unit_id"])
            )
            for c in env["result"]["observed_conditions"]
        }
        for review in env["result"]["condition_reviews"]:
            cid = str(review["condition_id"])
            verdict = str(review["verdict"])
            by_cid[(prop.name, cid)] = verdict
            key = identity.get(cid)
            if key is not None:
                by_key[key] = verdict
                cid_to_key[(prop.name, cid)] = key
    return {"by_cid": by_cid, "by_key": by_key, "cid_to_key": cid_to_key}


# ----------------------------------------------------------------- scoring

def _fires(answer: Mapping[str, Any], rule: str) -> bool:
    if rule == "derived class suppresses work":
        return answer.get("derived_class") in SUPPRESSING_CLASSES
    field, _, expected = rule.partition(" == ")
    return answer.get(field) == expected


def evaluate_gates(rows, answers) -> Dict[str, Any]:
    out = {}
    for gate_id, spec in GATES.items():
        population = [r for r in rows if r["klass"] == spec["klass"]]
        judged = [r for r in population
                  if (r["property_key"], r["condition_id"]) in answers]
        fired = sum(
            1 for r in judged
            if _fires(answers[(r["property_key"], r["condition_id"])], spec["rule"])
        )
        if not judged:
            passed = None
        elif spec["cmp"] == "<=":
            passed = fired <= spec["threshold"]
        else:
            passed = fired >= spec["threshold"]
        out[gate_id] = {
            **{k: spec[k] for k in
               ("title", "klass", "rule", "cmp", "threshold", "status", "basis")},
            "n_expected": spec["n"],
            "n_population": len(population),
            "n_judged": len(judged),
            "fired": fired,
            "passed": passed,
            "exploratory": len(judged) < EXPLORATORY_MIN_JUDGED,
        }
    return out


def evaluate_g4(answers) -> Dict[str, Any]:
    total = len(answers)
    classes = Counter(a.get("derived_class") for a in answers.values())
    unclear = {
        key: sum(1 for a in answers.values() if a.get(key) == "unclear")
        for key in FACTOR_KEYS
    }
    misnamed = classes["misnamed_but_warranted"] + classes["misnamed_and_trivial"]
    share = (lambda n: (n / total) if total else 0.0)
    checks = {
        "exact_and_warranted_share": {
            "value": share(classes["exact_and_warranted"]),
            "limit": G4_CHECKS["exact_and_warranted_share_max"],
            "cmp": "<=",
            "note": "Terra's own supported rate on this population is 84.2%; "
                    "near-total exactness means the verifier is not "
                    "discriminating.",
        },
        "misnamed_share": {
            "value": share(misnamed),
            "limit": G4_CHECKS["misnamed_share_min"],
            "cmp": ">=",
            "note": "the labels say the class is real and material; ~0% means "
                    "the accuracy axis is inert.",
        },
    }
    for key in FACTOR_KEYS:
        checks[f"unclear_{key}"] = {
            "value": share(unclear[key]),
            "limit": G4_CHECKS["unclear_share_max"],
            "cmp": "<=",
            "note": "Terra's cannot_assess rate on this population is 2.8%.",
        }
    for check in checks.values():
        check["passed"] = (
            check["value"] <= check["limit"] if check["cmp"] == "<="
            else check["value"] >= check["limit"]
        )
    return {
        "total_conditions": total,
        "class_distribution": {k: classes.get(k, 0) for k in DERIVED_CLASSES},
        "checks": checks,
        "passed": all(c["passed"] for c in checks.values()) if total else None,
    }


def axis_confusions(rows, answers) -> Dict[str, Any]:
    """Per-axis, the whole point of factorizing: which QUESTION is unreliable."""
    expectations = {
        "visible": ({"exact": "yes", "misnamed": "yes", "absent": "no"}, "claim"),
        "claim_accurate_as_written": ({"exact": "yes", "misnamed": "no"}, "claim"),
        "material_enough_for_work": ({"warranted": "yes", "trivial": "no"}, "work"),
    }
    out = {}
    for axis, (expected_by, label_field) in expectations.items():
        table: Dict[str, Counter] = defaultdict(Counter)
        agree = total = 0
        for row in rows:
            answer = answers.get((row["property_key"], row["condition_id"]))
            if answer is None:
                continue
            label_value = row[label_field]
            table[label_value][answer.get(axis)] += 1
            expected = expected_by.get(label_value)
            if expected is None:
                continue
            total += 1
            agree += int(answer.get(axis) == expected)
        out[axis] = {
            "table": {k: dict(v) for k, v in sorted(table.items())},
            "agree": agree,
            "scoreable": total,
            "rate": (agree / total) if total else None,
        }
    return out


def class_confusion(rows, answers) -> Dict[str, Dict[str, int]]:
    table: Dict[str, Counter] = defaultdict(Counter)
    for row in rows:
        answer = answers.get((row["property_key"], row["condition_id"]))
        if answer is None:
            continue
        table[LABEL_SLUG_TO_CLASS[row["slug"]]][answer.get("derived_class")] += 1
    return {k: dict(v) for k, v in sorted(table.items())}


def disagreements(answers, stored) -> Dict[str, List[Dict[str, str]]]:
    """The eyeball lists. Each row is one of the two sides being wrong."""
    out = {"terra_supported_factorized_absent": [],
           "terra_rejected_factorized_visible": [],
           "misnamed_with_description": []}
    for (prop, cid), answer in sorted(answers.items()):
        verdict = stored.get((prop, cid))
        derived = answer.get("derived_class")
        row = {"property_key": prop, "condition_id": cid,
               "stored_verdict": verdict or "—",
               "derived_class": derived or "—"}
        if verdict == "supported" and derived == "absent":
            out["terra_supported_factorized_absent"].append(row)
        if verdict in ("unsupported", "cannot_assess") and answer.get("visible") == "yes":
            out["terra_rejected_factorized_visible"].append(row)
        if answer.get("claim_accurate_as_written") == "no":
            out["misnamed_with_description"].append(
                {**row, "observed_description":
                    answer.get("observed_description") or "(none)"}
            )
    return out


def _presence(answer: Mapping[str, Any]) -> str:
    """Fold a factorized answer onto Terra's presence axis for a like-for-like
    count. Only `visible` is comparable; accuracy and materiality have no
    counterpart in the verdict vocabulary."""
    visible = answer.get("visible")
    if visible == "no":
        return "unsupported"
    if visible == "unclear":
        return "cannot_assess"
    return "supported"


def replica_benchmark(answers, run1, run2) -> Dict[str, Any]:
    """Factorized-vs-run_1 next to run_1-vs-run_2: never report against zero.

    Both sides are compared in the semantic key space so the replica pair can
    join at all (see load_stored_verdicts).
    """
    shared = set(run1["by_key"]) & set(run2["by_key"])
    replica_flips = sum(
        1 for key in shared if run1["by_key"][key] != run2["by_key"][key]
    )
    # Restrict the arm to the SAME shared population, so the two rates are
    # measured on one population rather than two.
    cid_to_key = run1["cid_to_key"]
    arm_on_shared = {
        cid_to_key[cid]: answer for cid, answer in answers.items()
        if cid in cid_to_key and cid_to_key[cid] in shared
    }
    arm_flips = sum(
        1 for key, answer in arm_on_shared.items()
        if _presence(answer) != run1["by_key"][key]
    )
    judged_all = [cid for cid in answers if cid in run1["by_cid"]]
    return {
        "replica_pairs": len(shared),
        "replica_flips": replica_flips,
        "replica_rate": (replica_flips / len(shared)) if shared else None,
        "arm_pairs": len(arm_on_shared),
        "arm_flips": arm_flips,
        "arm_rate": (arm_flips / len(arm_on_shared)) if arm_on_shared else None,
        "arm_pairs_all": len(judged_all),
        "arm_flips_all": sum(
            1 for cid in judged_all
            if _presence(answers[cid]) != run1["by_cid"][cid]
        ),
    }


def adherence(answers) -> Dict[str, int]:
    missing_description = sum(
        1 for a in answers.values()
        if a.get("claim_accurate_as_written") == "no"
        and not (a.get("observed_description") or "").strip()
    )
    invisible_not_unclear = sum(
        1 for a in answers.values()
        if a.get("visible") == "no" and any(
            a.get(key) != "unclear" for key in FACTOR_KEYS[1:]
        )
    )
    return {
        "missing_observed_description": missing_description,
        "invisible_factors_not_unclear": invisible_not_unclear,
    }


# ---------------------------------------------------------------- rendering

def _pct(value: Optional[float]) -> str:
    return "—" if value is None else f"{100.0 * value:.1f}%"


def render_md(a: Dict[str, Any]) -> str:
    arm = a["arm"]
    lines = [
        "# Factorized verifier replay — scorecard",
        "",
        f"Generated for arm `{arm['label']}` · contract "
        f"`{arm['response_contract']}` · prompt `{arm['prompt_version']}` · "
        f"labels {ls.LABEL_VERSION} · integrity "
        f"**{'OK' if a['integrity_ok'] else 'FAILED'}**",
        "",
        NOISE_FLOOR_NOTE,
        "",
        "## 1. Inputs and coverage",
        "",
        "| input | sha256 | bytes |",
        "|---|---|---|",
    ]
    for row in a["inputs"]:
        lines.append(f"| `{row['path']}` | `{row['sha256'][:12]}` | {row['bytes']:,} |")
    cov = a["coverage"]
    lines += [
        "",
        f"Conditions answered: **{cov['conditions_answered']:,}** · unit "
        f"records {cov['unit_statuses']} · labeled canary cards joined "
        f"**{cov['labels_joined']}/{cov['labels_total']}**",
        "",
        f"Production label cards excluded (by design): "
        f"**{a['label_meta']['excluded_production_cards']}**. Canary-only is "
        "the scoreable population.",
    ]
    if arm["dry_run"]:
        lines += ["", "> **DRY RUN root — nothing was re-decided.** Coverage "
                      "only; every gate below is unscored."]

    g4 = a["g4"]
    lines += [
        "",
        "## 2. Full-population sweep (label-free)",
        "",
        f"All **{g4['total_conditions']:,}** replayed conditions. Stored Terra "
        "on the same population: `supported` 84.2% · `unsupported` 13.0% · "
        "`cannot_assess` 2.8%.",
        "",
        "| derived class | n | share |",
        "|---|---|---|",
    ]
    total = g4["total_conditions"] or 1
    for name, count in g4["class_distribution"].items():
        lines.append(f"| `{name}` | {count} | {100.0 * count / total:.1f}% |")
    lines += ["", "### Degeneracy and health checks (G4)", "",
              "| check | value | rule | verdict |", "|---|---|---|---|"]
    for name, check in g4["checks"].items():
        lines.append(
            f"| `{name}` | {_pct(check['value'])} | {check['cmp']} "
            f"{_pct(check['limit'])} | "
            f"{'PASS' if check['passed'] else '**FAIL**'} |"
        )
    lines += [
        "",
        "*Evidence:* G1 and G2 are both satisfied by a verifier that answers "
        "`yes` to everything, and the category that would catch that (G3) is "
        "underpowered until L2. These label-free checks are what make the "
        "decisive gates trustworthy. No action is prescribed here.",
    ]

    lines += ["", "## 3. Per-axis agreement with the human labels", ""]
    for axis, data in a["axes"].items():
        lines += [
            f"### `{axis}` — {data['agree']}/{data['scoreable']} "
            f"({_pct(data['rate'])})",
            "",
            "| label | " + " | ".join(sorted(FACTOR_VALUES_ORDER)) + " |",
            "|---|" + "---|" * len(FACTOR_VALUES_ORDER),
        ]
        for label_value, counts in data["table"].items():
            cells = " | ".join(
                str(counts.get(v, 0)) for v in sorted(FACTOR_VALUES_ORDER)
            )
            lines.append(f"| `{label_value}` | {cells} |")
        lines.append("")

    lines += ["## 4. Class confusion (label → derived)", "",
              "| label class | " + " | ".join(f"`{c}`" for c in DERIVED_CLASSES)
              + " |",
              "|---|" + "---|" * len(DERIVED_CLASSES)]
    for label_class, counts in a["class_confusion"].items():
        cells = " | ".join(str(counts.get(c, 0)) for c in DERIVED_CLASSES)
        lines.append(f"| `{label_class}` | {cells} |")

    lines += ["", "## 5. Gates", "",
              "| gate | population | judged | fired | rule | verdict |",
              "|---|---|---|---|---|---|"]
    for gate_id, gate in a["gates"].items():
        if gate["passed"] is None:
            verdict = "unscored"
        else:
            verdict = "PASS" if gate["passed"] else "**FAIL**"
        if gate["status"] != "decisive":
            verdict += f" ({gate['status']})"
        lines.append(
            f"| **{gate_id}** {gate['title']} | `{gate['klass']}` "
            f"n={gate['n_population']} | {gate['n_judged']} | {gate['fired']} "
            f"| {gate['cmp']} {gate['threshold']} · {gate['rule']} | {verdict} |"
        )
    decisive = [g for g in a["gates"].values() if g["status"] == "decisive"]
    decisive_ok = all(g["passed"] for g in decisive) if all(
        g["passed"] is not None for g in decisive
    ) else None
    overall = (
        "unscored" if decisive_ok is None or g4["passed"] is None
        else "**PASS**" if (decisive_ok and g4["passed"]) else "**FAIL**"
    )
    lines += [
        "",
        f"Decisive result (G1 + G2 + G4, with G3 directional): {overall}.",
        "",
        "*Evidence:* thresholds were pre-committed in "
        "`docs/HANDOFF_factorized_verifier_replay.md` §7 before the run. A "
        "decisive failure means the design gets revised, not the threshold. "
        "No action is prescribed here.",
    ]

    dis = a["disagreements"]
    lines += ["", "## 6. Disagreement audits", ""]
    for key, title in (
        ("terra_supported_factorized_absent",
         "Terra supported, factorized absent — one side is hallucinating"),
        ("terra_rejected_factorized_visible",
         "Terra rejected, factorized sees it — the recovery class"),
        ("misnamed_with_description",
         "Factorized says misnamed — the future remap lane's input"),
    ):
        rows = dis[key]
        lines += [f"### {title} ({len(rows)})", ""]
        if not rows:
            lines += ["_none_", ""]
            continue
        for row in rows[:40]:
            extra = row.get("observed_description")
            suffix = f" — sees: {extra}" if extra else ""
            lines.append(
                f"- `{row['property_key']}` `{row['condition_id']}` "
                f"stored={row['stored_verdict']} "
                f"derived={row['derived_class']}{suffix}"
            )
        if len(rows) > 40:
            lines.append(f"- … {len(rows) - 40} more (see the json)")
        lines.append("")

    rep = a["replica"]
    lines += [
        "## 7. Stability benchmark", "",
        f"Both rates below are measured on the {rep['replica_pairs']} conditions "
        f"the two replicas share, keyed by (property, catalog item, unit) — "
        f"condition_id embeds the run id, so replicas share none of those.",
        "",
        f"- Factorized vs stored run_1: **{rep['arm_flips']}/{rep['arm_pairs']}** "
        f"({_pct(rep['arm_rate'])}) on Terra's presence axis",
        f"- Stored run_1 vs run_2 (the noise floor): "
        f"**{rep['replica_flips']}/{rep['replica_pairs']}** "
        f"({_pct(rep['replica_rate'])})",
        f"- Factorized vs run_1 across all replayed conditions: "
        f"**{rep['arm_flips_all']}/{rep['arm_pairs_all']}**",
        "",
        "## 8. Prompt adherence", "",
        f"- Misnamed answers with no `observed_description`: "
        f"**{a['adherence']['missing_observed_description']}**",
        f"- `visible = no` answers whose other factors are not `unclear`: "
        f"**{a['adherence']['invisible_factors_not_unclear']}**",
        "",
        "## 9. What this does not show", "",
    ]
    lines += [f"- {item}" for item in NOT_SHOWN]
    lines.append("")
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--arm-root", type=Path, required=True,
                        help="a --factorized harness output root")
    parser.add_argument("--labels", type=Path,
                        default=REPO_ROOT / "reports" / "labels_v1_1.json")
    parser.add_argument(
        "--canary-root", type=Path,
        default=REPO_ROOT / "artifacts_canary" / "renovation_session9_20260818",
        help="frozen canary root (for the stored verdicts and run_2 replica)",
    )
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--out-md", type=Path,
                        default=REPO_ROOT / "reports" / "factorized_review_scorecard.md")
    parser.add_argument("--out-json", type=Path,
                        default=REPO_ROOT / "reports" / "factorized_review_scorecard.json")
    args = parser.parse_args(argv)
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

    rows, label_meta = load_canary_labels(
        args.labels, allow_partial=args.allow_partial
    )
    arm = load_arm(args.arm_root)
    run1 = load_stored_verdicts(args.canary_root / "run_1" / "candidate")
    run2 = load_stored_verdicts(args.canary_root / "run_2" / "candidate")
    answers = arm["answers"]

    joined = sum(
        1 for r in rows if (r["property_key"], r["condition_id"]) in answers
    )
    analysis = {
        "schema_version": 1,
        "label_version": ls.LABEL_VERSION,
        "arm": {k: arm[k] for k in
                ("label", "dry_run", "response_contract", "prompt_version")},
        "inputs": [
            {"path": str(p), "sha256": sha256_file(p), "bytes": p.stat().st_size}
            for p in (args.labels, args.arm_root / "manifest.json")
            if p.is_file()
        ],
        "label_meta": label_meta,
        "coverage": {
            "conditions_answered": len(answers),
            "unit_statuses": arm["statuses"],
            "labels_total": len(rows),
            "labels_joined": joined,
            "stored_run1_conditions": len(run1["by_cid"]),
            "stored_run2_conditions": len(run2["by_cid"]),
        },
        "gates": evaluate_gates(rows, answers),
        "g4": evaluate_g4(answers),
        "axes": axis_confusions(rows, answers),
        "class_confusion": class_confusion(rows, answers),
        "disagreements": disagreements(answers, run1["by_cid"]),
        "replica": replica_benchmark(answers, run1, run2),
        "adherence": adherence(answers),
        "not_shown": NOT_SHOWN,
    }
    analysis["integrity_ok"] = bool(
        answers
        and not arm["dry_run"]
        and joined == len(rows)
        and not arm["statuses"].get("refused")
    )

    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text(render_md(analysis), encoding="utf-8")
    args.out_json.write_text(
        json.dumps(analysis, indent=1, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    print(f"labels: {ls.LABEL_VERSION} · {len(rows)} canary cards "
          f"({label_meta['excluded_production_cards']} production excluded)")
    print(f"arm: {arm['label']} · {len(answers)} conditions answered "
          f"· joined {joined}/{len(rows)}")
    for gate_id, gate in analysis["gates"].items():
        state = ("unscored" if gate["passed"] is None
                 else "PASS" if gate["passed"] else "FAIL")
        print(f"  {gate_id:4s} {gate['title']:38s} "
              f"{gate['fired']}/{gate['n_judged']:<3d} {state}")
    g4_passed = analysis["g4"]["passed"]
    g4_state = ("unscored" if g4_passed is None
                else "PASS" if g4_passed else "FAIL")
    print(f"  G4   degeneracy & health                    {g4_state}")
    print(f"integrity: {'OK' if analysis['integrity_ok'] else 'FAILED'}")
    print(f"scorecard: {args.out_md}")
    return 0 if analysis["integrity_ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
