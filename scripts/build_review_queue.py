"""Build reports/review_queue.json for the manual v5 output-quality review.

Reads the Session 9 canary (run_1 cards, run_2 only for Terra self-flips) and
production new-mode artifacts (run dirs >= --since), writes one generic card
queue. Idempotent and stateless (hash-threshold uniform sample), zero provider
calls; never touches the Session 9 builder or its outputs.

Run:
  .venv\\Scripts\\python.exe scripts\\build_review_queue.py [--check]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import review_cards as rc  # noqa: E402

EXPECTED_CANARY = {"dirA": 32, "dirB": 36, "terra_flip": 16, "p1_package": 13, "p3_bathroom": 10, "p6_forced_single": 5}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--canary-root", type=Path, default=rc.CANARY_ROOT)
    ap.add_argument("--no-canary", action="store_true")
    ap.add_argument("--prod-root", type=Path, default=rc.PROD_ROOT)
    ap.add_argument("--no-prod", action="store_true")
    ap.add_argument("--since", default=rc.DEFAULT_SINCE, help="production run-dir prefix floor (YYYYMMDD_HHMMSS)")
    ap.add_argument("--uniform-rate", type=float, default=rc.DEFAULT_UNIFORM_RATE)
    ap.add_argument("--order", choices=("hash", "tier"), default="hash")
    ap.add_argument("--packets", type=Path, default=rc.PACKETS_PATH)
    ap.add_argument("--fe-db", type=Path, default=rc.FE_DB)
    ap.add_argument("--out", type=Path, default=ROOT / "reports" / "review_queue.json")
    ap.add_argument("--check", action="store_true", help="assert the canary strata counts and legacy-id coverage")
    args = ap.parse_args(argv)

    q = rc.build_queue(canary_root=None if args.no_canary else args.canary_root,
                       prod_root=None if args.no_prod else args.prod_root, since=args.since,
                       uniform_rate=args.uniform_rate, packets_path=args.packets, fe_db=args.fe_db, order=args.order)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(q, indent=1, ensure_ascii=False), encoding="utf-8")

    print(f"wrote {args.out}  cards={len(q['cards'])}  order={q['order']}  uniform_rate={q['uniform_rate']}")
    for src, m in q["meta"].items():
        print(f"  {src}: listings={m['listings']} accepted={m['accepted']} "
              f"(dirA {m['accepted_dirA']}, other {m['accepted_non_dirA']})  strata={m['strata']}"
              + (f"  low_res_excluded={m['low_res_excluded']}" if m.get("low_res_excluded") else ""))
    phases = {}
    for c in q["cards"]:
        phases[c["phase"]] = phases.get(c["phase"], 0) + 1
    print(f"  phases: {dict(sorted(phases.items()))}")

    if not args.check:
        return 0
    rc_ok = True
    can = (q["meta"].get("canary") or {}).get("strata") or {}
    for k, v in EXPECTED_CANARY.items():
        got = can.get(k, 0)
        flag = "ok" if got == v else "FAIL"
        rc_ok &= got == v
        print(f"  [{flag}] canary {k}: {got} (expected {v})")
    legacy_expected = {"condition": 68, "package": 13, "bathroom": 10}
    have = {"condition": 0, "package": 0, "bathroom": 0}
    for c in q["cards"]:
        if c["source"] == "canary" and c.get("legacy_item_id") and str(c["legacy_item_id"])[0] in "CPB":
            have[c["kind"]] += 1
    for k, v in legacy_expected.items():
        flag = "ok" if have[k] == v else "FAIL"
        rc_ok &= have[k] == v
        print(f"  [{flag}] legacy ids mapped ({k}): {have[k]} (expected {v})")
    print("CHECK", "PASS" if rc_ok else "FAIL")
    return 0 if rc_ok else 1


if __name__ == "__main__":
    sys.exit(main())
