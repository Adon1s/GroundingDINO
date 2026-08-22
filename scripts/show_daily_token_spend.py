"""Show daily Terra/Sol token spend from the usage-guard ledgers (Session 9).

Answers "how much have I spent today, and how much free quota is left" during
a budget-guarded canary run, without hand-written SQL:

  .venv\\Scripts\\python.exe scripts\\show_daily_token_spend.py --root artifacts_canary\\<out>

Read-only: opens each ledger sqlite file if present and prints one row per
ledger for the requested UTC day (default: today). Ceilings reflect the
current environment (RENOVATION_TERRA_DAILY_TOKEN_CEILING /
RENOVATION_SOL_DAILY_TOKEN_CEILING overrides included).
"""
from __future__ import annotations

import argparse
import os
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.pipeline_config import (  # noqa: E402
    resolve_renovation_sol_daily_ceiling,
    resolve_renovation_terra_daily_ceiling,
)
from tools.renovation_architecture.usage_guard import (  # noqa: E402
    LEDGER_RELATIVE_PATH,
    SOL_LEDGER_RELATIVE_PATH,
)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True,
                        help="the usage root (RENOVATION_TERRA_USAGE_ROOT of "
                             "the run — the canary coordinator's --out)")
    parser.add_argument("--day", default=None,
                        help="UTC day YYYY-MM-DD (default: today)")
    args = parser.parse_args(argv)

    day = args.day or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    ledgers = (
        ("terra", "terra_usage", args.root / LEDGER_RELATIVE_PATH,
         resolve_renovation_terra_daily_ceiling(
             os.environ.get("RENOVATION_TERRA_DAILY_TOKEN_CEILING"))),
        ("sol", "sol_usage", args.root / SOL_LEDGER_RELATIVE_PATH,
         resolve_renovation_sol_daily_ceiling(
             os.environ.get("RENOVATION_SOL_DAILY_TOKEN_CEILING"))),
    )
    print(f"UTC day {day}  (usage root: {args.root.resolve()})")
    for label, table, path, ceiling in ledgers:
        if not path.is_file():
            print(f"  {label:5}  no ledger file ({path.name}) — no spend recorded")
            continue
        conn = sqlite3.connect(str(path))
        try:
            debited, rows, unsettled = conn.execute(
                f"SELECT COALESCE(SUM(debited_tokens), 0), COUNT(*), "
                f"COALESCE(SUM(state = 'reserved'), 0) "
                f"FROM {table} WHERE utc_day = ?",
                (day,),
            ).fetchone()
        finally:
            conn.close()
        remaining = max(0, ceiling - int(debited))
        print(
            f"  {label:5}  {int(debited):>9,} / {ceiling:,} tokens debited  "
            f"({remaining:,} remaining; {int(rows)} reservations"
            + (f", {int(unsettled)} still unsettled" if unsettled else "")
            + ")"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
