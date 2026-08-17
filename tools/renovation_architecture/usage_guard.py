"""Cross-process daily token budgets for the review models (UTC).

Terra condition review debits a 2,500,000/day ledger; Sol package review
debits its own 250,000/day ledger (Session 8). Each is a SQLite file under
<artifacts_root>/.renovation_architecture/, or under
RENOVATION_TERRA_USAGE_ROOT when several artifacts roots must share one
daily budget — both ledger files live under that one root, so the two
budgets stay separate while sharing the location. Every call reserves a
conservative token estimate inside one BEGIN IMMEDIATE transaction (write
lock up front, so concurrent workers on one artifacts root serialize —
including across processes on Windows), then settles to the
provider-reported total. Unknown usage keeps the reservation — the guard
over-counts rather than under-counts. The ledger path is injected per run;
nothing here runs at import time. Checkpoint reuse never touches a ledger.
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

TERRA_DAILY_TOKEN_CEILING = 2_500_000
RESERVATION_FLOOR_TOKENS = 25_000
RESERVATION_TOKENS_PER_IMAGE = 5_000
LEDGER_RELATIVE_PATH = Path(".renovation_architecture") / "terra_usage.sqlite3"

# Sol listing calls average ~8.6k tokens; Terra's 25k floor would spuriously
# deny fresh calls long before the much smaller Sol ceiling is truly reached.
SOL_DAILY_TOKEN_CEILING = 250_000
SOL_RESERVATION_FLOOR_TOKENS = 10_000
SOL_LEDGER_RELATIVE_PATH = Path(".renovation_architecture") / "sol_usage.sqlite3"

_SCHEMA_TEMPLATE = """
CREATE TABLE IF NOT EXISTS {table} (
  id INTEGER PRIMARY KEY,
  utc_day TEXT NOT NULL,
  property_key TEXT NOT NULL,
  source_run_id TEXT NOT NULL,
  estimate_unit_id TEXT NOT NULL,
  request_fingerprint TEXT NOT NULL,
  state TEXT NOT NULL CHECK (state IN ('reserved','settled')),
  debited_tokens INTEGER NOT NULL CHECK (debited_tokens >= 0),
  provider_total_tokens INTEGER,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS {table}_day_idx ON {table} (utc_day);
"""


class TerraDailyBudgetExceeded(RuntimeError):
    """Pre-call rejection: the reservation would cross the daily ceiling."""


class SolDailyBudgetExceeded(RuntimeError):
    """Pre-call rejection: the reservation would cross the daily ceiling."""


def _utc_today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def estimate_reservation_tokens(
    *, max_output_tokens: int, request_bytes: int, image_count: int
) -> int:
    """Conservative pre-call estimate: prompt bytes over-count prompt tokens,
    a flat per-image charge covers vision input, plus the full output cap."""
    return max(
        RESERVATION_FLOOR_TOKENS,
        int(max_output_tokens)
        + int(request_bytes)
        + RESERVATION_TOKENS_PER_IMAGE * int(image_count),
    )


def estimate_sol_reservation_tokens(
    *, max_output_tokens: int, request_bytes: int
) -> int:
    """Sol's conservative pre-call estimate — text-only, so no image charge."""
    return max(
        SOL_RESERVATION_FLOOR_TOKENS,
        int(max_output_tokens) + int(request_bytes),
    )


def resolve_ledger_root(artifacts_root: Path, *, override: Optional[str] = None) -> Path:
    """Where the daily ledgers live: the run's artifacts root by default, or a
    shared override (RENOVATION_TERRA_USAGE_ROOT) so several artifacts roots
    debit ONE daily budget per model."""
    if override and str(override).strip():
        return Path(str(override).strip())
    return Path(artifacts_root)


class _DailyUsageLedger:
    """One model's daily token ledger. Subclasses pin the table, the file,
    and the denial exception; reserve/settle semantics are shared."""

    _TABLE = ""
    _RELATIVE_PATH: Path = Path()
    _EXCEEDED: type = RuntimeError
    _MODEL_LABEL = ""

    def __init__(
        self,
        artifacts_root: Path,
        *,
        daily_ceiling: int,
        usage_root_override: Optional[str] = None,
    ):
        self._path = (
            resolve_ledger_root(artifacts_root, override=usage_root_override)
            / self._RELATIVE_PATH
        )
        self._daily_ceiling = int(daily_ceiling)

    @property
    def path(self) -> Path:
        return self._path

    def _connect(self) -> sqlite3.Connection:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        # isolation_level=None: no implicit transactions — every mutation runs
        # in an explicit BEGIN IMMEDIATE below. Default rollback journal (no
        # WAL: the artifacts root may be a network share).
        conn = sqlite3.connect(str(self._path), timeout=30, isolation_level=None)
        conn.executescript(_SCHEMA_TEMPLATE.format(table=self._TABLE))
        return conn

    def reserve(
        self,
        *,
        property_key: str,
        source_run_id: str,
        estimate_unit_id: str,
        request_fingerprint: str,
        tokens: int,
    ) -> int:
        """Atomically debit a reservation, or raise BEFORE any provider call."""
        amount = int(tokens)
        day = _utc_today()
        now = _utc_now_iso()
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            try:
                spent = int(
                    conn.execute(
                        f"SELECT COALESCE(SUM(debited_tokens), 0) "
                        f"FROM {self._TABLE} WHERE utc_day = ?",
                        (day,),
                    ).fetchone()[0]
                )
                if spent + amount > self._daily_ceiling:
                    raise self._EXCEEDED(
                        f"daily {self._MODEL_LABEL} ceiling of "
                        f"{self._daily_ceiling} tokens would be exceeded: "
                        f"{spent} already debited on {day}, reservation needs "
                        f"{amount}"
                    )
                cursor = conn.execute(
                    f"INSERT INTO {self._TABLE} (utc_day, property_key, "
                    "source_run_id, estimate_unit_id, request_fingerprint, "
                    "state, debited_tokens, provider_total_tokens, created_at, "
                    "updated_at) VALUES (?, ?, ?, ?, ?, 'reserved', ?, NULL, ?, ?)",
                    (day, property_key, source_run_id, estimate_unit_id,
                     request_fingerprint, amount, now, now),
                )
            except BaseException:
                conn.execute("ROLLBACK")
                raise
            conn.execute("COMMIT")
            return int(cursor.lastrowid)
        finally:
            conn.close()

    def settle(
        self, reservation_id: int, *, provider_total_tokens: Optional[int]
    ) -> int:
        """Settle to provider truth; unknown usage retains the reservation.
        Returns the final debited amount."""
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            try:
                if provider_total_tokens is None:
                    conn.execute(
                        f"UPDATE {self._TABLE} SET state='settled', updated_at=? "
                        "WHERE id=?",
                        (_utc_now_iso(), int(reservation_id)),
                    )
                else:
                    conn.execute(
                        f"UPDATE {self._TABLE} SET state='settled', "
                        "debited_tokens=?, provider_total_tokens=?, updated_at=? "
                        "WHERE id=?",
                        (int(provider_total_tokens), int(provider_total_tokens),
                         _utc_now_iso(), int(reservation_id)),
                    )
                row = conn.execute(
                    f"SELECT debited_tokens FROM {self._TABLE} WHERE id=?",
                    (int(reservation_id),),
                ).fetchone()
            except BaseException:
                conn.execute("ROLLBACK")
                raise
            conn.execute("COMMIT")
            return int(row[0]) if row else 0
        finally:
            conn.close()


class TerraUsageLedger(_DailyUsageLedger):
    _TABLE = "terra_usage"
    _RELATIVE_PATH = LEDGER_RELATIVE_PATH
    _EXCEEDED = TerraDailyBudgetExceeded
    _MODEL_LABEL = "Terra"

    def __init__(
        self,
        artifacts_root: Path,
        *,
        daily_ceiling: int = TERRA_DAILY_TOKEN_CEILING,
        usage_root_override: Optional[str] = None,
    ):
        super().__init__(
            artifacts_root,
            daily_ceiling=daily_ceiling,
            usage_root_override=usage_root_override,
        )


class SolUsageLedger(_DailyUsageLedger):
    _TABLE = "sol_usage"
    _RELATIVE_PATH = SOL_LEDGER_RELATIVE_PATH
    _EXCEEDED = SolDailyBudgetExceeded
    _MODEL_LABEL = "Sol"

    def __init__(
        self,
        artifacts_root: Path,
        *,
        daily_ceiling: int = SOL_DAILY_TOKEN_CEILING,
        usage_root_override: Optional[str] = None,
    ):
        super().__init__(
            artifacts_root,
            daily_ceiling=daily_ceiling,
            usage_root_override=usage_root_override,
        )
