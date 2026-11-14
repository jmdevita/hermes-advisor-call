"""SQLite ledger: one row per exchange and per call. Metadata only, never
message content.

In Hermes the connection comes from ``plugins.plugin_storage.plugin_db``
(``<hermes home>/plugin-data/advisor-call/``). Tests pass their own
connection factory. Every public method raises ``LedgerError`` on failure;
callers decide whether to fail closed (reading totals before spending) or
carry on (writing after the fact).
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import sqlite3
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .settings import PLUGIN_ID

log = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS exchanges (
  id INTEGER PRIMARY KEY,
  ts REAL NOT NULL,
  day TEXT NOT NULL,
  session_id TEXT NOT NULL,
  call_id TEXT NOT NULL,
  line TEXT NOT NULL,
  model TEXT,
  input_tokens INTEGER, output_tokens INTEGER,
  cache_read_tokens INTEGER, cache_write_tokens INTEGER,
  cost_usd REAL NOT NULL,
  ok INTEGER NOT NULL,
  estimated INTEGER NOT NULL DEFAULT 0  -- 1 when cost_usd is an estimate
);
CREATE INDEX IF NOT EXISTS idx_exchanges_day ON exchanges(day);

CREATE TABLE IF NOT EXISTS calls (
  call_id TEXT PRIMARY KEY, session_id TEXT, line TEXT,
  started_at REAL, ended_at REAL, end_reason TEXT,
  exchanges INTEGER, cost_usd REAL, initiated_by TEXT
);
"""


class LedgerError(Exception):
    pass


def day_key(day: dt.date) -> str:
    return day.isoformat()


class Ledger:
    def __init__(
        self, connect: Callable[[], sqlite3.Connection], transcripts_dir: Path | None = None
    ) -> None:
        self._connect = connect
        self._conn: sqlite3.Connection | None = None
        self._lock = threading.Lock()
        self.transcripts_dir = transcripts_dir

    @classmethod
    def for_hermes(cls) -> Ledger:
        """Ledger in Hermes' per-plugin storage. Imports Hermes lazily."""

        def connect() -> sqlite3.Connection:
            from plugins.plugin_storage import plugin_db

            return plugin_db(PLUGIN_ID, "ledger.db")

        def transcripts() -> Path | None:
            try:
                from plugins.plugin_storage import plugin_data_dir

                return plugin_data_dir(PLUGIN_ID) / "transcripts"
            except Exception:  # noqa: BLE001
                return None

        return cls(connect, transcripts_dir=transcripts())

    @classmethod
    def at_path(cls, path: Path, transcripts_dir: Path | None = None) -> Ledger:
        return cls(
            lambda: sqlite3.connect(str(path), check_same_thread=False),
            transcripts_dir=transcripts_dir,
        )

    # ----------------------------------------------------------------- internals

    def _db(self) -> sqlite3.Connection:
        if self._conn is None:
            conn = self._connect()
            conn.executescript(SCHEMA)
            conn.commit()
            self._conn = conn
        return self._conn

    def _run(self, fn: Callable[[sqlite3.Connection], Any]) -> Any:
        with self._lock:
            try:
                return fn(self._db())
            except LedgerError:
                raise
            except Exception as exc:  # noqa: BLE001
                # Drop the connection so the next call reconnects.
                self._conn = None
                raise LedgerError(f"{type(exc).__name__}: {exc}") from exc

    # ----------------------------------------------------------------- writes

    def record_exchange(
        self,
        *,
        ts: float,
        day: dt.date,
        session_id: str,
        call_id: str,
        line: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cache_read_tokens: int,
        cache_write_tokens: int,
        cost_usd: float,
        ok: bool,
        estimated: bool = False,
    ) -> None:
        def write(db: sqlite3.Connection) -> None:
            db.execute(
                "INSERT INTO exchanges (ts, day, session_id, call_id, line, model, input_tokens,"
                " output_tokens, cache_read_tokens, cache_write_tokens, cost_usd, ok, estimated)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    ts,
                    day_key(day),
                    session_id,
                    call_id,
                    line,
                    model,
                    input_tokens,
                    output_tokens,
                    cache_read_tokens,
                    cache_write_tokens,
                    float(cost_usd),
                    1 if ok else 0,
                    1 if estimated else 0,
                ),
            )
            db.commit()

        self._run(write)

    def record_call(
        self,
        *,
        call_id: str,
        session_id: str,
        line: str,
        started_at: float,
        ended_at: float | None,
        end_reason: str,
        exchanges: int,
        cost_usd: float,
        initiated_by: str,
    ) -> None:
        def write(db: sqlite3.Connection) -> None:
            db.execute(
                "INSERT INTO calls (call_id, session_id, line, started_at, ended_at, end_reason,"
                " exchanges, cost_usd, initiated_by) VALUES (?,?,?,?,?,?,?,?,?)"
                " ON CONFLICT(call_id) DO UPDATE SET ended_at=excluded.ended_at,"
                " end_reason=excluded.end_reason, exchanges=excluded.exchanges,"
                " cost_usd=excluded.cost_usd",
                (
                    call_id,
                    session_id,
                    line,
                    started_at,
                    ended_at,
                    end_reason,
                    exchanges,
                    float(cost_usd),
                    initiated_by,
                ),
            )
            db.commit()

        self._run(write)

    def append_transcript(self, call_id: str, record: dict[str, Any]) -> None:
        """Only used when log_transcripts is true."""
        if self.transcripts_dir is None:
            raise LedgerError("no transcripts directory available")
        safe = "".join(ch for ch in call_id if ch.isalnum() or ch in "-_") or "call"
        try:
            self.transcripts_dir.mkdir(parents=True, exist_ok=True)
            with open(self.transcripts_dir / f"{safe}.jsonl", "a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        except OSError as exc:
            raise LedgerError(f"transcript write failed: {exc}") from exc

    # ----------------------------------------------------------------- reads

    def day_total(self, day: dt.date) -> float:
        def read(db: sqlite3.Connection) -> float:
            row = db.execute(
                "SELECT COALESCE(SUM(cost_usd),0) FROM exchanges WHERE day = ?", (day_key(day),)
            ).fetchone()
            return float(row[0])

        return self._run(read)

    def totals(self, today: dt.date, cycle_start: dt.date) -> tuple[float, float]:
        """(spent_today, spent_before_today) for the cycle starting at
        cycle_start. Spend from earlier cycles is excluded."""

        def read(db: sqlite3.Connection) -> tuple[float, float]:
            row = db.execute(
                "SELECT COALESCE(SUM(CASE WHEN day = ? THEN cost_usd END),0),"
                " COALESCE(SUM(CASE WHEN day < ? THEN cost_usd END),0)"
                " FROM exchanges WHERE day >= ? AND day <= ?",
                (day_key(today), day_key(today), day_key(cycle_start), day_key(today)),
            ).fetchone()
            return float(row[0]), float(row[1])

        return self._run(read)

    def daily_totals(self, today: dt.date, days: int) -> list[tuple[dt.date, float]]:
        """The last `days` days ending today, oldest first, zeros included."""
        first = today - dt.timedelta(days=days - 1)

        def read(db: sqlite3.Connection) -> dict[str, float]:
            rows = db.execute(
                "SELECT day, SUM(cost_usd) FROM exchanges WHERE day >= ? AND day <= ? GROUP BY day",
                (day_key(first), day_key(today)),
            ).fetchall()
            return {d: float(v) for d, v in rows}

        found = self._run(read)
        out = []
        for i in range(days):
            d = first + dt.timedelta(days=i)
            out.append((d, found.get(day_key(d), 0.0)))
        return out

    def has_estimates(self, since: dt.date, until: dt.date) -> bool:
        """True if any exchange in [since, until] was billed by estimate."""

        def read(db: sqlite3.Connection) -> bool:
            row = db.execute(
                "SELECT 1 FROM exchanges WHERE estimated = 1 AND day >= ? AND day <= ? LIMIT 1",
                (day_key(since), day_key(until)),
            ).fetchone()
            return row is not None

        return self._run(read)

    def open_call_record(self, call_id: str, session_id: str) -> bool:
        """True if the ledger has this call for this session with no end
        recorded: it was open when the process stopped."""

        def read(db: sqlite3.Connection) -> bool:
            row = db.execute(
                "SELECT ended_at FROM calls WHERE call_id = ? AND session_id = ?",
                (call_id, session_id),
            ).fetchone()
            return row is not None and row[0] is None

        return self._run(read)
