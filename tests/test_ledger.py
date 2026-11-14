"""U-4, U-5 and the cycle-exclusion part of U-33."""

import datetime as dt
import json
import sqlite3

import pytest

from advisor_call.ledger import Ledger, LedgerError

D = dt.date


def _exchange(ledger, day, cost, *, session="s1", call="c-1", ok=True):
    ledger.record_exchange(
        ts=0.0,
        day=day,
        session_id=session,
        call_id=call,
        line="sonnet",
        model="claude-sonnet-5-5",
        input_tokens=10,
        output_tokens=5,
        cache_read_tokens=0,
        cache_write_tokens=0,
        cost_usd=cost,
        ok=ok,
    )


def test_writes_and_day_totals(db_path):
    ledger = Ledger.at_path(db_path)
    _exchange(ledger, D(2026, 11, 14), 0.10)
    _exchange(ledger, D(2026, 11, 14), 0.05, ok=False)  # failed but billed still counts
    _exchange(ledger, D(2026, 11, 15), 0.20)
    assert ledger.day_total(D(2026, 11, 14)) == pytest.approx(0.15)
    assert ledger.day_total(D(2026, 11, 15)) == pytest.approx(0.20)  # a new day starts at its own
    assert ledger.day_total(D(2026, 11, 16)) == 0.0


def test_totals_exclude_previous_cycle_and_future(db_path):
    ledger = Ledger.at_path(db_path)
    _exchange(ledger, D(2026, 11, 2), 5.0)  # previous cycle (start day 3)
    _exchange(ledger, D(2026, 11, 3), 1.0)
    _exchange(ledger, D(2026, 11, 13), 2.0)
    _exchange(ledger, D(2026, 11, 14), 0.5)
    _exchange(ledger, D(2026, 11, 15), 9.0)  # clock skew: a later day is not "before today"
    today, before = ledger.totals(D(2026, 11, 14), D(2026, 11, 3))
    assert today == pytest.approx(0.5)
    assert before == pytest.approx(3.0)


def test_daily_totals_fill_gaps(db_path):
    ledger = Ledger.at_path(db_path)
    _exchange(ledger, D(2026, 11, 12), 0.3)
    _exchange(ledger, D(2026, 11, 14), 0.1)
    rows = ledger.daily_totals(D(2026, 11, 14), 3)
    assert rows == [
        (D(2026, 11, 12), pytest.approx(0.3)),
        (D(2026, 11, 13), 0.0),
        (D(2026, 11, 14), pytest.approx(0.1)),
    ]


def test_call_upsert_and_open_record(db_path):
    ledger = Ledger.at_path(db_path)
    common = dict(
        call_id="c-1", session_id="s1", line="sonnet", started_at=1.0, initiated_by="executor"
    )
    ledger.record_call(**common, ended_at=None, end_reason="", exchanges=0, cost_usd=0.0)
    assert ledger.open_call_record("c-1", "s1")
    assert not ledger.open_call_record("c-1", "other-session")
    ledger.record_call(**common, ended_at=9.0, end_reason="hangup", exchanges=3, cost_usd=0.12)
    assert not ledger.open_call_record("c-1", "s1")
    row = (
        sqlite3.connect(db_path)
        .execute("SELECT end_reason, exchanges, cost_usd FROM calls")
        .fetchone()
    )
    assert row == ("hangup", 3, 0.12)


def test_no_message_content_columns(db_path):
    ledger = Ledger.at_path(db_path)
    _exchange(ledger, D(2026, 11, 14), 0.1)
    cols = {
        r[1]
        for t in ("exchanges", "calls")
        for r in sqlite3.connect(db_path).execute(f"PRAGMA table_info({t})")
    }
    assert not cols & {"content", "message", "messages", "text", "reply", "question"}


def test_read_failure_raises_ledger_error():
    def broken():
        raise sqlite3.OperationalError("disk I/O error")

    ledger = Ledger(broken)
    with pytest.raises(LedgerError):
        ledger.day_total(D(2026, 11, 14))
    with pytest.raises(LedgerError):
        ledger.totals(D(2026, 11, 14), D(2026, 11, 3))


def test_reconnects_after_failure(db_path):
    attempts = {"n": 0}

    def flaky():
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise sqlite3.OperationalError("locked")
        return sqlite3.connect(str(db_path), check_same_thread=False)

    ledger = Ledger(flaky)
    with pytest.raises(LedgerError):
        ledger.day_total(D(2026, 11, 14))
    assert ledger.day_total(D(2026, 11, 14)) == 0.0


def test_transcripts(tmp_path, db_path):
    ledger = Ledger.at_path(db_path, transcripts_dir=tmp_path / "t")
    ledger.append_transcript("c-1/../x", {"role": "user", "content": "hi"})
    files = list((tmp_path / "t").iterdir())
    assert [f.name for f in files] == ["c-1x.jsonl"]  # no path traversal
    assert json.loads(files[0].read_text()) == {"role": "user", "content": "hi"}
    with pytest.raises(LedgerError):
        Ledger.at_path(db_path).append_transcript("c-1", {})
