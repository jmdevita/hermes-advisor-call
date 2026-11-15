"""Shared plugin state: one Runtime per loaded plugin.

Holds the call manager, ledger, captured context and the accessors for
settings and Hermes' LLM API. Tools, hooks and the /second-opinion command all work
through it, which is also what tests construct with fakes.
"""

from __future__ import annotations

import datetime as dt
import logging
import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from typing import Any

from .calls import Call, CallManager
from .context import ContextStore, Redactor, hermes_redactor
from .cost import BudgetState, budget_state, cycle_for
from .ledger import Ledger, LedgerError
from .settings import Settings, read_settings

log = logging.getLogger(__name__)


class SystemClock:
    def time(self) -> float:
        return time.time()

    def today(self) -> dt.date:
        return dt.date.today()  # local date of the Hermes host


class Runtime:
    def __init__(
        self,
        *,
        llm: Callable[[], Any],
        ledger: Ledger,
        settings: Callable[[], Settings] = read_settings,
        clock: Any = None,
        context: ContextStore | None = None,
        redactor: Callable[[], Redactor | None] = hermes_redactor,
    ) -> None:
        self._llm = llm
        self.ledger = ledger
        self._settings = settings
        self.clock = clock or SystemClock()
        self.context = context or ContextStore()
        self._redactor_factory = redactor
        self._redactor: Redactor | None = None
        self.manager = CallManager(clock=self.clock.time, on_call_end=self.persist_call)
        self._lock = threading.Lock()
        # One-shot authorizations left by the approval hook, per session.
        self._authorizations: OrderedDict[str, tuple[str, float]] = OrderedDict()
        # Spend that could not be written to the ledger, per local day.
        self._unrecorded: dict[dt.date, float] = {}

    # ----------------------------------------------------------------- accessors

    def llm(self) -> Any:
        return self._llm()

    def settings(self) -> Settings:
        s = self._settings()
        self.manager.idle_timeout_s = s.budget.idle_timeout_minutes * 60.0
        return s

    def redactor(self) -> Redactor | None:
        if self._redactor is None:
            self._redactor = self._redactor_factory()
        return self._redactor

    def today(self) -> dt.date:
        return self.clock.today()

    # ----------------------------------------------------------------- authorization

    AUTHORIZATION_TTL_S = 15 * 60.0  # longer than Hermes' approval timeout (300 s)

    def authorize(self, session_id: str, kind: str) -> None:
        """Record what the approval hook decided for the next advisor_call_open
        in this session: "approved" (a human approval was requested), "grant"
        (a matching /second-opinion allow), or "existing" (a call was open)."""
        with self._lock:
            self._authorizations[session_id] = (kind, self.clock.time())
            self._authorizations.move_to_end(session_id)
            while len(self._authorizations) > 256:
                self._authorizations.popitem(last=False)

    def take_authorization(self, session_id: str) -> str | None:
        with self._lock:
            entry = self._authorizations.pop(session_id, None)
        if entry is None:
            return None
        kind, at = entry
        return kind if self.clock.time() - at <= self.AUTHORIZATION_TTL_S else None

    # ----------------------------------------------------------------- budget

    def add_unrecorded(self, cost_usd: float) -> None:
        """Spend the ledger could not store. Counted until the process stops."""
        if cost_usd > 0:
            with self._lock:
                day = self.today()
                self._unrecorded[day] = self._unrecorded.get(day, 0.0) + cost_usd

    def _unrecorded_split(self, today: dt.date, cycle_start: dt.date) -> tuple[float, float]:
        with self._lock:
            items = list(self._unrecorded.items())
        on_day = sum(v for d, v in items if d == today)
        before = sum(v for d, v in items if cycle_start <= d < today)
        return on_day, before

    def budget(self, settings: Settings) -> BudgetState:
        """Raises ValueError when cycle_start_day is unset and LedgerError
        when spend cannot be read. Callers fail closed on both."""
        budget = settings.budget
        if budget.cycle_start_day is None:
            raise ValueError(budget.cycle_error)
        today = self.today()
        cycle = cycle_for(today, budget.cycle_start_day)
        spent_today, spent_before = self.ledger.totals(today, cycle.start)
        extra_today, extra_before = self._unrecorded_split(today, cycle.start)
        spent_today += extra_today
        spent_before += extra_before
        return budget_state(budget, today, spent_today=spent_today, spent_before_today=spent_before)

    def bill(self, call: Call, settings: Settings) -> dict[str, Any]:
        """The bill block every tool result carries."""
        bill: dict[str, Any] = {
            "cost_usd": round(call.cost_usd, 4),
            "cap_usd": round(call.cap_usd, 4),
            "messages": call.exchanges,
            "max_messages": settings.budget.max_messages_per_call,
        }
        try:
            state = self.budget(settings)
            bill.update(
                day_usd=round(state.spent_today, 4),
                day_allowance_usd=round(state.allowance, 4),
                cycle_usd=round(state.spent_cycle, 4),
                cycle_cap_usd=round(state.per_month, 2),
            )
        except (ValueError, LedgerError) as exc:
            bill["budget_unavailable"] = str(exc)
        if call.estimated:
            bill["estimate"] = True
        return bill

    # ----------------------------------------------------------------- persistence

    def persist_call(self, call: Call) -> None:
        """Write the call's row to the ledger; a failure is logged, never raised."""
        try:
            self.ledger.record_call(
                call_id=call.call_id,
                session_id=call.session_id,
                line=call.line,
                started_at=call.started_at,
                ended_at=call.ended_at,
                end_reason=call.end_reason,
                exchanges=call.exchanges,
                cost_usd=call.cost_usd,
                initiated_by=call.initiated_by,
            )
        except LedgerError as exc:
            log.warning("advisor-call: could not record call %s: %s", call.call_id, exc)
