"""Shared plugin state: one Runtime per loaded plugin.

Holds the call manager, ledger, captured context and the accessors for
settings and Hermes' LLM API. Tools, hooks and the /second-opinion command all work
through it, which is also what tests construct with fakes.
"""

from __future__ import annotations

import datetime as dt
import logging
import time
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
        self.manager = CallManager(clock=self.clock.time, on_call_end=self._persist_call)

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

    # ----------------------------------------------------------------- budget

    def budget(self, settings: Settings) -> BudgetState:
        """Raises ValueError when cycle_start_day is unset and LedgerError
        when spend cannot be read. Callers fail closed on both."""
        budget = settings.budget
        if budget.cycle_start_day is None:
            raise ValueError(budget.cycle_error)
        today = self.today()
        cycle = cycle_for(today, budget.cycle_start_day)
        spent_today, spent_before = self.ledger.totals(today, cycle.start)
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

    def _persist_call(self, call: Call) -> None:
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
