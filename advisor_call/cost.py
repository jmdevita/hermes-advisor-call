"""Pricing, estimates, billing cycle, daily allowance and per-call cap.

Pure functions; no Hermes, no I/O. Dates are local dates of the Hermes host.
"""

from __future__ import annotations

import calendar
import datetime as dt
from dataclasses import dataclass
from typing import Any

from .settings import Budget, Prices

CHARS_PER_TOKEN = 3.0  # deliberately pessimistic (English is ~4)
_MTOK = 1_000_000.0


# --------------------------------------------------------------------------- usage


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0

    @classmethod
    def from_result(cls, usage: Any) -> Usage:
        """Read a Hermes PluginLlmUsage (or any object with those fields)."""

        def g(name: str) -> int:
            try:
                return max(0, int(getattr(usage, name, 0) or 0))
            except (TypeError, ValueError):
                return 0

        return cls(
            g("input_tokens"), g("output_tokens"), g("cache_read_tokens"), g("cache_write_tokens")
        )

    def add(self, other: Usage) -> None:
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.cache_read_tokens += other.cache_read_tokens
        self.cache_write_tokens += other.cache_write_tokens


def estimate_tokens(chars: int) -> int:
    return int(chars / CHARS_PER_TOKEN) + 1


def exact_cost(usage: Usage, prices: Prices) -> float:
    return (
        usage.input_tokens * prices.input
        + usage.cache_read_tokens * prices.cache_read
        + usage.cache_write_tokens * prices.cache_write
        + usage.output_tokens * prices.output
    ) / _MTOK


@dataclass(frozen=True)
class Charge:
    cost_usd: float
    pessimistic: bool  # True when cache usage had to be assumed


def charge(usage: Usage, prices: Prices, *, caching: bool, prompt_chars: int) -> Charge:
    """What to bill for one exchange.

    Hermes v0.23.1 drops Anthropic's cache counts (DESIGN.md, O-2). When
    caching is on and no cache counts came back, every prompt token we
    estimate was sent but Hermes did not report as input is billed as a
    cache write, the most expensive way it could have been billed. When
    cache counts are present, or caching is off, the bill is exact.
    """
    if not caching or usage.cache_read_tokens or usage.cache_write_tokens:
        return Charge(exact_cost(usage, prices), pessimistic=False)
    unreported = max(0, estimate_tokens(prompt_chars) - usage.input_tokens)
    assumed = Usage(usage.input_tokens, usage.output_tokens, 0, unreported)
    return Charge(exact_cost(assumed, prices), pessimistic=unreported > 0)


def estimate_cost(prompt_chars: int, max_tokens: int, prices: Prices, *, caching: bool) -> float:
    """Pre-flight upper bound for one exchange: the whole prompt at the
    highest input-side price we might bill, plus a full max_tokens reply."""
    input_price = max(prices.input, prices.cache_write) if caching else prices.input
    return (estimate_tokens(prompt_chars) * input_price + max_tokens * prices.output) / _MTOK


# --------------------------------------------------------------------------- cycle


def _clamped(year: int, month: int, day: int) -> dt.date:
    return dt.date(year, month, min(day, calendar.monthrange(year, month)[1]))


def _shift_month(year: int, month: int, delta: int) -> tuple[int, int]:
    index = year * 12 + (month - 1) + delta
    return index // 12, index % 12 + 1


@dataclass(frozen=True)
class Cycle:
    start: dt.date
    end: dt.date  # inclusive

    @property
    def days(self) -> int:
        return (self.end - self.start).days + 1

    def days_left(self, today: dt.date) -> int:
        """Days from today through the last day, inclusive."""
        return (self.end - today).days + 1

    def describe(self, today: dt.date) -> str:
        left = self.days_left(today)
        return (
            f"Cycle: {self.start:%b} {self.start.day} to {self.end:%b} {self.end.day} "
            f"({left} day{'s' if left != 1 else ''} left)"
        )


def cycle_for(today: dt.date, start_day: int) -> Cycle:
    """The billing cycle containing today. A cycle starts on start_day (or
    the month's last day, if the month is shorter) and ends the day before
    the next start."""
    start = _clamped(today.year, today.month, start_day)
    if start > today:
        y, m = _shift_month(today.year, today.month, -1)
        start = _clamped(y, m, start_day)
    ny, nm = _shift_month(start.year, start.month, 1)
    next_start = _clamped(ny, nm, start_day)
    return Cycle(start=start, end=next_start - dt.timedelta(days=1))


# --------------------------------------------------------------------------- budget


@dataclass(frozen=True)
class BudgetState:
    """Everything the caps need, for one moment. Money in USD."""

    cycle: Cycle
    today: dt.date
    spent_today: float
    spent_before_today: float  # this cycle, excluding today
    allowance: float  # today's allowance (fixed for the day)
    baseline: float  # per_month / days in cycle
    per_month: float

    @property
    def spent_cycle(self) -> float:
        return self.spent_before_today + self.spent_today

    @property
    def remaining_today(self) -> float:
        return max(0.0, self.allowance - self.spent_today)

    @property
    def remaining_cycle(self) -> float:
        return max(0.0, self.per_month - self.spent_cycle)

    @property
    def available(self) -> float:
        """The most that may be spent right now."""
        return min(self.remaining_today, self.remaining_cycle)

    @property
    def days_left(self) -> int:
        return self.cycle.days_left(self.today)


def daily_allowance(
    budget: Budget, cycle: Cycle, today: dt.date, spent_before_today: float
) -> float:
    if budget.pacing == "fixed":
        allowance = budget.per_day_usd
    else:
        allowance = (budget.per_month_usd - spent_before_today) / cycle.days_left(today)
        if budget.max_per_day_usd > 0:
            allowance = min(allowance, budget.max_per_day_usd)
    return max(0.0, allowance)


def budget_state(
    budget: Budget, today: dt.date, *, spent_today: float, spent_before_today: float
) -> BudgetState:
    """Needs budget.cycle_start_day; callers refuse spending when it is unset."""
    if budget.cycle_start_day is None:
        raise ValueError("cycle_start_day is not set")
    cycle = cycle_for(today, budget.cycle_start_day)
    return BudgetState(
        cycle=cycle,
        today=today,
        spent_today=spent_today,
        spent_before_today=spent_before_today,
        allowance=daily_allowance(budget, cycle, today, spent_before_today),
        baseline=budget.per_month_usd / cycle.days,
        per_month=budget.per_month_usd,
    )


def call_cap(budget: Budget, state: BudgetState, custom_cap: float | None = None) -> float:
    """Per-call bill cap, computed once when a call opens.

    Dynamic pacing scales the base cap by how far today's allowance is above
    or below the baseline daily share. A custom cap from /phone allow
    replaces the computed one. Never more than can be spent right now."""
    if custom_cap is not None:
        cap = custom_cap
    elif budget.pacing == "dynamic" and state.baseline > 0:
        cap = budget.per_call_usd * (state.allowance / state.baseline)
        if budget.max_per_call_usd > 0:
            cap = min(cap, budget.max_per_call_usd)
    else:
        cap = budget.per_call_usd
    return max(0.0, min(cap, state.available))


def preflight_refusal(
    *,
    call_cost: float,
    call_cap_usd: float,
    exchanges: int,
    max_messages: int,
    est_cost: float,
    state: BudgetState,
    reserved_elsewhere: float = 0.0,
) -> str:
    """End reason if the next exchange must not be sent, else "".

    reserved_elsewhere is the estimated cost of other exchanges in flight
    right now (other sessions), which the ledger does not show yet."""
    if exchanges >= max_messages:
        return "turn_limit"
    if call_cost + est_cost > call_cap_usd:
        return "bill_cap"
    if state.spent_today + reserved_elsewhere + est_cost > state.allowance:
        return "day_cap"
    if state.spent_cycle + reserved_elsewhere + est_cost > state.per_month:
        return "month_cap"
    return ""
