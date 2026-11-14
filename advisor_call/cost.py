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
    estimated: bool  # True when part of the bill is an estimate (see charge())


# Cache lifetimes, with a safety margin so a prefix close to expiry is
# treated as rewritten rather than read.
CACHE_TTL_SECONDS = {"5m": 300.0, "1h": 3600.0}
CACHE_TTL_MARGIN_SECONDS = 30.0


def reusable_prefix_chars(
    cached_chars: int, cached_at: float | None, now: float, cache_ttl: str
) -> int:
    """How much of the prompt we expect Anthropic to still have cached: the
    prefix written by the previous successful exchange, if it is young
    enough. Zero when caching is off or the entry has (or may have) expired."""
    ttl = CACHE_TTL_SECONDS.get(cache_ttl)
    if not ttl or cached_at is None or cached_chars <= 0:
        return 0
    if now - cached_at >= ttl - CACHE_TTL_MARGIN_SECONDS:
        return 0
    return cached_chars


# Token-per-character ratio used to size the cached part of a prompt.
DEFAULT_TOKENS_PER_CHAR = 1 / CHARS_PER_TOKEN
MIN_TOKENS_PER_CHAR = 1 / 4.0  # never assume text is sparser than typical English
MAX_TOKENS_PER_CHAR = 1 / 2.0
MIN_CALIBRATION_CHARS = 400


def tokens_per_char(input_tokens: int, uncached_chars: int) -> float:
    """Ratio measured on the part of the prompt Hermes did report (the
    uncached tail after the last cache breakpoint), clamped. Falls back to
    the deliberately high default when the tail is too short to measure."""
    if uncached_chars < MIN_CALIBRATION_CHARS or input_tokens <= 0:
        return DEFAULT_TOKENS_PER_CHAR
    ratio = input_tokens / uncached_chars
    return min(MAX_TOKENS_PER_CHAR, max(MIN_TOKENS_PER_CHAR, ratio))


def charge(
    usage: Usage,
    prices: Prices,
    *,
    caching: bool,
    prompt_chars: int,
    uncached_chars: int = 0,
    reused_chars: int = 0,
) -> Charge:
    """What to bill for one exchange.

    Exact when caching is off (Hermes' input_tokens is then the whole
    prompt) or when Hermes reports cache counts.

    Otherwise an estimate. Hermes v0.23.1 drops Anthropic's cache counts
    (DESIGN.md, O-2) and reports only the uncached tail as input. The cached
    part (prompt_chars - uncached_chars) is sized with the token ratio
    measured on that tail, then split using what we sent before: the reused
    prefix (reused_chars) is billed at the normal input price, more than a
    cache read costs; the rest at the cache-write price. An unexpected cache
    miss under-bills only the reused prefix, by the difference between the
    write and input prices.
    """
    if not caching or usage.cache_read_tokens or usage.cache_write_tokens:
        return Charge(exact_cost(usage, prices), estimated=False)
    cached_chars = max(0, prompt_chars - uncached_chars)
    if cached_chars == 0:
        return Charge(exact_cost(usage, prices), estimated=False)
    ratio = tokens_per_char(usage.input_tokens, uncached_chars)
    reused = int(min(reused_chars, cached_chars) * ratio)
    written = int(cached_chars * ratio) + 1 - reused
    cost = (
        (usage.input_tokens + reused) * prices.input
        + written * prices.cache_write
        + usage.output_tokens * prices.output
    ) / _MTOK
    return Charge(cost, estimated=True)


def estimate_cost(
    prompt_chars: int, max_tokens: int, prices: Prices, *, caching: bool, reused_chars: int = 0
) -> float:
    """Pre-flight upper bound for one exchange, using the same arithmetic as
    charge() at the high default token ratio: the reused prefix at the input
    price, the rest of the prompt at the highest input-side price we might
    bill, plus a full max_tokens reply."""
    ratio = DEFAULT_TOKENS_PER_CHAR
    total = int(max(0, prompt_chars) * ratio) + 1
    if not caching:
        return (total * prices.input + max_tokens * prices.output) / _MTOK
    reused = min(total, int(min(reused_chars, prompt_chars) * ratio)) if reused_chars > 0 else 0
    rest_price = max(prices.input, prices.cache_write)
    return (
        reused * prices.input + (total - reused) * rest_price + max_tokens * prices.output
    ) / _MTOK


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
