"""Plugin settings: defaults, merging, validation.

Settings live under ``plugins.entries.advisor-call`` in the user's
config.yaml. They are read on every call (see ``read_settings``) so budget
edits apply without a restart. Invalid values fall back to the default with
a logged warning; nothing here raises on bad user input.
"""

from __future__ import annotations

import copy
import logging
import math
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger(__name__)

PLUGIN_ID = "advisor-call"

CYCLE_SETUP_MESSAGE = (
    "Set plugins.entries.advisor-call.budget.cycle_start_day to the day of the month "
    "your API credit refreshes."
)

# Shipped so a fresh install works with only an API key. Prices are USD per
# million tokens for Claude Sonnet 5.5 (see DESIGN.md, O-11). cache_write is
# the 5-minute write price; use 4.00 with cache_ttl: 1h.
DEFAULT_LINES: dict[str, dict[str, Any]] = {
    "sonnet": {
        "provider": "anthropic",
        "model": "claude-sonnet-5-5",
        "prices_per_mtok": {
            "input": 2.00,
            "output": 10.00,
            "cache_read": 0.10,
            "cache_write": 2.50,
        },
    }
}

DEFAULTS: dict[str, Any] = {
    "default_line": "sonnet",
    "lines": DEFAULT_LINES,
    "budget": {
        "per_month_usd": 100.00,
        "cycle_start_day": None,  # required; no default on purpose
        "pacing": "dynamic",
        "per_call_usd": 0.50,
        "max_per_call_usd": 0.0,
        "per_day_usd": 3.00,
        "max_per_day_usd": 0.0,
        "max_messages_per_call": 20,
        "idle_timeout_minutes": 30,
    },
    "approval": {"refuse_when_approvals_off": True},
    "context": {
        "opening_transcript_chars": 40000,
        "tool_output_chars": 2000,
        "redact_secrets": True,
    },
    "advisor": {"max_tokens": 4000, "timeout_seconds": 180, "cache_ttl": "5m"},
    "allowed_senders": [],
    "bill_footer": True,
    "log_transcripts": False,
}

PRICE_KEYS = ("input", "output", "cache_read", "cache_write")
CACHE_TTLS = ("5m", "1h", "off")
PACINGS = ("dynamic", "fixed")


@dataclass(frozen=True)
class Prices:
    """USD per million tokens."""

    input: float
    output: float
    cache_read: float
    cache_write: float


@dataclass(frozen=True)
class Line:
    name: str
    provider: str
    model: str
    prices: Prices | None  # None when misconfigured
    problem: str = ""  # why the line cannot be used; "" when usable

    @property
    def usable(self) -> bool:
        return self.prices is not None and not self.problem

    @property
    def task_key(self) -> str:
        return aux_task_key(self.name)


@dataclass(frozen=True)
class Budget:
    per_month_usd: float
    cycle_start_day: int | None  # None when missing or invalid
    pacing: str
    per_call_usd: float
    max_per_call_usd: float
    per_day_usd: float
    max_per_day_usd: float
    max_messages_per_call: int
    idle_timeout_minutes: float

    @property
    def cycle_error(self) -> str:
        return "" if self.cycle_start_day is not None else CYCLE_SETUP_MESSAGE


@dataclass(frozen=True)
class Settings:
    default_line: str
    lines: dict[str, Line]
    budget: Budget
    refuse_when_approvals_off: bool
    opening_transcript_chars: int
    tool_output_chars: int
    redact_secrets: bool
    max_tokens: int
    timeout_seconds: float
    cache_ttl: str
    allowed_senders: tuple[str, ...]
    bill_footer: bool
    log_transcripts: bool
    warnings: tuple[str, ...] = field(default=())

    def line(self, name: str | None) -> Line | None:
        """The named line, or the default line when name is empty."""
        return self.lines.get(name or self.default_line)

    def usable_lines(self) -> list[str]:
        return sorted(n for n, line in self.lines.items() if line.usable)

    @property
    def caching(self) -> bool:
        return self.cache_ttl != "off"


def aux_task_key(line_name: str) -> str:
    return f"advisor_call_{line_name}"


# --------------------------------------------------------------------------- parsing


class _Collector:
    def __init__(self) -> None:
        self.warnings: list[str] = []

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)


def _number(raw: Any, default: float, path: str, w: _Collector, *, allow_zero: bool) -> float:
    if raw is None:
        return default
    ok = isinstance(raw, int | float) and not isinstance(raw, bool) and math.isfinite(raw)
    if ok and (raw > 0 or (allow_zero and raw == 0)):
        return float(raw)
    need = "a number >= 0" if allow_zero else "a positive number"
    w.warn(f"{path} must be {need}; got {raw!r}, using {default!r}")
    return default


def _integer(raw: Any, default: int, path: str, w: _Collector) -> int:
    if raw is None:
        return default
    if isinstance(raw, int) and not isinstance(raw, bool) and raw > 0:
        return raw
    w.warn(f"{path} must be a positive integer; got {raw!r}, using {default!r}")
    return default


def _boolean(raw: Any, default: bool, path: str, w: _Collector) -> bool:
    if raw is None:
        return default
    if isinstance(raw, bool):
        return raw
    w.warn(f"{path} must be true or false; got {raw!r}, using {default!r}")
    return default


def _section(raw: dict[str, Any], key: str, w: _Collector) -> dict[str, Any]:
    value = raw.get(key)
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    w.warn(f"{key} must be a mapping; got {type(value).__name__}, using defaults")
    return {}


def _cycle_start_day(raw: Any, w: _Collector) -> int | None:
    if raw is None:
        w.warn(f"budget.cycle_start_day is not set. {CYCLE_SETUP_MESSAGE}")
        return None
    if isinstance(raw, int) and not isinstance(raw, bool) and 1 <= raw <= 31:
        return raw
    w.warn(f"budget.cycle_start_day must be an integer from 1 to 31; got {raw!r}")
    return None


def _cache_ttl(raw: Any, w: _Collector) -> str:
    if raw is None:
        return DEFAULTS["advisor"]["cache_ttl"]
    if raw is False:  # YAML 1.1 reads a bare `off` as false
        return "off"
    if isinstance(raw, str) and raw.strip().lower() in CACHE_TTLS:
        return raw.strip().lower()
    w.warn(f"advisor.cache_ttl must be one of {', '.join(CACHE_TTLS)}; got {raw!r}, using 5m")
    return DEFAULTS["advisor"]["cache_ttl"]


def _line(name: str, raw: Any, w: _Collector) -> Line:
    base = DEFAULT_LINES.get(name, {})
    if not isinstance(raw, dict):
        if raw is not None:
            w.warn(f"lines.{name} must be a mapping")
        raw = {}
    provider = raw.get("provider") or base.get("provider") or ""
    model = raw.get("model") or base.get("model") or ""
    price_raw = raw.get("prices_per_mtok", base.get("prices_per_mtok"))
    problem = ""
    prices = None
    if not isinstance(price_raw, dict):
        problem = "no prices_per_mtok configured"
    else:
        values: dict[str, float] = {}
        for key in PRICE_KEYS:
            v = price_raw.get(key)
            valid = isinstance(v, int | float) and not isinstance(v, bool) and math.isfinite(v)
            # Input and output must be priced; cache prices may be 0.
            if not valid or v < 0 or (key in ("input", "output") and v == 0):
                problem = f"prices_per_mtok.{key} is missing or invalid ({v!r})"
                break
            values[key] = float(v)
        if not problem:
            prices = Prices(**values)
    if problem:
        w.warn(f"advisor line {name!r} is misconfigured: {problem}")
    return Line(name=name, provider=str(provider), model=str(model), prices=prices, problem=problem)


def parse_settings(raw: Any) -> Settings:
    """Merge the user's ``plugins.entries.advisor-call`` mapping over the
    defaults and validate it. Never raises; problems become warnings."""
    w = _Collector()
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        w.warn(f"plugins.entries.{PLUGIN_ID} must be a mapping; using defaults")
        raw = {}

    lines_raw = raw.get("lines")
    if lines_raw is None:
        lines_raw = copy.deepcopy(DEFAULT_LINES)
    elif not isinstance(lines_raw, dict) or not lines_raw:
        w.warn("lines must be a non-empty mapping; using the default sonnet line")
        lines_raw = copy.deepcopy(DEFAULT_LINES)
    lines = {str(n): _line(str(n), v, w) for n, v in lines_raw.items()}

    default_line = raw.get("default_line") or DEFAULTS["default_line"]
    if not isinstance(default_line, str) or default_line not in lines:
        w.warn(f"default_line {default_line!r} is not a configured line")
        default_line = str(default_line)

    b = _section(raw, "budget", w)
    bd = DEFAULTS["budget"]
    pacing = b.get("pacing", bd["pacing"])
    if pacing not in PACINGS:
        w.warn(f"budget.pacing must be dynamic or fixed; got {pacing!r}, using dynamic")
        pacing = "dynamic"
    budget = Budget(
        per_month_usd=_number(
            b.get("per_month_usd"), bd["per_month_usd"], "budget.per_month_usd", w, allow_zero=False
        ),
        cycle_start_day=_cycle_start_day(b.get("cycle_start_day"), w),
        pacing=pacing,
        per_call_usd=_number(
            b.get("per_call_usd"), bd["per_call_usd"], "budget.per_call_usd", w, allow_zero=False
        ),
        max_per_call_usd=_number(
            b.get("max_per_call_usd"),
            bd["max_per_call_usd"],
            "budget.max_per_call_usd",
            w,
            allow_zero=True,
        ),
        per_day_usd=_number(
            b.get("per_day_usd"), bd["per_day_usd"], "budget.per_day_usd", w, allow_zero=False
        ),
        max_per_day_usd=_number(
            b.get("max_per_day_usd"),
            bd["max_per_day_usd"],
            "budget.max_per_day_usd",
            w,
            allow_zero=True,
        ),
        max_messages_per_call=_integer(
            b.get("max_messages_per_call"),
            bd["max_messages_per_call"],
            "budget.max_messages_per_call",
            w,
        ),
        idle_timeout_minutes=_number(
            b.get("idle_timeout_minutes"),
            bd["idle_timeout_minutes"],
            "budget.idle_timeout_minutes",
            w,
            allow_zero=False,
        ),
    )

    a = _section(raw, "approval", w)
    c = _section(raw, "context", w)
    cd = DEFAULTS["context"]
    adv = _section(raw, "advisor", w)
    ad = DEFAULTS["advisor"]

    senders_raw = raw.get("allowed_senders")
    if senders_raw is None:
        senders: tuple[str, ...] = ()
    elif isinstance(senders_raw, list | tuple):
        senders = tuple(str(s).strip() for s in senders_raw if str(s).strip())
    else:
        w.warn("allowed_senders must be a list of user IDs; ignoring it")
        senders = ()

    settings = Settings(
        default_line=default_line,
        lines=lines,
        budget=budget,
        refuse_when_approvals_off=_boolean(
            a.get("refuse_when_approvals_off"), True, "approval.refuse_when_approvals_off", w
        ),
        opening_transcript_chars=_integer(
            c.get("opening_transcript_chars"),
            cd["opening_transcript_chars"],
            "context.opening_transcript_chars",
            w,
        ),
        tool_output_chars=_integer(
            c.get("tool_output_chars"), cd["tool_output_chars"], "context.tool_output_chars", w
        ),
        redact_secrets=_boolean(c.get("redact_secrets"), True, "context.redact_secrets", w),
        max_tokens=_integer(adv.get("max_tokens"), ad["max_tokens"], "advisor.max_tokens", w),
        timeout_seconds=_number(
            adv.get("timeout_seconds"),
            ad["timeout_seconds"],
            "advisor.timeout_seconds",
            w,
            allow_zero=False,
        ),
        cache_ttl=_cache_ttl(adv.get("cache_ttl"), w),
        allowed_senders=senders,
        bill_footer=_boolean(raw.get("bill_footer"), True, "bill_footer", w),
        log_transcripts=_boolean(raw.get("log_transcripts"), False, "log_transcripts", w),
        warnings=tuple(w.warnings),
    )
    return settings


# --------------------------------------------------------------------------- Hermes


_logged_warnings: set[str] = set()


def _log_once(warnings: tuple[str, ...]) -> None:
    for msg in warnings:
        if msg not in _logged_warnings:
            _logged_warnings.add(msg)
            log.warning("advisor-call: %s", msg)


def plugin_section(config: dict[str, Any]) -> Any:
    """``plugins.entries.advisor-call`` from a full Hermes config mapping."""
    plugins = config.get("plugins") if isinstance(config, dict) else None
    entries = plugins.get("entries") if isinstance(plugins, dict) else None
    return entries.get(PLUGIN_ID) if isinstance(entries, dict) else None


def read_settings() -> Settings:
    """Read settings from the live Hermes config. Never mutates Hermes'
    cached config (we deep-copy our section). Falls back to defaults if the
    config cannot be read, which leaves cycle_start_day unset and therefore
    refuses spending."""
    try:
        from hermes_cli.config import load_config_readonly

        section = copy.deepcopy(plugin_section(load_config_readonly()))
    except Exception as exc:  # noqa: BLE001 - config read must never break a turn
        log.warning("advisor-call: could not read config (%s); spending is refused", exc)
        section = None
    settings = parse_settings(section)
    _log_once(settings.warnings)
    return settings
