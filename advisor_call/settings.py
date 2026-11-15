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
import re
from dataclasses import dataclass, field
from typing import Any

from .suggest import MODES as SUGGEST_MODES
from .suggest import SuggestSettings

log = logging.getLogger(__name__)

PLUGIN_ID = "advisor-call"

CYCLE_SETUP_MESSAGE = (
    "Set plugins.entries.advisor-call.budget.cycle_start_day to the day of the month "
    "your API credit refreshes."
)

# Models with shipped prices, USD per million tokens (DESIGN.md, O-11;
# Anthropic's published prices, 2026-11). The cache-write price is derived
# from the cache TTL (CACHE_WRITE_MULTIPLIER). Users can override any price,
# or price another model, under advisor.prices_per_mtok.
KNOWN_MODELS: dict[str, dict[str, float]] = {
    "claude-sonnet-5-5": {"input": 2.00, "output": 10.00, "cache_read": 0.10},
    "claude-opus-5-5": {"input": 4.00, "output": 20.00, "cache_read": 0.20},
}
MODEL_ALIASES: dict[str, str] = {"sonnet": "claude-sonnet-5-5", "opus": "claude-opus-5-5"}
DEFAULT_MODEL = "sonnet"
CACHE_WRITE_MULTIPLIER = {"5m": 1.25, "1h": 2.0, "off": 1.25}
PROVIDER = "anthropic"

DEFAULTS: dict[str, Any] = {
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
    "advisor": {
        "model": DEFAULT_MODEL,
        "also_allow": [],
        "prices_per_mtok": {},
        "max_tokens": 4000,
        "timeout_seconds": 180,
        "cache_ttl": "5m",
    },
    "suggest": {
        "mode": "both",
        "error_streak": 3,
        "same_error": 3,
        "loop": 3,
        "cycles": True,
        "cooldown_calls": 10,
        "max_per_session": 3,
        "ignore_tools": [],
    },
    "allowed_senders": [],
    "bill_footer": True,
    "log_transcripts": False,
}

# Below this, thinking can use the whole reply limit (seen live, DESIGN.md 7).
MIN_USEFUL_MAX_TOKENS = 1024

PRICE_KEYS = ("input", "output", "cache_read")  # cache_write: optional, derived
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
    """One callable advisor model. ``name`` is the short name used in
    approvals, grants, the ledger and Hermes' auxiliary slot
    (``auxiliary.advisor_call_<name>``): the alias for a shipped model
    (``sonnet``), else the model id with non-alphanumerics as ``_``."""

    name: str
    provider: str
    model: str
    prices: Prices | None  # None when misconfigured
    problem: str = ""  # why the line cannot be used; "" when usable
    max_tokens: int = 4000  # effective reply limit for this line

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
    approvals_mode: str = "manual"  # Hermes' approvals.mode: manual | smart | off
    suggest: SuggestSettings = field(default_factory=SuggestSettings)
    warnings: tuple[str, ...] = field(default=())

    def line(self, name: str | None) -> Line | None:
        """The advisor for a name, alias or full model id; the default
        advisor when name is empty. None when that model is not allowed."""
        if not name:
            return self.lines.get(self.default_line)
        return self.lines.get(name) or self.lines.get(resolve_model(name)[0])

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


def _count(raw: Any, default: int, path: str, w: _Collector) -> int:
    """A whole number >= 0 (0 turns the rule off)."""
    if raw is None:
        return default
    if isinstance(raw, int) and not isinstance(raw, bool) and raw >= 0:
        return raw
    w.warn(f"{path} must be a whole number >= 0; got {raw!r}, using {default!r}")
    return default


def _suggest(raw: Any, w: _Collector) -> SuggestSettings:
    d = DEFAULTS["suggest"]
    if raw is None:
        raw = {}
    if raw is False:  # `suggest: off`
        raw = {"mode": "off"}
    if not isinstance(raw, dict):
        w.warn("suggest must be a mapping; using defaults")
        raw = {}
    mode = raw.get("mode", d["mode"])
    if mode is False:  # YAML `mode: off`
        mode = "off"
    if not isinstance(mode, str) or mode.strip().lower() not in SUGGEST_MODES:
        w.warn(f"suggest.mode must be one of {', '.join(SUGGEST_MODES)}; got {mode!r}, using both")
        mode = d["mode"]
    ignore = raw.get("ignore_tools") or []
    if isinstance(ignore, str):
        ignore = [ignore]
    if not isinstance(ignore, list | tuple):
        w.warn("suggest.ignore_tools must be a list of tool names; ignoring it")
        ignore = []
    return SuggestSettings(
        mode=mode.strip().lower(),
        error_streak=_count(raw.get("error_streak"), d["error_streak"], "suggest.error_streak", w),
        same_error=_count(raw.get("same_error"), d["same_error"], "suggest.same_error", w),
        loop=_count(raw.get("loop"), d["loop"], "suggest.loop", w),
        cycles=_boolean(raw.get("cycles"), d["cycles"], "suggest.cycles", w),
        cooldown_calls=_count(
            raw.get("cooldown_calls"), d["cooldown_calls"], "suggest.cooldown_calls", w
        ),
        max_per_session=_count(
            raw.get("max_per_session"), d["max_per_session"], "suggest.max_per_session", w
        ),
        ignore_tools=frozenset(str(t).strip() for t in ignore if str(t).strip()),
    )


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


def resolve_model(ref: str) -> tuple[str, str]:
    """(short name, model id) for an alias or a model id."""
    low = ref.strip().lower()
    if low in MODEL_ALIASES:
        return low, MODEL_ALIASES[low]
    for alias, model_id in MODEL_ALIASES.items():
        if low == model_id:
            return alias, model_id
    return re.sub(r"[^a-z0-9_]", "_", low), ref.strip()


def _price_value(v: Any, key: str) -> float | None:
    if not (isinstance(v, int | float) and not isinstance(v, bool) and math.isfinite(v)):
        return None
    if v < 0 or (key in ("input", "output") and v == 0):  # cache prices may be 0
        return None
    return float(v)


def _model_line(ref: str, overrides: dict[str, Any], cache_ttl: str, w: _Collector) -> Line:
    name, model_id = resolve_model(ref)
    given = overrides.get(model_id, overrides.get(name, overrides.get(ref)))
    if given is not None and not isinstance(given, dict):
        w.warn(f"advisor.prices_per_mtok.{model_id} must be a mapping; ignoring it")
        given = None
    raw = {**KNOWN_MODELS.get(model_id, {}), **(given or {})}
    problem = ""
    values: dict[str, float] = {}
    if not raw:
        problem = (
            f"no prices for model {model_id!r}; add them under "
            f"advisor.prices_per_mtok.{model_id} (input, output, cache_read)"
        )
    else:
        for key in PRICE_KEYS:
            value = _price_value(raw.get(key), key)
            if value is None:
                problem = f"price {key!r} for {model_id!r} is missing or invalid ({raw.get(key)!r})"
                break
            values[key] = value
    prices = None
    if not problem:
        write = (
            _price_value(raw.get("cache_write"), "cache_write") if "cache_write" in raw else None
        )
        if write is None:
            write = values["input"] * CACHE_WRITE_MULTIPLIER.get(cache_ttl, 1.25)
        prices = Prices(cache_write=write, **values)
    else:
        w.warn(f"advisor model {model_id!r} cannot be used: {problem}")
    return Line(name=name, provider=PROVIDER, model=model_id, prices=prices, problem=problem)


def _apply_auxiliary(line: Line, auxiliary: Any, max_tokens: int, w: _Collector) -> Line:
    """Apply Hermes' ``auxiliary.<task>`` slot to a line.

    - The slot's model and provider win over the line's defaults.
    - The plugin caps reply length through the slot's ``extra_body.max_tokens``
      (DESIGN.md, O-3b). A user ``extra_body`` *replaces* the plugin's instead
      of merging with it, so a user ``extra_body`` without ``max_tokens``
      would leave replies unbounded: the line is then refused. A user
      ``max_tokens`` there becomes the line's reply limit.
    """
    provider, model, limit, problem = line.provider, line.model, max_tokens, line.problem
    slot = auxiliary.get(line.task_key) if isinstance(auxiliary, dict) else None
    if isinstance(slot, dict):
        if isinstance(slot.get("provider"), str) and slot["provider"]:
            provider = slot["provider"]
        if isinstance(slot.get("model"), str) and slot["model"]:
            model = slot["model"]
        extra = slot.get("extra_body")
        if isinstance(extra, dict) and extra:
            user_limit = extra.get("max_tokens")
            if isinstance(user_limit, int) and not isinstance(user_limit, bool) and user_limit > 0:
                limit = user_limit
            elif not problem:
                problem = (
                    f"auxiliary.{line.task_key}.extra_body replaces the plugin's reply "
                    f"limit; add max_tokens: {max_tokens} to it"
                )
                w.warn(f"advisor model {line.name!r} cannot be used: {problem}")
    return Line(
        name=line.name,
        provider=provider,
        model=model,
        prices=line.prices,
        problem=problem,
        max_tokens=limit,
    )


APPROVAL_MODES = ("manual", "smart", "off")


def normalize_approvals_mode(mode: Any) -> str:
    """Same rules as Hermes' tools.approval_context._normalize_approval_mode:
    YAML `off` arrives as False; unknown values mean manual."""
    if isinstance(mode, bool):
        return "off" if mode is False else "manual"
    if isinstance(mode, str) and mode.strip().lower() in APPROVAL_MODES:
        return mode.strip().lower()
    return "manual"


def parse_settings(raw: Any, auxiliary: Any = None, approvals: Any = None) -> Settings:
    """Merge the user's ``plugins.entries.advisor-call`` mapping over the
    defaults and validate it. ``auxiliary`` is Hermes' ``auxiliary`` config
    block, used for each line's effective model and provider. Never raises;
    problems become warnings."""
    w = _Collector()
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        w.warn(f"plugins.entries.{PLUGIN_ID} must be a mapping; using defaults")
        raw = {}

    for old in ("lines", "default_line"):
        if old in raw:
            w.warn(
                f"{old} is no longer used; choose the advisor with advisor.model "
                "(and advisor.also_allow for more than one)"
            )

    adv = _section(raw, "advisor", w)
    ad = DEFAULTS["advisor"]
    cache_ttl = _cache_ttl(adv.get("cache_ttl"), w)
    overrides = adv.get("prices_per_mtok") or {}
    if not isinstance(overrides, dict):
        w.warn("advisor.prices_per_mtok must be a mapping of model to prices; ignoring it")
        overrides = {}
    model_ref = adv.get("model", ad["model"])
    if not isinstance(model_ref, str) or not model_ref.strip():
        w.warn(f"advisor.model must be a model name such as sonnet or opus; got {model_ref!r}")
        model_ref = ad["model"]
    also = adv.get("also_allow") or []
    if isinstance(also, str):
        also = [also]
    if not isinstance(also, list | tuple):
        w.warn("advisor.also_allow must be a list of model names; ignoring it")
        also = []
    lines: dict[str, Line] = {}
    for ref in [model_ref, *[str(x) for x in also if str(x).strip()]]:
        line = _model_line(ref, overrides, cache_ttl, w)
        lines.setdefault(line.name, line)
    default_line = resolve_model(model_ref)[0]

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
    senders_raw = raw.get("allowed_senders")
    if senders_raw is None:
        senders: tuple[str, ...] = ()
    elif isinstance(senders_raw, list | tuple):
        senders = tuple(str(s).strip() for s in senders_raw if str(s).strip())
    else:
        w.warn("allowed_senders must be a list of user IDs; ignoring it")
        senders = ()

    max_tokens = _integer(adv.get("max_tokens"), ad["max_tokens"], "advisor.max_tokens", w)
    if max_tokens < MIN_USEFUL_MAX_TOKENS:
        w.warn(
            f"advisor.max_tokens is {max_tokens}: current Claude models think before "
            "answering and the thinking counts against this limit, so replies may come "
            f"back empty. Use at least {MIN_USEFUL_MAX_TOKENS}."
        )
    lines = {n: _apply_auxiliary(line, auxiliary, max_tokens, w) for n, line in lines.items()}

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
        max_tokens=max_tokens,
        timeout_seconds=_number(
            adv.get("timeout_seconds"),
            ad["timeout_seconds"],
            "advisor.timeout_seconds",
            w,
            allow_zero=False,
        ),
        cache_ttl=cache_ttl,
        allowed_senders=senders,
        bill_footer=_boolean(raw.get("bill_footer"), True, "bill_footer", w),
        log_transcripts=_boolean(raw.get("log_transcripts"), False, "log_transcripts", w),
        suggest=_suggest(raw.get("suggest"), w),
        approvals_mode=normalize_approvals_mode(
            approvals.get("mode") if isinstance(approvals, dict) else None
        ),
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

        config = load_config_readonly()
        section = copy.deepcopy(plugin_section(config))
        auxiliary = copy.deepcopy(config.get("auxiliary")) if isinstance(config, dict) else None
        approvals = copy.deepcopy(config.get("approvals")) if isinstance(config, dict) else None
    except Exception as exc:  # noqa: BLE001 - config read must never break a turn
        log.warning("advisor-call: could not read config (%s); spending is refused", exc)
        section = auxiliary = approvals = None
    settings = parse_settings(section, auxiliary, approvals)
    _log_once(settings.warnings)
    return settings
