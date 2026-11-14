"""Shared test doubles. Nothing here imports Hermes.

- FakeCtx: records every registration a plugin's register(ctx) makes.
- FakeLlm: returns scripted replies with usage, and records each request.
- FakeClock: a controllable time source for idle expiry and day boundaries.
- db_path: a temporary SQLite path for the ledger.
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass, field
from typing import Any

import pytest

# --------------------------------------------------------------------------- LLM


@dataclass
class FakeUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0


@dataclass
class FakeResult:
    text: str
    provider: str = "anthropic"
    model: str = "claude-sonnet-5-5"
    usage: FakeUsage = field(default_factory=FakeUsage)


@dataclass
class FakeLlmCall:
    messages: list[dict[str, Any]]
    kwargs: dict[str, Any]


class FakeLlm:
    """Stands in for ctx.llm. Queue replies with .script(); each complete()
    pops one. A queued Exception is raised instead of returned."""

    def __init__(self) -> None:
        self._queue: list[FakeResult | BaseException] = []
        self.calls: list[FakeLlmCall] = []

    def script(self, *items: FakeResult | BaseException | str) -> FakeLlm:
        for item in items:
            self._queue.append(FakeResult(text=item) if isinstance(item, str) else item)
        return self

    def complete(self, messages: list[dict[str, Any]], **kwargs: Any) -> FakeResult:
        self.calls.append(FakeLlmCall(messages=messages, kwargs=kwargs))
        if not self._queue:
            raise AssertionError("FakeLlm.complete called with no scripted reply left")
        item = self._queue.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item


# --------------------------------------------------------------------------- ctx


class FakeCtx:
    """Records registrations. Method names mirror Hermes' PluginContext; the
    exact signatures are pinned against v0.23.1 in DESIGN.md."""

    def __init__(self, llm: FakeLlm | None = None) -> None:
        self.llm = llm or FakeLlm()
        self.tools: dict[str, dict[str, Any]] = {}
        self.hooks: dict[str, list[Any]] = {}
        self.middleware: dict[str, list[Any]] = {}
        self.commands: dict[str, dict[str, Any]] = {}
        self.system_prompt_sections: list[dict[str, Any]] = []
        self.auxiliary_tasks: dict[str, dict[str, Any]] = {}

    def register_tool(self, name, toolset, schema, handler, **kwargs):
        self.tools[name] = {"toolset": toolset, "schema": schema, "handler": handler, **kwargs}

    def register_hook(self, name, fn):
        self.hooks.setdefault(name, []).append(fn)

    def register_middleware(self, name, fn, **kwargs):
        self.middleware.setdefault(name, []).append(fn)

    def register_command(self, name, handler, **kwargs):
        self.commands[name] = {"handler": handler, **kwargs}

    def register_system_prompt_section(self, *args, **kwargs):
        self.system_prompt_sections.append({"args": args, **kwargs})

    def register_auxiliary_task(self, key, **kwargs):
        self.auxiliary_tasks[key] = kwargs


# --------------------------------------------------------------------------- clock


class FakeClock:
    """time() returns epoch seconds; today() returns the local date. Advance
    with .advance(seconds=...) or jump with .set(datetime)."""

    def __init__(self, start: _dt.datetime | None = None) -> None:
        self._now = start or _dt.datetime(2026, 11, 14, 12, 0, 0)

    def time(self) -> float:
        return self._now.timestamp()

    def now(self) -> _dt.datetime:
        return self._now

    def today(self) -> _dt.date:
        return self._now.date()

    def advance(self, **delta: float) -> None:
        self._now += _dt.timedelta(**delta)

    def set(self, when: _dt.datetime) -> None:
        self._now = when


# --------------------------------------------------------------------------- fixtures


@pytest.fixture
def fake_llm() -> FakeLlm:
    return FakeLlm()


@pytest.fixture
def fake_ctx(fake_llm: FakeLlm) -> FakeCtx:
    return FakeCtx(llm=fake_llm)


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def db_path(tmp_path):
    return tmp_path / "advisor-call.sqlite3"


# --------------------------------------------------------------------------- runtime


@pytest.fixture
def config() -> dict[str, Any]:
    """The plugin's config section; tests edit it in place (read per call)."""
    return {"budget": {"cycle_start_day": 3}}


@pytest.fixture
def runtime(fake_llm, db_path, clock, config):
    from advisor_call.ledger import Ledger
    from advisor_call.runtime import Runtime
    from advisor_call.settings import parse_settings

    return Runtime(
        llm=lambda: fake_llm,
        ledger=Ledger.at_path(db_path),
        settings=lambda: parse_settings(config),
        clock=clock,
        redactor=lambda: None,
    )


def reply(
    text: str = "advice",
    *,
    input_tokens: int = 500,
    output_tokens: int = 200,
    cache_read: int = 0,
    cache_write: int = 0,
    provider: str = "anthropic",
) -> FakeResult:
    return FakeResult(
        text=text,
        provider=provider,
        usage=FakeUsage(input_tokens, output_tokens, cache_read, cache_write),
    )
