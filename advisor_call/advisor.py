"""Advisor requests: message layout, cache markers, the ctx.llm call.

Layout for exchange n:

    system:    ADVISOR_SYSTEM                                  (static)
    user:      <transcript>...</transcript><executor_message>   <- breakpoint A
    assistant: reply 1
    user:      executor message 2
    ...
    assistant: reply n-1                                       <- breakpoint B
    user:      executor message n                              (uncached tail)

At most two breakpoints. On the first exchange A is on the last message, so
the whole prompt is cached and there is no uncached tail.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .cost import Usage, estimate_tokens
from .prompts import ADVISOR_SYSTEM
from .settings import Line, Settings

_SYSTEM_EST = estimate_tokens(ADVISOR_SYSTEM)


@dataclass(frozen=True)
class AdvisorRequest:
    messages: list[dict[str, Any]]
    # Token estimates (cost.estimate_tokens), not exact counts:
    prompt_est: float  # system + every message
    uncached_est: float  # after the last breakpoint (all of it when caching is off)
    cached_prefix_est: float  # through the last breakpoint; what this request leaves cached


@dataclass(frozen=True)
class AdvisorReply:
    text: str
    usage: Usage
    provider: str
    model: str


class AdvisorError(Exception):
    """The advisor could not give a usable reply.

    usage: set when the provider reported usage.
    billable: False when the usage was not on this line's provider (Hermes
      fell back to another one): it is recorded but not priced by the line.
    bill_estimate: the request may have been processed and billed although no
      usable reply or usage came back (a timeout, an unreadable result); the
      caller bills the pre-flight estimate."""

    def __init__(
        self,
        message: str,
        usage: Usage | None = None,
        *,
        billable: bool = True,
        bill_estimate: bool = False,
        provider: str = "",
    ) -> None:
        super().__init__(message)
        self.message = message
        self.usage = usage
        self.billable = billable
        self.bill_estimate = bill_estimate
        self.provider = provider


def _marker(cache_ttl: str) -> dict[str, str]:
    marker = {"type": "ephemeral"}
    if cache_ttl == "1h":
        marker["ttl"] = "1h"
    return marker


def _marked(role: str, text: str, cache_ttl: str) -> dict[str, Any]:
    return {
        "role": role,
        "content": [{"type": "text", "text": text, "cache_control": _marker(cache_ttl)}],
    }


def build_request(conversation: list[dict[str, str]], cache_ttl: str) -> AdvisorRequest:
    """conversation: the advisor-side messages including the pending
    executor message last (user, assistant, user, ..., user)."""
    if not conversation or conversation[-1].get("role") != "user":
        raise ValueError("the advisor conversation must end with the executor's message")
    caching = cache_ttl != "off"
    marked: set[int] = set()
    if caching:
        marked.add(0)  # A: the opening message
        if len(conversation) >= 3:
            marked.add(len(conversation) - 2)  # B: the most recent reply
    messages: list[dict[str, Any]] = [{"role": "system", "content": ADVISOR_SYSTEM}]
    for i, msg in enumerate(conversation):
        role, text = msg["role"], msg["content"]
        if i in marked:
            messages.append(_marked(role, text, cache_ttl))
        else:
            messages.append({"role": role, "content": text})

    sizes = [_SYSTEM_EST] + [estimate_tokens(m["content"]) for m in conversation]
    prompt_est = sum(sizes)
    if caching:
        last = max(marked) + 1  # index into sizes (offset by the system prompt)
        cached_prefix = sum(sizes[: last + 1])
    else:
        cached_prefix = 0.0
    return AdvisorRequest(
        messages=messages,
        prompt_est=prompt_est,
        uncached_est=prompt_est - cached_prefix,
        cached_prefix_est=cached_prefix,
    )


def ask(llm: Any, line: Line, settings: Settings, request: AdvisorRequest) -> AdvisorReply:
    """One attempt, no retry loop (Hermes itself retries transient errors).

    Raises AdvisorError for any failure, including an empty reply and a
    reply that came from a provider other than the line's (Hermes'
    fallback ladder)."""
    try:
        result = llm.complete(
            request.messages,
            task=line.task_key,
            max_tokens=line.max_tokens,
            timeout=settings.timeout_seconds,
            purpose="advisor-call",
        )
    except Exception as exc:  # noqa: BLE001 - every provider/SDK error ends up here
        raise AdvisorError(_describe(exc), bill_estimate=_is_timeout(exc)) from exc

    try:
        usage = Usage.from_result(getattr(result, "usage", None))
        provider = str(getattr(result, "provider", "") or "")
        model = str(getattr(result, "model", "") or line.model)
        text = getattr(result, "text", None)
    except Exception as exc:  # noqa: BLE001 - a malformed result object
        raise AdvisorError(
            f"unreadable advisor result: {type(exc).__name__}", bill_estimate=True
        ) from exc
    if provider and line.provider and provider.lower() != line.provider.lower():
        # Not the advisor the user approved, and not priced by this line.
        raise AdvisorError(
            f"Hermes fell back to provider {provider!r} instead of {line.provider!r}; "
            "the reply was discarded",
            usage=usage,
            billable=False,
            provider=provider,
        )
    if not isinstance(text, str) or not text.strip():
        raise AdvisorError("the advisor returned an empty reply", usage=usage)
    return AdvisorReply(text=text.strip(), usage=usage, provider=provider, model=model)


def _is_timeout(exc: BaseException) -> bool:
    """A timeout means the request may still have been processed (and billed)."""
    seen: set[int] = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        name = type(exc).__name__.lower()
        if isinstance(exc, TimeoutError) or "timeout" in name or "timed out" in str(exc).lower():
            return True
        exc = exc.__cause__ or exc.__context__
    return False


def _describe(exc: BaseException) -> str:
    message = str(exc).strip() or "no details"
    hint = ""
    low = message.lower()
    if any(s in low for s in ("credit balance", "billing", "insufficient", "402")):
        hint = " Check the credit balance in the Claude Console."
    elif any(s in low for s in ("401", "authentication", "api key", "x-api-key")):
        hint = " Check ANTHROPIC_API_KEY in ~/.hermes/.env."
    elif "not_found" in low or "404" in low or ("model" in low and "not" in low):
        hint = " Check the model name in auxiliary.advisor_call_<line>."
    return f"{type(exc).__name__}: {message[:500]}{hint}"
