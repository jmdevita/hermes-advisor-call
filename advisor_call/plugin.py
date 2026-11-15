"""register(ctx): wiring only. Called by the root __init__.py."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from .commands import ARGS_HINT, COMMAND, DESCRIPTION, FALLBACK_COMMAND, SecondOpinionCommand
from .hooks import Hooks
from .ledger import Ledger
from .prompts import EXECUTOR_GUIDANCE, SYSTEM_PROMPT_SECTION_ID
from .runtime import Runtime
from .settings import Settings, read_settings
from .tools import TOOLS, TOOLSET, make_handler

log = logging.getLogger(__name__)

HOOKS = ("pre_tool_call", "pre_llm_call", "transform_llm_output", "on_session_finalize")


def register(
    ctx: Any,
    *,
    settings: Callable[[], Settings] = read_settings,
    ledger: Ledger | None = None,
    clock: Any = None,
) -> Runtime:
    """Register tools, hooks, middleware, /second-opinion, the executor guidance and
    one auxiliary task per configured advisor line. Never touches ctx.llm
    here: it is resolved lazily, on the first advisor request."""
    rt = Runtime(
        llm=lambda: ctx.llm, ledger=ledger or Ledger.for_hermes(), settings=settings, clock=clock
    )
    initial = rt.settings()

    for name, schema, fn, emoji in TOOLS:
        ctx.register_tool(
            name=name,
            toolset=TOOLSET,
            schema=schema,
            handler=make_handler(rt, fn),
            description=schema["description"],
            emoji=emoji,
        )

    hooks = Hooks(rt)
    for hook in HOOKS:
        ctx.register_hook(hook, getattr(hooks, hook))
    ctx.register_middleware("llm_request", hooks.llm_request)

    command = SecondOpinionCommand(rt)
    if ctx.register_command(COMMAND, command, description=DESCRIPTION, args_hint=ARGS_HINT) is None:
        log.warning(
            "advisor-call: /%s is taken; registering /%s instead", COMMAND, FALLBACK_COMMAND
        )
        ctx.register_command(
            FALLBACK_COMMAND, command, description=DESCRIPTION, args_hint=ARGS_HINT
        )

    ctx.register_system_prompt_section(SYSTEM_PROMPT_SECTION_ID, EXECUTOR_GUIDANCE)

    # One auxiliary slot per line, shown in Hermes' model picker. max_tokens
    # rides in extra_body because Hermes does not forward complete()'s
    # max_tokens on the native Anthropic route (DESIGN.md, O-3b). Changing
    # lines or max_tokens needs a restart.
    for line in initial.lines.values():
        ctx.register_auxiliary_task(
            line.task_key,
            display_name=f"Advisor call: {line.name}",
            description=f"advisor-call advisor model ({line.name}: {line.model})",
            defaults={
                "provider": line.provider or "auto",
                "model": line.model,
                "timeout": initial.timeout_seconds,
                "extra_body": {"max_tokens": line.max_tokens},
            },
        )
    return rt
