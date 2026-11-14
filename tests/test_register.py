"""U-30: register() wires everything, and the root package imports like
Hermes imports it."""

import importlib.util
import json
import pathlib
import sys

from conftest import FakeCtx, reply

from advisor_call.ledger import Ledger
from advisor_call.plugin import HOOKS, register
from advisor_call.prompts import EXECUTOR_GUIDANCE
from advisor_call.settings import parse_settings

ROOT = pathlib.Path(__file__).resolve().parent.parent


def _register(ctx, tmp_path, clock, section=None):
    section = section or {"budget": {"cycle_start_day": 3}}
    return register(
        ctx,
        settings=lambda: parse_settings(section),
        ledger=Ledger.at_path(tmp_path / "l.db"),
        clock=clock,
    )


def test_register_wires_everything(tmp_path, clock):
    ctx = FakeCtx()
    _register(ctx, tmp_path, clock)
    assert set(ctx.tools) == {"advisor_call_open", "advisor_call_say", "advisor_call_hangup"}
    assert {t["toolset"] for t in ctx.tools.values()} == {"advisor_call"}
    assert all(t["schema"]["name"] == n for n, t in ctx.tools.items())
    assert set(ctx.hooks) == set(HOOKS) and all(len(v) == 1 for v in ctx.hooks.values())
    assert list(ctx.middleware) == ["llm_request"]
    assert "phone" in ctx.commands
    assert not ctx.commands["phone"]["args_hint"].startswith("<")  # stays in Telegram's menu
    assert ctx.system_prompt_sections == [{"args": ("advisor-call", EXECUTOR_GUIDANCE)}]
    assert ctx.auxiliary_tasks == {
        "advisor_call_sonnet": {
            "display_name": "Advisor call: sonnet",
            "description": "Advisor model for advisor-call line 'sonnet'",
            "defaults": {
                "provider": "anthropic",
                "model": "claude-sonnet-5-5",
                "timeout": 180.0,
                "extra_body": {"max_tokens": 4000},
            },
        }
    }


def test_one_aux_task_per_line(tmp_path, clock):
    ctx = FakeCtx()
    opus_prices = {"input": 4, "output": 20, "cache_read": 0.2, "cache_write": 5}
    _register(
        ctx,
        tmp_path,
        clock,
        {
            "budget": {"cycle_start_day": 3},
            "lines": {
                "sonnet": {},
                "opus": {"model": "claude-opus-5-5", "prices_per_mtok": opus_prices},
            },
        },
    )
    assert set(ctx.auxiliary_tasks) == {"advisor_call_sonnet", "advisor_call_opus"}
    assert ctx.auxiliary_tasks["advisor_call_opus"]["defaults"]["model"] == "claude-opus-5-5"


def test_register_does_not_touch_llm(tmp_path, clock):
    class NoLlm(FakeCtx):
        @property
        def llm(self):
            raise AttributeError("ctx.llm is not available during register()")

        @llm.setter
        def llm(self, value):
            pass

    _register(NoLlm(), tmp_path, clock)  # must not raise


def test_command_name_fallback(tmp_path, clock):
    class Taken(FakeCtx):
        def register_command(self, name, handler, **kwargs):
            if name == "phone":
                return None
            super().register_command(name, handler, **kwargs)
            return object()

    ctx = Taken()
    _register(ctx, tmp_path, clock)
    assert list(ctx.commands) == ["advisorcall"]


def test_registered_handlers_work_end_to_end(tmp_path, clock):
    ctx = FakeCtx()
    _register(ctx, tmp_path, clock)
    ctx.llm.script(reply("wired"))
    hook = ctx.hooks["pre_tool_call"][0]
    assert (
        hook(tool_name="advisor_call_open", args={"reason": "r", "question": "q"}, session_id="s")[
            "action"
        ]
        == "approve"
    )
    out = json.loads(
        ctx.tools["advisor_call_open"]["handler"](
            {"reason": "r", "question": "q"}, session_id="s", task_id="t"
        )
    )
    assert out["reply"] == "wired"
    assert "📞" in ctx.hooks["transform_llm_output"][0](response_text="ok", session_id="s")


def test_root_package_imports_like_hermes():
    """Hermes imports the plugin directory as package hermes_plugins.advisor_call
    (DESIGN.md, O-6); relative imports must resolve that way."""
    name = "hermes_plugins_test.advisor_call"
    sys.modules.setdefault("hermes_plugins_test", type(sys)("hermes_plugins_test"))
    spec = importlib.util.spec_from_file_location(
        name, ROOT / "__init__.py", submodule_search_locations=[str(ROOT)]
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
        ctx = FakeCtx()
        module.register(ctx)  # Hermes absent: settings fall back to defaults
        assert f"{name}.advisor_call.plugin" in sys.modules
        assert len(ctx.tools) == 3
    finally:
        for key in [k for k in sys.modules if k.startswith("hermes_plugins_test")]:
            del sys.modules[key]
