"""U-1 settings merge and validation; part of U-37."""

from advisor_call.settings import CYCLE_SETUP_MESSAGE, aux_task_key, parse_settings

VALID = {"budget": {"cycle_start_day": 3}}


def test_defaults_when_empty():
    s = parse_settings(None)
    assert s.default_line == "sonnet"
    sonnet = s.line(None)
    assert sonnet.usable and sonnet.provider == "anthropic" and sonnet.model == "claude-sonnet-5-5"
    assert sonnet.prices.cache_read == 0.10  # DESIGN.md O-11, not the plan's 0.20
    assert s.cache_ttl == "5m" and s.caching
    assert s.budget.per_month_usd == 100.0 and s.budget.pacing == "dynamic"
    assert s.max_tokens == 4000
    assert sonnet.task_key == aux_task_key("sonnet") == "advisor_call_sonnet"


def test_merge_over_defaults_and_unknown_keys_ignored():
    s = parse_settings(
        {
            "budget": {"cycle_start_day": 3, "per_call_usd": 0.25},
            "surprise": 1,
            "context": {"tool_output_chars": 500, "nope": True},
        }
    )
    assert s.budget.per_call_usd == 0.25
    assert s.budget.per_month_usd == 100.0  # untouched default
    assert s.tool_output_chars == 500
    assert s.opening_transcript_chars == 40000
    assert s.warnings == ()


def test_invalid_values_fall_back_with_warning():
    s = parse_settings(
        {
            "budget": {
                "cycle_start_day": 3,
                "per_call_usd": -1,
                "max_messages_per_call": 0,
                "pacing": "weird",
                "per_month_usd": True,
            },
            "advisor": {"cache_ttl": "10m", "max_tokens": "lots"},
            "bill_footer": "yes",
        }
    )
    assert s.budget.per_call_usd == 0.50
    assert s.budget.max_messages_per_call == 20
    assert s.budget.pacing == "dynamic"
    assert s.budget.per_month_usd == 100.0  # booleans are not numbers
    assert s.cache_ttl == "5m"
    assert s.max_tokens == 4000
    assert s.bill_footer is True
    assert len(s.warnings) == 7


def test_yaml_off_is_false():
    # A bare `off` in YAML 1.1 is boolean false.
    assert parse_settings({**VALID, "advisor": {"cache_ttl": False}}).cache_ttl == "off"
    assert parse_settings({**VALID, "advisor": {"cache_ttl": "1h"}}).cache_ttl == "1h"


def test_zero_allowed_for_optional_ceilings():
    s = parse_settings(
        {"budget": {"cycle_start_day": 3, "max_per_call_usd": 0, "max_per_day_usd": 0}}
    )
    assert s.budget.max_per_call_usd == 0 and s.budget.max_per_day_usd == 0
    assert s.warnings == ()


def test_unpriced_line_is_unusable():
    s = parse_settings(
        {
            **VALID,
            "default_line": "opus",
            "lines": {
                "opus": {
                    "provider": "anthropic",
                    "model": "claude-opus-5-5",
                    "prices_per_mtok": {"input": 4, "output": 20},
                },
                "sonnet": {},
            },
        }
    )
    assert not s.line("opus").usable
    assert "cache_read" in s.line("opus").problem
    assert s.line("sonnet").usable  # missing fields fall back to the shipped sonnet line
    assert s.usable_lines() == ["sonnet"]


def test_unknown_default_line_warns():
    s = parse_settings({**VALID, "default_line": "gpt"})
    assert s.line(None) is None
    assert any("default_line" in w for w in s.warnings)


def test_cycle_start_day_required():
    # U-37 (settings part): no default, and invalid values are rejected.
    for raw in (None, 0, 32, "3", 3.0, True):
        s = parse_settings({"budget": {"cycle_start_day": raw}})
        assert s.budget.cycle_start_day is None
        assert s.budget.cycle_error == CYCLE_SETUP_MESSAGE
    assert parse_settings(VALID).budget.cycle_error == ""
    assert parse_settings({"budget": {"cycle_start_day": 31}}).budget.cycle_start_day == 31


def test_allowed_senders_are_strings():
    s = parse_settings({**VALID, "allowed_senders": [12345, " abc ", ""]})
    assert s.allowed_senders == ("12345", "abc")


def test_non_mapping_root():
    s = parse_settings(["not", "a", "mapping"])
    assert s.default_line == "sonnet"
    assert s.warnings
