"""Settings: merge over defaults and validation."""

import pytest

from advisor_call.settings import CYCLE_SETUP_MESSAGE, Prices, aux_task_key, parse_settings

VALID = {"budget": {"cycle_start_day": 3}}


def test_defaults_when_empty():
    s = parse_settings(None)
    sonnet = s.advisor
    assert sonnet.name == "sonnet"
    assert sonnet.usable and sonnet.provider == "anthropic" and sonnet.model == "claude-sonnet-5-5"
    assert sonnet.prices.cache_read == 0.10  # DESIGN.md O-11
    assert s.cache_ttl == "5m" and s.caching
    assert s.budget.per_month_usd == 100.0 and s.budget.per_day_usd == 3.0
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
                "per_month_usd": True,
            },
            "advisor": {"cache_ttl": "10m", "max_tokens": "lots"},
            "bill_footer": "yes",
        }
    )
    assert s.budget.per_call_usd == 0.50
    assert s.budget.max_messages_per_call == 20
    assert s.budget.per_month_usd == 100.0  # booleans are not numbers
    assert s.cache_ttl == "5m"
    assert s.max_tokens == 4000
    assert s.bill_footer is True
    assert len(s.warnings) == 6


def test_yaml_off_is_false():
    # A bare `off` in YAML 1.1 is boolean false.
    assert parse_settings({**VALID, "advisor": {"cache_ttl": False}}).cache_ttl == "off"
    assert parse_settings({**VALID, "advisor": {"cache_ttl": "1h"}}).cache_ttl == "1h"


def test_one_line_switch_to_opus():
    s = parse_settings({**VALID, "advisor": {"model": "opus"}})
    line = s.advisor
    assert (line.name, line.provider, line.model) == ("opus", "anthropic", "claude-opus-5-5")
    assert line.prices == Prices(input=4.0, output=20.0, cache_read=0.20, cache_write=5.0)
    assert s.warnings == ()


def test_full_model_id_and_case_resolve_to_the_alias():
    s = parse_settings({**VALID, "advisor": {"model": "Claude-Opus-5-5"}})
    assert (s.advisor.name, s.advisor.model) == ("opus", "claude-opus-5-5")


def test_removed_settings_warn_and_are_ignored():
    s = parse_settings(
        {
            "budget": {"cycle_start_day": 3, "pacing": "dynamic", "max_per_day_usd": 9},
            "advisor": {"also_allow": ["opus"]},
        }
    )
    assert s.advisor.name == "sonnet" and s.budget.per_day_usd == 3.0
    for key in ("budget.pacing", "budget.max_per_day_usd", "advisor.also_allow"):
        assert any(key in w for w in s.warnings), key


def test_cache_write_price_follows_the_ttl():
    five = parse_settings({**VALID, "advisor": {"cache_ttl": "5m"}}).advisor.prices
    hour = parse_settings({**VALID, "advisor": {"cache_ttl": "1h"}}).advisor.prices
    assert five.cache_write == pytest.approx(2.50) and hour.cache_write == pytest.approx(4.00)


def test_price_override_and_custom_model():
    s = parse_settings(
        {
            **VALID,
            "advisor": {
                "model": "claude-opus-4-8",
                "prices_per_mtok": {
                    "claude-opus-4-8": {"input": 5, "output": 25, "cache_read": 0.5},
                },
            },
        }
    )
    custom = s.advisor
    assert (custom.name, custom.model) == ("claude_opus_4_8", "claude-opus-4-8")
    assert custom.usable and custom.prices.cache_write == pytest.approx(6.25)
    assert custom.task_key == "advisor_call_claude_opus_4_8"  # a valid Hermes task key
    # One shipped price overridden, by alias; the others keep their shipped value.
    over = parse_settings({**VALID, "advisor": {"prices_per_mtok": {"sonnet": {"input": 3.0}}}})
    assert over.advisor.prices.input == 3.0 and over.advisor.prices.output == 10.0


def test_unpriced_model_is_refused_with_a_clear_message():
    s = parse_settings({**VALID, "advisor": {"model": "claude-mystery-9"}})
    line = s.advisor
    assert not line.usable and line.prices is None
    assert "advisor.prices_per_mtok.claude-mystery-9" in line.problem
    bad = parse_settings(
        {
            **VALID,
            "advisor": {
                "model": "x",
                "prices_per_mtok": {"x": {"input": 1, "output": 0, "cache_read": 0}},
            },
        }
    )
    assert "'output'" in bad.advisor.problem


def test_old_line_settings_warn():
    s = parse_settings({**VALID, "lines": {"opus": {}}, "default_line": "opus"})
    assert s.advisor.name == "sonnet"  # ignored
    assert sum("no longer used" in w for w in s.warnings) == 2


def test_invalid_model_value_falls_back():
    s = parse_settings({**VALID, "advisor": {"model": 5}})
    assert s.advisor.name == "sonnet" and any("advisor.model" in w for w in s.warnings)


def test_cycle_start_day_required():
    # No default, and invalid values are rejected.
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
    assert s.advisor.name == "sonnet"
    assert s.warnings


def test_auxiliary_slot_sets_model_and_provider():
    aux = {
        "advisor_call_sonnet": {"provider": "openrouter", "model": "anthropic/x"},
        "other": "junk",
    }
    line = parse_settings(VALID, aux).advisor
    assert (line.provider, line.model) == ("openrouter", "anthropic/x")
    plain = parse_settings(VALID, {"advisor_call_sonnet": {"timeout": 9}}).advisor
    assert (plain.provider, plain.model) == ("anthropic", "claude-sonnet-5-5")
    assert parse_settings(VALID, "not a dict").advisor.model == "claude-sonnet-5-5"


def test_user_extra_body_without_max_tokens_refuses_line():
    # Hermes replaces the plugin's extra_body with the user's (DESIGN.md 2.2),
    # which would drop the reply limit.
    aux = {"advisor_call_sonnet": {"extra_body": {"metadata": {"user_id": "x"}}}}
    line = parse_settings(VALID, aux).advisor
    assert not line.usable and "add max_tokens: 4000" in line.problem
    aux["advisor_call_sonnet"]["extra_body"]["max_tokens"] = 1500
    line = parse_settings(VALID, aux).advisor
    assert line.usable and line.max_tokens == 1500
    assert parse_settings(VALID, {"advisor_call_sonnet": {"extra_body": {}}}).advisor.usable
    assert parse_settings({**VALID, "advisor": {"max_tokens": 900}}).advisor.max_tokens == 900


def test_low_max_tokens_warns():
    s = parse_settings({**VALID, "advisor": {"max_tokens": 300}})
    assert s.max_tokens == 300 and any("replies may come back empty" in w for w in s.warnings)
    assert not parse_settings({**VALID, "advisor": {"max_tokens": 1024}}).warnings
