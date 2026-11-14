"""U-2, U-3, U-31 to U-35, U-38: cost, estimates, cycle, allowance, caps."""

import dataclasses
import datetime as dt

import pytest

from advisor_call.cost import (
    Usage,
    budget_state,
    call_cap,
    charge,
    cycle_for,
    estimate_cost,
    exact_cost,
    preflight_refusal,
    reusable_prefix_chars,
    tokens_per_char,
)
from advisor_call.settings import Prices, parse_settings

SONNET = Prices(input=2.0, output=10.0, cache_read=0.10, cache_write=2.50)
D = dt.date


def budget(**kw):
    return parse_settings({"budget": {"cycle_start_day": 1, **kw}}).budget


# ------------------------------------------------------------------- U-2 / U-3


def test_exact_cost_hand_computed():
    u = Usage(
        input_tokens=1_000, output_tokens=500, cache_read_tokens=10_000, cache_write_tokens=2_000
    )
    # 1000*2 + 10000*0.10 + 2000*2.50 + 500*10 = 2000 + 1000 + 5000 + 5000 = 13000 / 1e6
    assert exact_cost(u, SONNET) == pytest.approx(0.013)


def test_charge_exact_when_caching_off():
    c = charge(Usage(12_000, 800), SONNET, caching=False, prompt_chars=40_000)
    assert c.cost_usd == pytest.approx((12_000 * 2 + 800 * 10) / 1e6)
    assert not c.estimated


def test_charge_uses_reported_cache_counts():
    u = Usage(100, 800, cache_read_tokens=12_000, cache_write_tokens=50)
    c = charge(u, SONNET, caching=True, prompt_chars=40_000)
    assert c.cost_usd == pytest.approx(exact_cost(u, SONNET))
    assert not c.estimated


def test_charge_estimate_first_exchange_all_written():
    # Nothing reused and no tail to measure: the cached part is sized at the
    # high default ratio (chars/3) and billed as a cache write.
    c = charge(Usage(100, 800), SONNET, caching=True, prompt_chars=30_000)
    written = 30_000 // 3 + 1
    assert c.cost_usd == pytest.approx((100 * 2 + written * 2.5 + 800 * 10) / 1e6)
    assert c.estimated


def test_charge_estimate_reused_prefix_at_input_price():
    # 30,000 chars: 24,000 reused, 4,000 newly cached, 2,000 uncached tail
    # that Hermes reported as 500 tokens (4 chars per token).
    c = charge(
        Usage(500, 800),
        SONNET,
        caching=True,
        prompt_chars=30_000,
        uncached_chars=2_000,
        reused_chars=24_000,
    )
    reused = 24_000 // 4
    written = 28_000 // 4 + 1 - reused
    expected = ((500 + reused) * 2.0 + written * 2.5 + 800 * 10) / 1e6
    assert c.cost_usd == pytest.approx(expected)
    assert c.estimated


def test_token_ratio_calibration_is_clamped():
    assert tokens_per_char(500, 2_000) == pytest.approx(0.25)
    assert tokens_per_char(100, 2_000) == pytest.approx(0.25)  # never sparser than chars/4
    assert tokens_per_char(1_000, 1_000) == pytest.approx(0.5)  # capped
    assert tokens_per_char(700, 2_000) == pytest.approx(0.35)
    assert tokens_per_char(50, 100) == pytest.approx(1 / 3)  # tail too short to measure
    assert tokens_per_char(0, 5_000) == pytest.approx(1 / 3)


def test_unexpected_miss_undercount_is_bounded():
    # We expected a cache hit on the reused prefix, but the cache missed:
    # really the prefix was written again (2.50), we billed it at 2.00.
    chars, uncached, reused_chars = 60_000, 2_000, 50_000
    reported = Usage(500, 800)  # tail measured at 4 chars per token
    billed = charge(
        reported,
        SONNET,
        caching=True,
        prompt_chars=chars,
        uncached_chars=uncached,
        reused_chars=reused_chars,
    ).cost_usd
    real = exact_cost(Usage(500, 800, 0, (chars - uncached) // 4), SONNET)
    reused_tokens = reused_chars // 4
    assert 0 < real - billed <= reused_tokens * (2.5 - 2.0) / 1e6 + 1e-9


def test_no_estimate_when_nothing_cached():
    c = charge(Usage(1_000, 10), SONNET, caching=True, prompt_chars=3_000, uncached_chars=3_000)
    assert not c.estimated


def test_reusable_prefix_ttl():
    assert reusable_prefix_chars(1_000, 100.0, 200.0, "5m") == 1_000
    assert reusable_prefix_chars(1_000, 100.0, 100.0 + 271, "5m") == 0  # inside the margin
    assert reusable_prefix_chars(1_000, 100.0, 100.0 + 3000, "1h") == 1_000
    assert reusable_prefix_chars(1_000, 100.0, 101.0, "off") == 0
    assert reusable_prefix_chars(1_000, None, 101.0, "5m") == 0
    assert reusable_prefix_chars(0, 100.0, 101.0, "5m") == 0


def simulate(
    messages=10, *, opening=40_000, q=2_000, r=3_200, system=1_500, out=800, chars_per_token=4.0
):
    """A call where Hermes reports only the uncached tail (the newest
    question) as input. Returns (billed, real, caching_off_real)."""
    billed = real = off = 0.0
    reused = 0
    prompt = system + opening
    for n in range(messages):
        if n:
            prompt += r + q  # the previous reply and the new question
        tail = q if n else 0  # after the last breakpoint
        tail_tokens = int(tail / chars_per_token)
        billed += charge(
            Usage(tail_tokens, out),
            SONNET,
            caching=True,
            prompt_chars=prompt,
            uncached_chars=tail,
            reused_chars=reused,
        ).cost_usd
        total = int(prompt / chars_per_token)
        hit = int(reused / chars_per_token)
        real += exact_cost(Usage(tail_tokens, out, hit, total - hit - tail_tokens), SONNET)
        off += exact_cost(Usage(total, out), SONNET)
        reused = prompt - tail  # cached through the last breakpoint
    return billed, real, off


def test_worked_example_ten_messages():
    """The 10-message example from the design discussion: the count lands
    above the real cost, near the caching-off cost, and under the $0.50
    default cap."""
    billed, real, off = simulate()
    assert real < billed < 0.50
    assert billed <= off * 1.1  # near what the call would cost without caching
    assert real == pytest.approx(0.15, abs=0.03)
    assert billed == pytest.approx(0.42, abs=0.03)


@pytest.mark.parametrize("caching", [False, True])
@pytest.mark.parametrize("chars,out", [(40_000, 4000), (1_200, 50), (90_000, 2500)])
@pytest.mark.parametrize("reused_frac", [0.0, 0.8])
def test_estimate_is_at_least_actual(caching, chars, out, reused_frac):
    # Real token counts for English text are ~chars/4; the estimate assumes /3.
    reused_chars = int(chars * reused_frac)
    est = estimate_cost(chars, 4000, SONNET, caching=caching, reused_chars=reused_chars)
    real_tokens = chars // 4
    # Worst case for what we would bill: everything unreported, full reply.
    worst = Usage(0, out) if caching else Usage(real_tokens, out)
    billed = charge(
        worst, SONNET, caching=caching, prompt_chars=chars, reused_chars=reused_chars
    ).cost_usd
    assert est >= billed


# ------------------------------------------------------------------- cycle (U-33, U-38)


def test_cycle_start_3_on_nov_14():
    c = cycle_for(D(2026, 11, 14), 3)
    assert (c.start, c.end) == (D(2026, 11, 3), D(2026, 12, 2))
    assert c.days_left(D(2026, 11, 14)) == 19
    assert c.describe(D(2026, 11, 14)) == "Cycle: Nov 3 to Dec 2 (19 days left)"


def test_cycle_last_day_and_rollover():
    assert cycle_for(D(2026, 12, 2), 3).days_left(D(2026, 12, 2)) == 1
    new = cycle_for(D(2026, 12, 3), 3)
    assert (new.start, new.end) == (D(2026, 12, 3), D(2027, 1, 2))


def test_cycle_before_start_day_uses_previous_month():
    c = cycle_for(D(2026, 11, 1), 3)
    assert (c.start, c.end) == (D(2026, 10, 3), D(2026, 11, 2))


def test_cycle_start_31_in_short_months():
    nov = cycle_for(D(2026, 11, 30), 31)
    assert (nov.start, nov.end) == (D(2026, 11, 30), D(2026, 12, 30))
    jan = cycle_for(D(2027, 2, 10), 31)
    assert (jan.start, jan.end) == (D(2027, 1, 31), D(2027, 2, 27))
    feb = cycle_for(D(2027, 2, 28), 31)
    assert (feb.start, feb.end) == (D(2027, 2, 28), D(2027, 3, 30))
    leap = cycle_for(D(2028, 2, 29), 31)
    assert leap.start == D(2028, 2, 29)


def test_cycle_across_year_end():
    c = cycle_for(D(2027, 1, 1), 15)
    assert (c.start, c.end) == (D(2026, 12, 15), D(2027, 1, 14))


# ------------------------------------------------------------------- allowance (U-31, U-32, U-34)

# A 30-day cycle: start day 1 in November (Nov 1 to Nov 30).
NOV1 = D(2026, 11, 1)


def test_dynamic_allowance_day_one_and_rollforward():
    b = budget()
    assert budget_state(b, NOV1, spent_today=0, spent_before_today=0).allowance == pytest.approx(
        100 / 30
    )
    # Nothing spent for 10 days: day 11 allows 100 / 20.
    assert budget_state(
        b, D(2026, 11, 11), spent_today=0, spent_before_today=0
    ).allowance == pytest.approx(5.00)
    # $40 left with 4 days to go: $10/day.
    assert budget_state(
        b, D(2026, 11, 27), spent_today=0, spent_before_today=60
    ).allowance == pytest.approx(10.00)
    # Last day: everything left.
    assert budget_state(
        b, D(2026, 11, 30), spent_today=0, spent_before_today=73
    ).allowance == pytest.approx(27.00)


def test_allowance_fixed_for_the_day_and_clamped():
    b = budget()
    a1 = budget_state(b, NOV1, spent_today=0, spent_before_today=0).allowance
    a2 = budget_state(b, NOV1, spent_today=3.0, spent_before_today=0).allowance
    assert a1 == a2  # spend today does not change today's allowance
    capped = budget(max_per_day_usd=2.0)
    assert budget_state(capped, NOV1, spent_today=0, spent_before_today=0).allowance == 2.0
    over = budget_state(b, D(2026, 11, 20), spent_today=0, spent_before_today=120)
    assert over.allowance == 0.0


def test_fixed_pacing_and_cycle_hard_limit():
    b = budget(pacing="fixed", per_day_usd=5.0)
    s = budget_state(b, D(2026, 11, 29), spent_today=0, spent_before_today=98.0)
    assert s.allowance == 5.0
    assert s.available == pytest.approx(2.0)  # cycle budget is the hard limit
    reason = preflight_refusal(
        call_cost=0, call_cap_usd=10, exchanges=0, max_messages=20, est_cost=2.5, state=s
    )
    assert reason == "month_cap"
    dyn = budget_state(budget(), D(2026, 11, 29), spent_today=1.0, spent_before_today=98.0)
    assert preflight_refusal(
        call_cost=0, call_cap_usd=10, exchanges=0, max_messages=20, est_cost=1.5, state=dyn
    ) in ("day_cap", "month_cap")


# ------------------------------------------------------------------- call cap (U-35)


def _state_with_allowance(b, allowance_factor, spent_today=0.0):
    """A state on Nov 1 whose allowance is factor x the baseline."""
    s = budget_state(b, NOV1, spent_today=spent_today, spent_before_today=0)
    target = s.baseline * allowance_factor
    return dataclasses.replace(s, allowance=target)


@pytest.mark.parametrize("factor,expected", [(1.0, 0.50), (1.5, 0.75), (3.0, 1.50)])
def test_dynamic_call_cap_scales(factor, expected):
    b = budget()
    assert call_cap(b, _state_with_allowance(b, factor)) == pytest.approx(expected)


def test_call_cap_clamps():
    b = budget(max_per_call_usd=1.0)
    assert call_cap(b, _state_with_allowance(b, 3.0)) == pytest.approx(1.0)
    # Never more than what is left today.
    b2 = budget()
    s = _state_with_allowance(b2, 3.0, spent_today=9.8)  # allowance 10.0, 0.2 left
    assert call_cap(b2, s) == pytest.approx(0.2)
    # Fixed pacing keeps the base cap.
    bf = budget(pacing="fixed", per_day_usd=3.0)
    assert call_cap(bf, budget_state(bf, NOV1, spent_today=0, spent_before_today=0)) == 0.5


def test_custom_cap_replaces_but_is_clamped():
    b = budget()
    s = budget_state(b, NOV1, spent_today=3.0, spent_before_today=0)  # ~0.33 left today
    assert call_cap(b, s, custom_cap=2.0) == pytest.approx(100 / 30 - 3.0)
    assert call_cap(b, s, custom_cap=0.1) == pytest.approx(0.1)


def test_mid_cycle_install_example():
    # Plan 7.6: installed Nov 14, cycle_start_day 3 -> $100/19, cap ~ $0.79.
    b = parse_settings({"budget": {"cycle_start_day": 3}}).budget
    s = budget_state(b, D(2026, 11, 14), spent_today=0, spent_before_today=0)
    assert s.allowance == pytest.approx(100 / 19)
    assert call_cap(b, s) == pytest.approx(0.5 * (100 / 19) / (100 / 30), abs=1e-9)
    assert round(call_cap(b, s), 2) == 0.79


def test_preflight_order_and_reserved_elsewhere():
    s = budget_state(budget(), NOV1, spent_today=0, spent_before_today=0)
    kw = dict(call_cost=0.0, call_cap_usd=0.5, exchanges=0, max_messages=20, state=s)
    assert preflight_refusal(**{**kw, "exchanges": 20}, est_cost=0.01) == "turn_limit"
    assert preflight_refusal(**kw, est_cost=0.6) == "bill_cap"
    assert preflight_refusal(**kw, est_cost=0.1) == ""
    assert preflight_refusal(**kw, est_cost=0.1, reserved_elsewhere=3.3) == "day_cap"


def test_budget_state_requires_cycle_day():
    b = parse_settings({}).budget
    with pytest.raises(ValueError):
        budget_state(b, NOV1, spent_today=0, spent_before_today=0)


def test_usage_from_result_tolerates_junk():
    class R:
        input_tokens = "12"
        output_tokens = None
        cache_read_tokens = -5

    u = Usage.from_result(R())
    assert (u.input_tokens, u.output_tokens, u.cache_read_tokens, u.cache_write_tokens) == (
        12,
        0,
        0,
        0,
    )
