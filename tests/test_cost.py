"""U-2, U-3, U-31 to U-35, U-38: cost, estimates, cycle, allowance, caps."""

import dataclasses
import datetime as dt

import pytest

from advisor_call.cost import (
    Usage,
    budget_state,
    calibration,
    call_cap,
    charge,
    cycle_for,
    estimate_cost,
    estimate_tokens,
    exact_cost,
    preflight_refusal,
    reusable_prefix,
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
    c = charge(Usage(12_000, 800), SONNET, caching=False, prompt_est=13_000)
    assert c.cost_usd == pytest.approx((12_000 * 2 + 800 * 10) / 1e6)
    assert not c.estimated


def test_charge_uses_reported_cache_counts():
    u = Usage(100, 800, cache_read_tokens=12_000, cache_write_tokens=50)
    c = charge(u, SONNET, caching=True, prompt_est=13_000)
    assert c.cost_usd == pytest.approx(exact_cost(u, SONNET))
    assert not c.estimated


def test_charge_estimate_first_exchange_all_written():
    # Nothing reused and no tail to calibrate on: the cached part is the
    # unscaled estimate, billed as a cache write.
    c = charge(Usage(3, 500), SONNET, caching=True, prompt_est=10_000)
    assert c.cost_usd == pytest.approx((3 * 2 + 10_000 * 2.5 + 500 * 10) / 1e6)
    assert c.estimated


def test_charge_estimate_reused_prefix_at_input_price():
    # Estimated 10,000 tokens: 8,000 reused, 600 uncached tail that Hermes
    # reported as 450 tokens (scale 0.75).
    c = charge(
        Usage(450, 800), SONNET, caching=True, prompt_est=10_000, uncached_est=600, reused_est=8_000
    )
    cached = (10_000 - 600) * 0.75
    reused = 8_000 * 0.75
    expected = ((450 + reused) * 2.0 + (cached - reused) * 2.5 + 800 * 10) / 1e6
    assert c.cost_usd == pytest.approx(expected)
    assert c.estimated


def test_calibration_is_clamped():
    assert calibration(450, 600) == pytest.approx(0.75)
    assert calibration(100, 600) == pytest.approx(0.75)  # never below the floor
    assert calibration(5_000, 600) == pytest.approx(3.0)  # capped
    assert calibration(900, 600) == pytest.approx(1.5)  # denser than estimated: bill more
    assert calibration(40, 50) == 1.0  # tail too short to measure
    assert calibration(0, 5_000) == 1.0


def test_unexpected_miss_undercount_is_bounded():
    # We expected a cache hit on the reused prefix, but the cache missed:
    # really the prefix was written again (2.50), we billed it at 2.00.
    reported = Usage(450, 800)
    billed = charge(
        reported, SONNET, caching=True, prompt_est=20_000, uncached_est=600, reused_est=16_000
    ).cost_usd
    real_cached = (20_000 - 600) * 0.75  # the calibration was right about density
    real = exact_cost(Usage(450, 800, 0, int(real_cached)), SONNET)
    reused_tokens = 16_000 * 0.75
    assert 0 < real - billed <= reused_tokens * (2.5 - 2.0) / 1e6 + 1e-6


def test_no_estimate_when_nothing_cached():
    c = charge(Usage(1_000, 10), SONNET, caching=True, prompt_est=900, uncached_est=900)
    assert not c.estimated


def test_reusable_prefix_ttl():
    assert reusable_prefix(1_000, 100.0, 200.0, "5m") == 1_000
    assert reusable_prefix(1_000, 100.0, 100.0 + 271, "5m") == 0  # inside the margin
    assert reusable_prefix(1_000, 100.0, 100.0 + 3000, "1h") == 1_000
    assert reusable_prefix(1_000, 100.0, 101.0, "off") == 0
    assert reusable_prefix(1_000, None, 101.0, "5m") == 0
    assert reusable_prefix(0, 100.0, 101.0, "5m") == 0


def test_estimate_tokens_by_kind_of_character():
    assert estimate_tokens("abc def") == pytest.approx(7 / 3)  # ASCII letters and spaces
    assert estimate_tokens("1,2") == pytest.approx(3 / 1.5)  # digits and punctuation
    assert estimate_tokens("漢字") == pytest.approx(3.0)  # CJK: 1.5 each
    assert estimate_tokens(chr(0x1F600)) == pytest.approx(3.0)  # an emoji
    assert estimate_tokens("") == 0


def test_dense_text_is_not_underestimated():
    # 30,000 CJK characters are about 30,000 to 40,000 real tokens.
    assert estimate_tokens("漢" * 30_000) >= 40_000
    # Base64-like text: mostly letters with digits and symbols.
    import base64

    b64 = base64.b64encode(bytes(range(256)) * 40).decode()
    assert estimate_tokens(b64) >= len(b64) / 3


def simulate(
    messages=10, *, opening=40_000, q=2_000, r=3_200, system=1_500, out=800, chars_per_token=4.0
):
    """A call on English text (real ~4 chars per token, estimated 3) where
    Hermes reports only the uncached tail (the newest question) as input.
    Returns (billed, real, caching_off_real)."""
    billed = real = off = 0.0
    reused_est = 0.0
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
            prompt_est=prompt / 3,
            uncached_est=tail / 3,
            reused_est=reused_est,
        ).cost_usd
        total = int(prompt / chars_per_token)
        hit = int(reused_est * 3 / chars_per_token)
        real += exact_cost(Usage(tail_tokens, out, hit, total - hit - tail_tokens), SONNET)
        off += exact_cost(Usage(total, out), SONNET)
        reused_est = (prompt - tail) / 3  # cached through the last breakpoint
    return billed, real, off


def test_worked_example_ten_messages():
    """10 messages, a 40,000-character opening: the count lands above the
    real cost, near the caching-off cost, and under the $0.50 default cap."""
    billed, real, off = simulate()
    assert real < billed < 0.50
    assert billed <= off * 1.1
    assert real == pytest.approx(0.15, abs=0.03)
    assert billed == pytest.approx(0.40, abs=0.04)


@pytest.mark.parametrize("caching", [False, True])
@pytest.mark.parametrize("prompt_est,out", [(13_000, 4000), (400, 50), (30_000, 2500)])
@pytest.mark.parametrize("reused_frac", [0.0, 0.8])
def test_estimate_is_at_least_billed(caching, prompt_est, out, reused_frac):
    # U-3: for text the estimator does not underestimate, the pre-flight
    # estimate is at least what the exchange is billed.
    reused = prompt_est * reused_frac
    est = estimate_cost(prompt_est, 4000, SONNET, caching=caching, reused_est=reused)
    worst = Usage(0, out) if caching else Usage(int(prompt_est), out)
    billed = charge(
        worst, SONNET, caching=caching, prompt_est=prompt_est, reused_est=reused
    ).cost_usd
    assert est >= billed - 1e-12


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
    # Installed mid-cycle on Nov 14 with cycle_start_day 3: $100/19, cap ~ $0.79.
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
