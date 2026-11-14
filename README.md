# hermes-advisor-call

A [Hermes Agent](https://github.com/NousResearch/hermes-agent) plugin that
lets a cheap or local main model **phone a stronger advisor model** for a
multi-message consult, with human approval per call and hard bill caps.

> **Status: in development, not usable yet.** Phase 0 (verification and
> scaffold) is done; the plugin registers nothing yet. See
> [DESIGN.md](DESIGN.md) for what has been verified against Hermes v0.23.1.

- The executor (your main model) asks to call; you approve once through
  Hermes' own approval prompt.
- The advisor (for example Claude Sonnet 5.5) has **no tools**. It only
  advises; all execution stays with the executor.
- Per-call, daily and monthly spending caps are enforced in plugin code, and
  each turn with advisor activity ends with a one-line bill.

Requires Hermes Agent **>= 0.22.0**.

## Known gaps

- **With prompt caching on (the default), the bill is an estimate.** Hermes
  (v0.23.1) does not pass Anthropic's cache token counts to plugins. The
  plugin estimates the cached part on the safe side (above the real cost,
  close to what the call would cost without caching), and marks the amount
  as an estimate. If Anthropic's cache misses unexpectedly, the estimate can
  run low by about half a cent per message. Set `advisor.cache_ttl: off` for
  an exact bill, at the cost of re-sending the transcript at full price on
  every follow-up.
- **Only this plugin's spend is counted.** The budget assumes the whole
  monthly amount is available to the plugin. Spend from other tools on the
  same credit is invisible; lower `budget.per_month_usd` to allow for it.

## Development

No runtime dependencies. Tests run without Hermes installed.

```bash
scripts/ci.sh            # ruff + pytest on Python 3.11 and 3.12 (needs uv)
scripts/hermes-smoke.sh  # load this checkout into a real Hermes and run its plugin checks
```

## License

MIT
