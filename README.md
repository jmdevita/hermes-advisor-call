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

## Development

No runtime dependencies. Tests run without Hermes installed.

```bash
scripts/ci.sh            # ruff + pytest on Python 3.11 and 3.12 (needs uv)
scripts/hermes-smoke.sh  # load this checkout into a real Hermes and run its plugin checks
```

## License

MIT
