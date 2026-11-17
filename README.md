# hermes-advisor-call

A [Hermes Agent](https://github.com/NousResearch/hermes-agent) plugin that
lets a cheap or local main model **get a second opinion from a stronger
advisor model**, as a multi-message consult with human approval per call and
hard bill caps.

> **Status: v0.2.2.** Tested against Hermes v0.23.1: unit tests, Hermes'
> own plugin checks, an end-to-end run through a real Hermes against a local
> fake Anthropic server, and live checks on a real Hermes with the real API
> through Telegram (approval and deny, session approvals, pre-authorization,
> multi-message calls, the bill caps, the reply limit, and the bill against
> the Claude Console's). See [DESIGN.md](DESIGN.md) section 7.

## How it works

1. Your main model (the **executor**, for example a local Qwen) decides it
   needs help and calls `advisor_call_open` with its question.
2. Hermes shows you its normal approval prompt (Telegram buttons or the
   CLI) with the reason, the call's bill cap and today's spend.
3. Once approved, the executor and the **advisor** (by default Claude
   Sonnet 5.5) exchange messages. The advisor sees a redacted transcript of
   the session plus what the executor sends; it can ask for files, logs or
   errors, and the executor fetches them with its own tools.
4. The call ends when the executor hangs up, you hang up, a cap or the
   message limit is reached, or the call is idle for 30 minutes.
5. Your next reply ends with the bill, for example
   `📞 Advisor call (sonnet): 4 messages · ~$0.12 of $0.50 (estimate) · ended (hung up) · today $0.41 of $3.33`.

The advisor **never gets tools**. It only advises; all execution stays with
the executor. The executor's own model never switches, so its prompt cache
is unaffected.

## Requirements

- Hermes Agent **>= 0.22.0** (tested on 0.23.1).
- `ANTHROPIC_API_KEY` in `~/.hermes/.env`. The plugin never reads or
  stores the key; requests go through Hermes' own provider auth.
- No other dependencies.

## Install

Plugins are per profile: install it only in the profile that should be
able to spend money.

`main` only ever holds released versions; development happens on the `beta`
branch. Installing without `--ref` therefore gets the latest release. To pin
an exact version (recommended for a plugin that spends money), pass the
release's commit id; Hermes requires a full 40-character commit id for
`--ref` and refuses tag names:

```bash
hermes plugins install jmdevita/hermes-advisor-call --ref <full-commit-id> --no-enable
```

Look up a release's commit id from its tag, for example `v0.2.0`:

```bash
git ls-remote https://github.com/jmdevita/hermes-advisor-call refs/tags/v0.2.0
```

```bash
hermes plugins enable advisor-call
```

Then add at least this to `~/.hermes/config.yaml` and restart the gateway:

```yaml
plugins:
  entries:
    advisor-call:
      budget:
        cycle_start_day: 3   # required: the day of the month your API credit refreshes
```

**`cycle_start_day` is required.** The plugin refuses every call until it is
set, because a wrong billing cycle makes the cycle budget wrong for a whole
month. Find the date under the credit's expiry on the Claude Console billing
page. `/second-opinion status` prints the resolved cycle (for example
"Cycle: Nov 3 to Dec 2 (19 days left)") so you can compare.

To upgrade, it depends on how you installed:

- **Installed without `--ref`** (follows `main`, which only holds releases):

  ```bash
  hermes plugins update advisor-call
  ```

- **Installed with `--ref`** (pinned): `update` refuses to move a pin, and a
  reinstall without `--ref` keeps the old one. Move it explicitly:

  ```bash
  hermes plugins install jmdevita/hermes-advisor-call --force --ref <full-commit-id>
  ```

To switch a pinned install to following `main`, remove it with
`hermes plugins remove advisor-call` and install again without `--ref`, then
enable it. **Removing deletes its settings** (`plugins.entries.advisor-call`
and its `enabled` entry in `config.yaml`), so copy them first and add them
back. The spend ledger in `plugin-data/advisor-call/` is kept.

## Configuration

All settings live under `plugins.entries.advisor-call` and are read on every
call, so budget edits apply without a restart. Changing the advisor model
(`advisor.model`) or `advisor.max_tokens` needs a restart. Invalid values fall back to
the default with a warning in the Hermes log.

```yaml
plugins:
  entries:
    advisor-call:
      budget:
        per_month_usd: 100.00       # budget per billing cycle
        cycle_start_day: 3          # REQUIRED, 1-31, no default
        per_call_usd: 0.50          # cap per call
        per_day_usd: auto           # cap per day: auto, or a number for a flat cap
        max_messages_per_call: 20   # executor messages per call, including the first
        idle_timeout_minutes: 30
      approval:
        refuse_when_approvals_off: true
      context:
        opening_transcript_chars: 40000
        tool_output_chars: 2000
        redact_secrets: true
      advisor:
        model: sonnet               # sonnet (default) or opus, or a full model id
        effort: medium              # optional: low | medium | high | xhigh | max; unset = the model's default
        prices_per_mtok: {}         # only for models without shipped prices (see below)
        max_tokens: 4000            # reply limit per message
        timeout_seconds: 180
        cache_ttl: 5m               # 5m | 1h | off
      suggest: true                 # auto-suggest, see below
      allowed_senders: []           # gateway user IDs allowed to spend; [] = anyone the gateway admits
      bill_footer: true
      log_transcripts: false
```

The shipped defaults are the ones above, so a fresh install needs only
`cycle_start_day`. Prices are configuration because they change; check them
against Anthropic's pricing page.

### Budget

Three caps, all hard limits:

- **Per call** (`per_call_usd`): fixed when the call opens and shown on the
  approval prompt. Never more than what is left today.
- **Per day** (`per_day_usd`): with `auto`, the default, what is left of
  the cycle budget divided by the days left, fixed for the day. Unspent
  money rolls forward, so quiet days leave more room later: $100 over 30
  days starts at $3.33 a day. Set a number for a flat daily cap instead.
- **Per cycle** (`per_month_usd`): the cycle runs from `cycle_start_day` to
  the day before the next one (in shorter months, the last day of the
  month).

### Choosing the advisor model

The advisor is Claude Sonnet 5.5 unless you choose otherwise. Switching is
one line, then a restart:

```yaml
      advisor:
        model: opus                 # Claude Opus 5.5
```

The plugin ships prices for these, in USD per million tokens:

| Name | Model | Input | Output | Cache read |
|---|---|---|---|---|
| `sonnet` | `claude-sonnet-5-5` | 2.00 | 10.00 | 0.10 |
| `opus` | `claude-opus-5-5` | 4.00 | 20.00 | 0.20 |

The cache-write price is derived: 1.25 times input with `cache_ttl: 5m`,
2 times with `1h`. Opus costs about twice as much per message, so a call
reaches its dollar cap sooner; raise `budget.per_call_usd` if calls end
early.

There is one advisor model, set here; the executor cannot choose another.

### Choosing the advisor's effort

Effort sets how hard the advisor thinks before it answers. It is optional,
and one line next to the model (restart after changing it):

```yaml
      advisor:
        model: opus
        effort: medium              # low | medium | high | xhigh | max
```

Leave it out and the model uses its own default: `high` on Sonnet 5.5,
`medium` on Opus 5.5. The setting follows `model`, so it still applies after
you switch models. Any other value is ignored with a warning, and the model's
default is used.

Thinking is billed as output and counts against `max_tokens`, so a higher
effort costs more per message and is more likely to fill the reply limit.
With `xhigh` or `max`, consider raising `max_tokens`. `/second-opinion status`
shows the effort next to the model when one is set.

Any other Claude model id works if you give its prices; a model without
prices is refused, never called. You can also override a shipped price this
way, for example if Anthropic changes prices before a plugin update:

```yaml
      advisor:
        model: claude-opus-4-8
        prices_per_mtok:
          claude-opus-4-8: {input: 5.00, output: 25.00, cache_read: 0.50}
```

Each allowed model gets a slot in Hermes' model picker,
`auxiliary.advisor_call_<name>` (`advisor_call_sonnet`, `advisor_call_opus`,
or the model id with `-` as `_`). You do not need to set it. **If you add your
own `extra_body` to that slot, include `max_tokens` in it:** Hermes replaces
the plugin's `extra_body` with yours instead of merging them, and the plugin
uses `extra_body.max_tokens` to limit reply length. A model whose slot has an
`extra_body` without `max_tokens` is refused with a message saying so.

## Approval

Each call goes through Hermes' approval gate, the same one used for
dangerous shell commands:

- **Once:** this call only.
- **Session:** later calls in the same session are approved too.
- **Always:** behaves like Session. The approval rule includes the session
  id, so a new session always asks again. It leaves a harmless unused entry
  in `command_allowlist` in `config.yaml`.
- **Deny**, a timeout, or a gate error: nothing is sent, and the executor is
  told not to retry.

To skip the prompt for a session, type `/second-opinion allow` (optionally
with a model and a cap). It covers one model: the default, or the one you
name. Calls with no human present (cron, unattended
platforms) are blocked by Hermes unless you configured those to
auto-approve.

## Commands

| Command | What it does |
|---|---|
| `/second-opinion` or `/second-opinion status` | The advisor model, the open call, today's spend against the daily cap, this cycle's spend, the cycle window, whether this session is pre-authorized |
| `/second-opinion allow [$cap]` | Pre-authorize calls for this session, optionally with a per-call cap (limited to what is left today). The executor is told on its next turn |
| `/second-opinion revoke` | Remove the pre-authorization |
| `/second-opinion ask <question>` | Ask the advisor yourself: one message with the session transcript. No approval prompt, since you typed it; counts toward the budget |
| `/second-opinion hangup` | End the open call now; the executor is told on its next message |
| `/second-opinion bill` | Today, this cycle, days left, and the last 7 days |
| `/second-opinion transcript` | The last call in this session, from memory |

With a non-empty `allowed_senders`, `allow`, `ask` and `transcript` are
limited to those gateway users, and so are the executor's advisor calls in
their sessions. Local CLI and TUI sessions are treated as the owner.

In Telegram the command appears as `/second_opinion`. If the name is taken by
another plugin, it is registered as `/advisorcall` instead.

## Auto-suggest

Small models often don't notice they are stuck. Hermes already watches for
that: when the executor repeats a failing or identical tool call, Hermes
adds a loop warning to the tool result. The plugin does no detection of its
own and never touches a tool result. It only adds what Hermes doesn't do:

- **You get a line** at the end of the reply when Hermes flagged a loop in
  that turn and the executor did not call the advisor:
  `💡 Hermes flagged a tool loop this turn and the agent didn't call the advisor.`
  with how to ask yourself or allow calls for the session.
- **The executor is told once**, in its standing guidance, that a loop
  warning it cannot resolve is a good moment to call.

It never calls the advisor itself: approval and the caps work exactly as
before, and it costs nothing. You get the line at most once per turn and 3
times per session, and only when a call could actually happen (no call
open, budget left, setup complete; in unattended sessions such as cron,
only after `/second-opinion allow`). `/second-opinion status` shows how
many loop warnings were seen in the last 7 days and how many led to a call.

When a loop is flagged is Hermes' decision, set in Hermes' own
`tool_loop_guardrails` settings; if you turn those warnings off, this goes
quiet too. If Hermes stops the turn outright (its hard stop), there is no
line. Turn the line off with `suggest: false`.

## Billing

With prompt caching off, the bill is exact. **With caching on (the default),
the bill is an estimate**, marked `~` and "(estimate)": Hermes v0.23.1 does
not pass Anthropic's cache token counts to plugins. The plugin estimates the
cached part on the safe side: for ordinary text, above the real cost and
close to what the call would cost without caching. See
[DESIGN.md](DESIGN.md) section 2.1 for how, and set `advisor.cache_ttl: off`
if you want an exact bill.

A 10-message call in English with a 40,000-character opening is estimated
at about $0.40 (real cost about $0.15 with caching, about $0.40 without).

A request that times out is billed at its pre-send estimate, because it may
have been processed and billed anyway.

## Privacy and security

- **What leaves the machine:** your first request and the most recent part of
  the session transcript (once, in the first message) and what the executor writes, sent to the advisor's provider
  (Anthropic). Nothing else is sent anywhere: no telemetry, no third-party
  service.
- **Secrets** are redacted with Hermes' own redactor before sending, even if
  you turned Hermes' redaction off. API keys and credentials in URLs are
  masked. Redaction is pattern-based and can miss unusual formats.
- **Prompt injection:** the transcript may contain web pages or file text.
  The advisor is told it is data, and the advisor has no tools, so the worst
  case is bad advice, which the executor and you still review.
- **Isolation:** strictly per session. One session can never see, speak on
  or end another session's call.
- **At rest:** the ledger (`<hermes home>/plugin-data/advisor-call/ledger.db`)
  holds costs, token counts and call metadata, never message text. Call
  transcripts stay in memory unless `log_transcripts: true`, which writes
  them to `plugin-data/advisor-call/transcripts/`.
- **Spend control:** approval per call, per-call, daily and cycle caps, a
  message limit per call, and `allowed_senders`. The caps hold even when
  approval is bypassed.

## Limits

- **Approval can be bypassed by Hermes itself.** Hermes auto-approves under
  `--yolo`, `/yolo`, `approvals.mode: off` and one-shot `hermes -z` runs. The
  plugin refuses calls when `approvals.mode` is `off` unless you ran
  `/second-opinion allow`, but it cannot detect YOLO or `-z`. **The bill
  caps still apply in every case.**
- **A cap can be overshot by part of one message.** Before each message the
  plugin checks an estimate that assumes a full-length reply and counts
  tokens on the high side by kind of character (letters, digits and
  punctuation, CJK, emoji). Text denser than that can make one message cost
  more than estimated. After any message that reaches the cap the call
  ends, and later checks on the same call scale up to the density measured.
- **Only this plugin's spend is counted.** The budget assumes the whole
  `per_month_usd` is available. Spend from other tools on the same API key
  or Console organization, or from before installing mid-cycle, is
  invisible. Lower `per_month_usd` to allow for it; the credit balance
  running out is the final backstop.
- **The bill is an estimate while caching is on** (see Billing).
- **Calls do not survive a gateway restart.** The next message gets "call
  dropped (gateway restarted); open a new call". The ledger is kept.
- **The executor decides when to call.** Auto-suggest tells you when
  Hermes flags a loop, but never calls on its own.
  A small model may call too rarely or too often.
- Subagents (`delegate_task` children) are not specially supported; the
  normal approval rule applies to them.

## Troubleshooting

| Symptom | Check |
|---|---|
| No approval prompt appears | YOLO mode or `approvals.mode: off` (see Limits). With `off`, use `/second-opinion allow` |
| "Set ... cycle_start_day ..." | Add `budget.cycle_start_day` |
| "advisor unavailable: ..." | `ANTHROPIC_API_KEY` in `~/.hermes/.env`, the credit balance in the Claude Console, `advisor.model` |
| "... not scoped to a workspace ... anthropic-workspace-id header" | The API key was created with Organization scope. Create one with Scope set to a workspace and use that instead (an organization key also has Admin API rights, so delete it from the machine) |
| "advisor model ... cannot be used" | The model has no prices (add them under `advisor.prices_per_mtok`); or its auxiliary slot has an `extra_body` without `max_tokens` |
| The executor never calls | Hermes puts plugin tools behind tool search by default. The executor is told how to find them; if it still fails, set `tools.tool_search.enabled: off` (global) |
| `/second-opinion allow` says "Send a message first" | A new chat has no session until its first message |
| The bill is higher than the Console | Expected while caching is on: the estimate is on the safe side |

## Development

Work happens on the `beta` branch; `main` is fast-forwarded to it for each
release, which is then tagged. To try unreleased changes, install a commit
from `beta` with `--ref`.

No runtime dependencies. Tests run without Hermes installed.

```bash
scripts/ci.sh            # ruff + pytest on Python 3.11 and 3.12 (needs uv)
scripts/hermes-smoke.sh  # load this checkout into a real Hermes and run its plugin checks
scripts/hermes-e2e.sh    # full call through real Hermes against a local fake Anthropic server (no spend)
```

CI is not active yet: the GitHub Actions workflow is kept in
`ci/github-actions.yml`. Move it to `.github/workflows/ci.yml` to turn it
on; it runs the same checks as `scripts/ci.sh`.

The two Hermes scripts expect a Hermes checkout with a virtual environment
next to this repo (`../hermes-agent-ref/.venv`), or the path to its `hermes`
or `python` as the first argument.

## License

MIT
