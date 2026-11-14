# hermes-advisor-call

A [Hermes Agent](https://github.com/NousResearch/hermes-agent) plugin that
lets a cheap or local main model **get a second opinion from a stronger
advisor model**, as a multi-message consult with human approval per call and
hard bill caps.

> **Status: pre-release.** Built and tested against Hermes v0.23.1 (unit
> tests, Hermes' own plugin checks, and an end-to-end run through a real
> Hermes against a local fake Anthropic server). Not yet run against the
> real API. See [DESIGN.md](DESIGN.md) for what has been verified.

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
   message limit is reached, or the line is idle for 30 minutes.
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
able to spend money. Pin `--ref` to a commit.

```bash
hermes plugins install jmdevita/hermes-advisor-call --ref <commit-sha> --no-enable
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
set, because a wrong billing cycle makes the daily pacing wrong for a whole
month. Find the date under the credit's expiry on the Claude Console billing
page. `/second-opinion status` prints the resolved cycle (for example
"Cycle: Nov 3 to Dec 2 (19 days left)") so you can compare.

To upgrade:

```bash
hermes plugins install jmdevita/hermes-advisor-call --force --ref <new-commit-sha>
```

## Configuration

All settings live under `plugins.entries.advisor-call` and are read on every
call, so budget edits apply without a restart. Adding or changing a line, or
changing `advisor.max_tokens`, needs a restart. Invalid values fall back to
the default with a warning in the Hermes log.

```yaml
plugins:
  entries:
    advisor-call:
      default_line: sonnet
      lines:
        sonnet:
          prices_per_mtok:          # USD per million tokens, used for the bill
            input: 2.00
            output: 10.00
            cache_read: 0.10
            cache_write: 2.50       # 5-minute write; use 4.00 with cache_ttl: 1h
      budget:
        per_month_usd: 100.00       # budget per billing cycle
        cycle_start_day: 3          # REQUIRED, 1-31, no default
        pacing: dynamic             # dynamic | fixed
        per_call_usd: 0.50          # base per-call cap; scales with the allowance when dynamic
        max_per_call_usd: 0         # optional ceiling on the dynamic per-call cap; 0 = none
        per_day_usd: 3.00           # used only with pacing: fixed
        max_per_day_usd: 0          # optional ceiling on the dynamic allowance; 0 = none
        max_messages_per_call: 20   # executor messages per call, including the first
        idle_timeout_minutes: 30
      approval:
        refuse_when_approvals_off: true
      context:
        opening_transcript_chars: 40000
        tool_output_chars: 2000
        redact_secrets: true
      advisor:
        max_tokens: 4000            # reply limit per message
        timeout_seconds: 180
        cache_ttl: 5m               # 5m | 1h | off
      allowed_senders: []           # platform user IDs allowed to spend; [] = anyone the gateway admits
      bill_footer: true
      log_transcripts: false
```

The shipped defaults are the ones above, so a fresh install needs only
`cycle_start_day`. Prices are configuration because they change; check them
against Anthropic's pricing page.

### Budget and pacing

- **Cycle:** runs from `cycle_start_day` to the day before the next one (in
  shorter months, the last day of the month).
- **Daily allowance** (`pacing: dynamic`): what is left of the cycle budget
  divided by the days left, fixed for the day. Unspent money rolls forward,
  so the allowance grows late in a cycle. At $100 over 30 days it starts at
  $3.33.
- **Per-call cap** (dynamic): `per_call_usd` scaled by how far today's
  allowance is above the baseline ($0.50 at $3.33/day, $0.75 at $5.00/day),
  never more than what is left today. It is fixed when the call opens and
  shown on the approval prompt.
- **The cycle budget is a hard limit** in both pacing modes.

### Advisor lines

Each line gets its own model slot, `auxiliary.advisor_call_<line>`, shown in
Hermes' model picker. The shipped `sonnet` line uses provider `anthropic`
and model `claude-sonnet-5-5`. To add an Opus line, add it with its prices
and restart:

```yaml
plugins:
  entries:
    advisor-call:
      lines:
        sonnet: {}
        opus:
          prices_per_mtok: {input: 4.00, output: 20.00, cache_read: 0.20, cache_write: 5.00}
auxiliary:
  advisor_call_opus:
    provider: anthropic
    model: claude-opus-5-5
```

A line with missing or invalid prices is refused, never called.

**If you set your own `extra_body` in an `auxiliary.advisor_call_<line>`
block, include `max_tokens` in it.** Hermes replaces the plugin's
`extra_body` with yours instead of merging them, and the plugin uses
`extra_body.max_tokens` to limit reply length. A line whose `extra_body`
has no `max_tokens` is refused with a message saying so.

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
with a line and a cap). Calls with no human present (cron, unattended
platforms) are blocked by Hermes unless you configured those to
auto-approve.

## Commands

| Command | What it does |
|---|---|
| `/second-opinion` or `/second-opinion status` | Line and model, the open call, today's spend against today's allowance, this cycle's spend, the cycle window, whether this session is pre-authorized |
| `/second-opinion allow [line] [$cap]` | Pre-authorize calls for this session, optionally with a per-call cap (limited to what is left today). The executor is told on its next turn |
| `/second-opinion revoke` | Remove the pre-authorization |
| `/second-opinion ask <question>` | Ask the advisor yourself: one message with the session transcript. No approval prompt, since you typed it; counts toward the budget |
| `/second-opinion hangup` | End the open call now; the executor is told on its next message |
| `/second-opinion bill` | Today, this cycle, days left, and the last 7 days |
| `/second-opinion transcript` | The last call in this session, from memory |

In Telegram the command appears as `/second_opinion`. If the name is taken by
another plugin, it is registered as `/advisorcall` instead.

## Billing

With prompt caching off, the bill is exact. **With caching on (the default),
the bill is an estimate**, marked `~` and "(estimate)": Hermes v0.23.1 does
not pass Anthropic's cache token counts to plugins. The plugin estimates the
cached part on the safe side, above the real cost and close to what the call
would cost without caching. See [DESIGN.md](DESIGN.md) section 2.1 for how,
and set `advisor.cache_ttl: off` if you want an exact bill.

A 10-message call with a 40,000-character opening is estimated at about
$0.42 (real cost about $0.16 with caching, about $0.40 without).

## Privacy and security

- **What leaves the machine:** the tail of the session transcript (once, in
  the first message) and what the executor writes, sent to the advisor
  line's provider. Nothing else is sent anywhere: no telemetry, no
  third-party service.
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
  plugin checks an estimate that assumes a full-length reply and about 3
  characters per token. Unusually dense text (for example CJK or minified
  code) can make the real cost of that message higher than the estimate.
  After any message that reaches the cap, the call ends.
- **Only this plugin's spend is counted.** The budget assumes the whole
  `per_month_usd` is available. Spend from other tools on the same API key
  or Console organization, or from before installing mid-cycle, is
  invisible. Lower `per_month_usd` to allow for it; the credit balance
  running out is the final backstop.
- **The bill is an estimate while caching is on** (see Billing).
- **Calls do not survive a gateway restart.** The next message gets "call
  dropped (gateway restarted); open a new call". The ledger is kept.
- **The executor decides when to call.** There is no automatic escalation.
  A small model may call too rarely or too often.
- Subagents (`delegate_task` children) are not specially supported; the
  normal approval rule applies to them.

## Troubleshooting

| Symptom | Check |
|---|---|
| No approval prompt appears | YOLO mode or `approvals.mode: off` (see Limits). With `off`, use `/second-opinion allow` |
| "Set ... cycle_start_day ..." | Add `budget.cycle_start_day` |
| "advisor unavailable: ..." | `ANTHROPIC_API_KEY` in `~/.hermes/.env`, the credit balance in the Claude Console, the model name in `auxiliary.advisor_call_<line>` |
| "advisor line ... is misconfigured" | Missing prices, or an `extra_body` without `max_tokens` in the line's auxiliary block |
| The executor never calls | Hermes puts plugin tools behind tool search by default. The executor is told how to find them; if it still fails, set `tools.tool_search.enabled: off` (global) |
| `/second-opinion allow` says "Send a message first" | A new chat has no session until its first message |
| The bill is higher than the Console | Expected while caching is on: the estimate is on the safe side |

## Development

No runtime dependencies. Tests run without Hermes installed.

```bash
scripts/ci.sh            # ruff + pytest on Python 3.11 and 3.12 (needs uv)
scripts/hermes-smoke.sh  # load this checkout into a real Hermes and run its plugin checks
scripts/hermes-e2e.sh    # full call through real Hermes against a local fake Anthropic server (no spend)
```

The two Hermes scripts expect a Hermes checkout with a virtual environment
next to this repo (`../hermes-agent-ref/.venv`), or the path to its `hermes`
or `python` as the first argument.

## License

MIT
