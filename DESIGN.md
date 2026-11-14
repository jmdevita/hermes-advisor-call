# advisor-call: design notes

This file records what was verified in the Hermes source and how it changes
the build plan. It grows into the trimmed design document by release.

**Reference:** Hermes Agent `v0.23.1` (commit `3b421578`). Line numbers drift
between releases; search by name. "Verified" means read in the source and,
where marked, exercised with a local script against the real functions (no
network). "Inferred" means read but not exercised.

---

## 1. Open items (plan section 12)

### O-1 Do caller-set `cache_control` markers reach Anthropic?

**Yes, unchanged, including `ttl`.** Exercised: a scripted call with a fake
Anthropic client showed both user-part markers and a system marker arriving
as `{"type": "ephemeral", "ttl": "1h"}`, with no markers added by Hermes.

- `agent/plugin_llm.py` passes `messages` through untouched (`_host_kwargs`).
- `agent/anthropic_message_convert.py` copies a part's `cache_control` dict
  onto the converted block. A system message keeps its block list only when
  a part carries `cache_control`; otherwise its text is flattened.
- Markers can move: blank text blocks are dropped and their marker moves to
  the last surviving block; consecutive same-role messages are merged.
  Markers on thinking blocks are removed.
- Hermes does not enforce Anthropic's 4-breakpoint limit on this path. We
  use at most 2 (plan 7.5), so this is not a concern.
- **Caveat:** if the primary call fails and Hermes' recovery ladder moves to
  another route, `_replan_synchronous_cache_sections` may strip or re-plan
  markers. See section 2.4.
- A top-level automatic `cache_control` can also be sent through the slot's
  `extra_body` (exercised). Not needed.

**But see O-2: caching works, yet the plugin cannot see what it cost.**

### O-2 Does `input_tokens` exclude cache tokens?

**Yes, but Hermes drops the cache counts entirely on the native Anthropic
route.** `cache_read_tokens` and `cache_write_tokens` are always 0.

- `agent/auxiliary_client.py`, Anthropic adapter (`usage = SimpleNamespace(...)`)
  rebuilds usage with only `prompt_tokens`, `completion_tokens`,
  `total_tokens`. Anthropic's `cache_read_input_tokens` and
  `cache_creation_input_tokens` are discarded there.
- `agent/plugin_llm.py::_extract_usage` does map those fields, but they are
  already gone. Exercised: Anthropic usage of 10 input, 5 output, 900 cache
  read, 300 cache write came back as `input_tokens=10, cache_read_tokens=0,
  cache_write_tokens=0`.
- `input_tokens` is Anthropic's raw value, which by API definition excludes
  cached tokens (API behaviour, not shown in Hermes code).
- `usage.cost_usd` is never set on this path, and `result.model` echoes the
  requested model rather than the one the API reports.

**Consequence:** with caching on, the plugin would under-bill every exchange.
Cache writes (1.25x input price) and reads (0.05x on Sonnet 5.5) would be
invisible, so the budget caps would not hold. Decision in section 2.1.

### O-3 Can reasoning effort be set for an auxiliary task?

**Yes, through config.** Set `auxiliary.<key>.reasoning_effort` (or
`extra_body.reasoning`). For Sonnet 5.5 the adapter maps an effort to
`thinking: {type: adaptive, display: summarized}` plus
`output_config.effort`, and a disable to `thinking: {type: between_tools}`.
With nothing set, Hermes sends no `thinking` field and the model runs its
default adaptive thinking (default effort `high` on Sonnet 5.5).

### O-3b (new) `max_tokens` is not forwarded on the native Anthropic route

**Verified, likely a Hermes bug.** `_forwards_max_tokens` is false for the
native `anthropic` provider, so `ctx.llm.complete(max_tokens=4000)` is
silently replaced by the model's output ceiling (128,000 for Sonnet 5.5).
One reply could then cost up to about $1.28 in output alone, more than a
whole default per-call cap.

**Workaround, verified at the SDK level:** the slot's `extra_body` is merged
into the request body, and `extra_body: {max_tokens: 4000}` overrides the
body's `max_tokens` (exercised with `anthropic==0.125.0` and a mock
transport: the sent body had `max_tokens: 4000`). Decision in section 2.2.

### O-4 Which kwargs does `llm_request` middleware receive?

**`session_id` is included.** Call site `agent/turn_api_request.py`:
`request` (provider kwargs, a copy), `original_request`, `task_id`,
`turn_id`, `api_request_id`, `session_id`, `platform`, `model`, `provider`,
`base_url`, `api_mode`, `api_call_count`, plus schema-version strings.

- Return `None` to leave the request unchanged (`{"request": {...}}` would
  replace it; we never do). Exceptions are logged and skipped.
- It runs for main-loop provider calls and `ctx.side_question` only. **Our
  own `ctx.llm.complete` does not pass through it**, so there is no
  recursion.
- Middleware is disabled in `safe_worker` mode; capture then falls back to
  the `pre_llm_call` snapshot.

### O-5 Signature of `register_system_prompt_section`

```python
ctx.register_system_prompt_section(
    id: str,                                  # ^[a-z0-9][a-z0-9._-]{0,127}$, unique
    content: str | Callable[[Mapping], str],
    *, position: str = "after_memory",        # the only valid value
    max_chars: int = 4000,                    # 1..4000
)
```

Sections are rendered once per session and frozen, so they never disturb
the executor's prompt cache. We pass a static string with id `advisor-call`.

### O-6 How Hermes loads a directory plugin

**Keep the sub-package.** The plugin directory is imported as a real
package, `hermes_plugins.advisor_call` (hyphen becomes underscore), with
`__path__` set to the plugin directory. `sys.path` is not changed.

- Use **relative imports only** (`from .advisor_call import hooks`). An
  absolute `import advisor_call` would not resolve.
- Exercised with a scratch package using the same loader calls:
  `hermes_plugins.advisor_call.advisor_call.core` resolved.

### O-7 Is `/phone` free?

**Yes.** Neither `phone` nor `advisorcall` appears among the 131 built-in
command names and aliases, and no bundled plugin uses them.

- A built-in collision would be skipped with a warning; a collision with
  another plugin is a silent last-wins overwrite.
- Telegram: names are limited to `[a-z0-9_]{1,32}`. **A plugin command whose
  `args_hint` starts with `<` is left out of the Telegram command menu**, so
  `/phone` uses `args_hint="[status|allow|revoke|ask|hangup|bill|transcript]"`.

### O-8 Does the command `source` carry the sender?

**On the gateway, yes:** `source` is a `SessionSource` and the sender is
`source.user_id`. On CLI and TUI, `source` is `None`.

- Handler kwargs on every surface: `source`, `session_id`, `session_key`,
  `platform`. The first positional argument is `raw_args`.
- A truthy return value becomes the reply text.
- **Use a sync handler.** Async command results are bounded at 30 s on the
  CLI, and `/phone ask` may wait up to the advisor timeout. Sync handlers run
  in an executor on the gateway.
- Rule for `allowed_senders`: gateway uses `source.user_id`; CLI has no
  sender and is treated as the local owner, so the list does not apply. If
  the list is non-empty and a gateway command carries no `user_id`, refuse.

### O-9 Manifest keys and capabilities

Accepted manifest fields (`hermes_cli/plugins_manifest.py`,
`_KNOWN_MANIFEST_FIELDS`) include `name`, `version`, `description`,
`author`, `license`, `homepage`, `requires_hermes`, `provides_tools`,
`provides_hooks`, `capabilities`, `config_keys`, `config_schema`, `tags`,
`manifest_version`.

- There are **no keys** for commands, auxiliary tasks, system prompt
  sections or LLM access, and nothing gates them at runtime.
- `provides_middleware` is not a known field (the loader logs "unknown
  manifest field ignored"), but `hermes plugins validate` checks it against
  what is registered. **Declare it anyway** and expect the log line.
- `hermes plugins validate` fails on registered-but-undeclared tools, hooks
  or middleware. **`hermes plugins doctor` errors on any hook callback
  without `**kwargs`**, so every hook takes `**kwargs`.
- `requires_hermes: ">=0.22.0"`: if unmet, the plugin is skipped before
  import.
- **No capability is needed.** `ctx.llm.complete(task=...)` with a task this
  plugin registered is allowed without any grant (`plugin_llm.py`, task
  ownership check). Capabilities are consent metadata; only provider, model
  or built-in task overrides need them, and we use none. So `capabilities`
  is omitted.
- **Do not touch `ctx.llm` inside `register()`.** It is a property, and the
  `validate` probe's stub context exposes only callable members. Access it
  lazily in handlers.
- Public API check (`hermes_cli/data/plugin_api.json`): `plugins.plugin_storage`
  and `hermes_cli.config` are whole-module entries;
  `agent.redact:redact_sensitive_text` is listed by name.
  - `plugin_db(name, filename="data.db") -> sqlite3.Connection`, WAL mode,
    `check_same_thread=False`, under `<home>/plugin-data/<name>/`.
  - `redact_sensitive_text(text, *, force=False, ...) -> str`.
  - `load_config_readonly() -> dict`. **Never mutate the result**; it is a
    shared cache.

### O-10 Is the tool visible to the executor?

**Not directly. Plugin tools sit behind tool search by default.** With
`tools.tool_search.enabled: auto` (the default), plugin tools are replaced
in the model's tool list by the bridge tools `tool_search`, `tool_describe`
and `tool_call`. The only way to list them directly is
`tools.tool_search.enabled: off`, which is global.

- **The approval hook still sees the real name.** The executor unwraps a
  `tool_call` bridge call before running `pre_tool_call`
  (`agent/tool_executor.py::_unwrap_tool_search_call`), so the hook sees
  `advisor_call_open`, never `tool_call`. Verified in the source; this
  matters because a hook keyed on the wrong name would skip approval.
- The `advisor_call` toolset is on by default on every platform, unless the
  user later saves a platform tool list that leaves it out.
- Plan change in section 2.3.

### O-11 Sonnet 5.5 prices and cache minimum

From Anthropic's published pricing (via the Claude API reference, cached
2026-11-03; confirm on the pricing page before release):

| Item | USD per MTok |
|---|---|
| Input | 2.00 |
| Output | 10.00 |
| Cache read | **0.10** (the plan said 0.20) |
| Cache write, 5 minutes | 2.50 (1.25x input) |
| Cache write, 1 hour | 4.00 (2x input) |

Minimum cacheable prefix on Sonnet 5.5: **512 tokens**.

---

## 2. Changes to the plan

### 2.1 Caching and billing (from O-2): **decided, known gap, billed by estimate**

**Decision (2026-11-14):** accepted as a known gap for v1. No Admin API key
is used to read real spend. Caching stays on by default
(`advisor.cache_ttl: 5m`), and the cached part of each exchange is billed
**by estimate**, labelled as such wherever a bill is shown.

How the estimate works (`cost.charge`):

1. Hermes reports only the uncached tail of the prompt (after the last
   cache breakpoint, normally the executor's newest message) as
   `input_tokens`. That part is exact.
2. The cached part is sized in tokens with a ratio **measured on that
   tail** (`input_tokens / tail characters`), clamped between 1/4 and 1/2
   tokens per character. When the tail is under 400 characters (the
   opening exchange) the ratio is the deliberately high 1/3.
3. The part of the prompt the previous successful exchange left in the
   cache, if that was less than the TTL minus 30 seconds ago, is billed at
   the **normal input price** (2.00), more than a cache read (0.10). The
   rest is billed at the cache-write price (2.50).

Bounds:

- The count never exceeds the caching-off cost by more than the write
  premium on new text, and is always above the real cost when the cache
  behaves.
- An unexpected cache miss (Anthropic evicts early, or a prompt is under
  the 512-token minimum) under-bills only the reused prefix, by
  2.50 - 2.00 = 0.50 per MTok: about half a cent per missed message for a
  10K-token transcript.
- Worked example (test `test_worked_example_ten_messages`): 10 messages, a
  40,000-character opening, ~2,000-character questions, ~3,200-character
  replies, English text. Plugin count ~$0.42; real cost ~$0.16; caching off
  ~$0.40. Fits the $0.50 default cap.

Where it is labelled: each exchange row has `estimated = 1` in the ledger;
the call carries `estimated`; the tool result's `bill` gets
`"estimate": true`; the footer and `/phone bill` mark the amount as an
estimate. The plugin logs a one-time warning that bills are estimated while
Hermes hides cache counts.

`cache_ttl: off` gives an exact bill, at full input price for the whole
transcript on every follow-up. If a later Hermes reports cache counts
(nonzero `cache_read_tokens` or `cache_write_tokens`), those are used and
the bill is exact.

Options that were considered: (A) caching off by default, exact; (B)
caching on, every hidden token billed as a cache write (safe but about 4x
the real cost, so calls hit their cap early); (C) fix upstream in Hermes.
Chosen: the estimate above, a middle ground between A and B. C is optional.
Reading real spend through the Admin API was ruled out: it needs an
organization admin key on the Hermes host, and its data lags 30 minutes to
24 hours.

### 2.2 `max_tokens` (from O-3b)

- Each line's auxiliary task is registered with
  `defaults={"provider": ..., "model": ..., "timeout": ...,
  "extra_body": {"max_tokens": <limit>}}`. `max_tokens=` is also passed to
  `complete()` for routes that honour it.
- **Verified end to end** (`scripts/hermes-e2e.sh`): the request Hermes
  sends carries `max_tokens: 4000`.
- **A user `extra_body` replaces the plugin's, it does not merge.** Hermes
  merges provider, model and timeout defaults under the user's
  `auxiliary.advisor_call_<line>` block, but an `extra_body` there replaces
  the registered one whole (verified). Without a `max_tokens` in it, replies
  would be unbounded (up to 128K tokens, about $1.28). So:
  - a user `extra_body` without `max_tokens` makes the line unusable, with
    the message "auxiliary.advisor_call_<line>.extra_body replaces the
    plugin's reply limit; add max_tokens: 4000 to it";
  - a user `max_tokens` there becomes the line's reply limit, used for the
    pre-flight estimate too.
- Because `extra_body` is fixed at registration, a change to
  `advisor.max_tokens` needs a restart.
- Sonnet 5.5 thinks by default and thinking counts against `max_tokens`.
  The empty-text guard (plan 7.5) stays.

### 2.3 Tool visibility (from O-10)

- The system prompt section tells the executor that the advisor tools are
  reachable through tool search, by name: "call `tool_search` for
  `advisor_call` if the tools are not listed".
- README documents `tools.tool_search.enabled: off` as the fallback if the
  executor fails to find them. Live check L-3 records which way it worked.

### 2.4 Provider fallback (from O-1 and the error path)

`call_llm` retries the same provider on transient errors (up to 2 extra
attempts, `auxiliary.transient_retries`), then may walk a recovery ladder
to other routes. A reply from another provider would be priced as Sonnet
and is not the advisor the user approved.

- After each exchange the plugin checks `result.provider` against the
  line's configured provider. A mismatch is treated as "advisor
  unavailable (Hermes fell back to <provider>)": the reply is not used, the
  exchange is recorded with `ok=0`, and its tokens are not billed at
  Anthropic prices.
- `PluginLlmTrustError`, the SDK's own errors (for example
  `anthropic.APIStatusError`), `RuntimeError` and
  `InvalidProviderResponseError` can all surface. The handler catches
  `Exception` and reports "advisor unavailable: <type>: <message>".

### 2.5 Approval (plan 3.1), confirmed with three additions

Confirmed: `approve` routes to the human gate keyed `plugin_rule:<rule_key>`;
deny, timeout and gate errors fail closed; `block` beats `approve`; plugin
rules skip the smart approver; no human present blocks.

- **`approvals.mode` defaults to `smart`.** Values are `manual`, `smart`,
  `off`. Plugin rules still reach the human under `smart`.
- **More bypasses than the plan listed.** The gate auto-approves under
  CLI `--yolo`, gateway `/yolo`, `approvals.mode: off`, and one-shot
  `hermes -z` turns. Cron, `-q` and unattended platforms auto-approve only
  if their `*_mode` is set to `approve` (default `deny`). The YOLO and `-z`
  flags are not public plugin API, so the plugin cannot detect them. Bill
  caps still apply; the README must say so.
- **An "Always" tap is written to `command_allowlist` in config.yaml.**
  Because the `rule_key` contains the session id, it never matches another
  session, but the entry stays in config.yaml.
- The hook times out at `plugins.hook_callback_timeout` (default 30 s) and
  fails closed. The hook only returns `approve`; the gate itself does the
  waiting, so the human's response time does not count against this.

### 2.6 Hook and tool details

- `pre_tool_call` kwargs: `tool_name`, `args`, `task_id`, `session_id`,
  `tool_call_id`, `turn_id`, `api_request_id`, `middleware_trace`,
  `conversation_id`. No sender: the sender comes from the `pre_llm_call`
  snapshot for the same session.
- `pre_llm_call` kwargs include `sender_id` (gateway identity) and
  `platform`. Returned context is **appended to the user message and kept
  in history** (not ephemeral). The one-time "/phone allow" notice is
  therefore sent once and stays in history.
- `transform_llm_output` kwargs: `response_text`, `session_id`, `model`,
  `platform`, `turn_id`. The first non-empty string returned replaces the
  reply.
- `on_session_finalize` kwargs: `session_id`, `platform`, `reason`; the
  gateway `/new` path adds `old_session_id` and `new_session_id`. A returned
  string is sent to the user. We return `None`.
- Tool handlers are **sync**, receive only the kwargs they declare
  (`task_id`, `session_id`, `conversation_id`, `user_task`), and return a
  JSON string. (Async handlers would need `is_async=True`.)

### 2.7 Local testing

There is no path install. For local smoke tests, symlink the repo into
`$HERMES_HOME/plugins/advisor-call`, then run `hermes plugins enable
advisor-call`, `hermes plugins validate <path>` and `hermes plugins doctor
<path>`. Tests point `HERMES_HOME` at a temporary directory.

---

## 3. Implementation notes (Phase 1)

- **Call ids** are `c-` plus 8 hex characters (the plan's 4 would collide in
  the ledger's `calls` primary key over months of use).
- **Concurrent sessions.** Every exchange reserves its pre-flight estimate
  while it is in flight. Other sessions' pre-flight checks add those
  reservations to today's spend, so two sessions cannot both spend the last
  of today's allowance.
- **A failed exchange** leaves the advisor-side conversation unchanged (the
  executor's message is not appended), so user/assistant turns always
  alternate. If it was billed, it still counts toward the message limit.
- **A user hang-up while a reply is in flight** ends the call at once; the
  late reply's cost is still added and the call row is updated.
- **Pre-flight with caching on** is an upper bound consistent with the
  estimate (section 2.1): the reused prefix at the input price, the rest of
  the prompt at the cache-write price, all at the high 1/3 token ratio, plus
  a full `max_tokens` reply.
- **Redaction** calls Hermes' `redact_sensitive_text(text, force=True,
  redact_url_credentials=True)`, so it applies even if the user turned
  Hermes' own redaction off. Verified live: API keys and URL passwords are
  masked. A small built-in redactor is used only if Hermes' cannot be
  imported.
- **Ledger file** is `plugin-data/advisor-call/ledger.db`.

## 4. Implementation notes (Phase 2)

- **`runtime.py`** (not in the plan's layout) holds the shared state: call
  manager, ledger, captured context, settings and `ctx.llm` accessors. Tools,
  hooks and `/phone` all go through one `Runtime`, which tests build with
  fakes.
- **Model and provider** for each line are read from Hermes'
  `auxiliary.advisor_call_<line>` slot when set, falling back to the line's
  defaults. The provider is what a reply is checked against (section 2.4).
- **Redaction covers executor messages too**, not only the opening
  transcript: every `advisor_call_say` message and the opening question pass
  through the redactor, since all of it leaves the machine.
- **A failed opening exchange ends the call** with reason `error`, so no call
  is left open without its transcript. The executor is told it may open
  again (which asks for approval again unless the session was approved).
- **No usage reported on a successful reply** bills the pre-flight upper
  bound, flagged as an estimate. A failure with no usage is not billed; a
  failure with usage (for example an empty reply) is.
- **Restart detection.** A call row is written to the ledger when the call
  opens. After a gateway restart, `advisor_call_say` with that `call_id`
  finds the open row and reports "call dropped (gateway restarted)".

## 5. Implementation notes (Phase 3)

- **End-to-end check in real Hermes**, no network and no spend:
  `scripts/hermes-e2e.sh` loads the plugin into a throwaway Hermes home and
  points the advisor slot at a local fake Anthropic server. It checks the
  approval directive, a full open / say / hangup through Hermes' own tool
  dispatch (`model_tools.handle_function_call`, so `pre_tool_call` runs),
  the model and `max_tokens` on the wire, breakpoint A on the opening and
  A + B on the follow-up, and that the bill is flagged as an estimate
  because Hermes dropped the server's cache counts (O-2 confirmed). The
  script relaxes Hermes' "Anthropic-compatible host" check for 127.0.0.1
  inside its own process only; it lives under `tests/e2e/` so Hermes'
  private-import lint skips it.
- **Approvals mode** is read from Hermes' `approvals.mode` with the same
  normalisation Hermes uses (YAML `off` is false; unknown means manual).
- **The sender check also guards `advisor_call_say`**, not only
  `advisor_call_open`, so a different sender in the same session cannot
  drive an approved call.
- **Session ids for `/phone`.** The gateway passes the chat's current
  session id, the same one the agent's tools and hooks see. If `/phone`
  is the first message in a new chat there is no session yet, and the
  command says to send a message first.
- **`/phone ask`** opens a one-message call marked `initiated_by: user`,
  tells the advisor the message is from the user, ends the call, and
  suppresses the executor's bill footer for it. It is refused while the
  executor has a call open.
- **Bill footer** marks estimated amounts: `~$0.12 of $0.50 (estimate)`.
- **The root `__init__.py` imports lazily** inside `register()`, so pytest
  can import the repo root outside a package.
- **Source files must not contain invisible characters.** Hermes' security
  scan flags them; the reason sanitiser builds its pattern from code points.
