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

### 2.1 Caching and billing (from O-2): **open decision**

Hermes hides cache usage, so the plugin cannot bill a cached exchange
correctly. Options:

| | Option | Billing | Cost |
|---|---|---|---|
| A | Ship with `cache_ttl: off` by default; keep the marker code behind the setting | Exact: with no caching, `input_tokens` is the whole prompt | Follow-ups re-send the transcript at full input price. About $0.02 more per follow-up for an 11K-token opening |
| B | Keep caching on; bill pessimistically by treating every input token Hermes did not report as a cache write | Always over-states (reads are charged as writes), so caps hold but pacing is stricter than reality, and `/phone bill` will not match the Console (L-12) | Cheapest real spend |
| C | Fix upstream: keep `cache_read_input_tokens` and `cache_creation_input_tokens` in the adapter's usage | Exact with caching | One-line Hermes change; needs a release |

Recommendation: **A now, and C as an upstream issue.** When cache counts
arrive (a nonzero `cache_write_tokens` on a call with markers), default
caching back on in a later release. If `cache_ttl` is set to `5m` or `1h`
on a Hermes that still drops the counts, the plugin uses B's pessimistic
billing and logs a warning.

### 2.2 `max_tokens` (from O-3b)

- Register each line's auxiliary task with
  `defaults={"provider": ..., "model": ..., "timeout": ...,
  "extra_body": {"max_tokens": <advisor.max_tokens>}}`. Still pass
  `max_tokens=` to `complete()` for routes that honour it.
- Because `extra_body` is fixed at registration, a change to
  `advisor.max_tokens` needs a restart. A user's own
  `auxiliary.advisor_call_<line>.extra_body` overrides it.
- The pre-flight estimate assumes the full `max_tokens` of output, so the
  per-call cap stays bounded only if this override works. Phase 4 adds a
  live check: a deliberately long question with `max_tokens: 300` must come
  back with at most 300 output tokens.
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
