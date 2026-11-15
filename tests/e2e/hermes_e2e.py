"""End-to-end check against a real Hermes, with no network and no API spend.

Real Hermes loads this plugin; a local fake Anthropic server records what
Hermes sends and answers in Anthropic's streaming format. Checks the approval
directive, a full open / say / hangup through Hermes' own tool dispatch, the
cache markers and max_tokens on the wire, and that the bill is flagged as an
estimate (Hermes drops cache counts). Run it with scripts/hermes-e2e.sh (not collected by pytest).

TEST ONLY: Hermes refuses advisor hosts that are not Anthropic-compatible,
so this script relaxes that one check for 127.0.0.1 inside its own process.
"""

import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

PORT = int(sys.argv[1])
ALIAS = sys.argv[2] if len(sys.argv) > 2 else "sonnet"
MODEL_ID = {"sonnet": "claude-sonnet-5-5", "opus": "claude-opus-5-5"}[ALIAS]
SEEN: list[dict] = []
REPLIES = iter(["First advice: check the lockfile.", "Second advice: pin the version."])


class FakeAnthropic(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["content-length"])))
        SEEN.append(body)
        usage = {
            "input_tokens": 37,
            "output_tokens": 0,
            "cache_read_input_tokens": 900,
            "cache_creation_input_tokens": 300,
        }
        events = [
            (
                "message_start",
                {
                    "type": "message_start",
                    "message": {
                        "id": "msg_1",
                        "type": "message",
                        "role": "assistant",
                        "model": body.get("model"),
                        "content": [],
                        "stop_reason": None,
                        "stop_sequence": None,
                        "usage": usage,
                    },
                },
            ),
            (
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {"type": "text", "text": ""},
                },
            ),
            (
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "text_delta", "text": next(REPLIES)},
                },
            ),
            ("content_block_stop", {"type": "content_block_stop", "index": 0}),
            (
                "message_delta",
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                    "usage": {"output_tokens": 21},
                },
            ),
            ("message_stop", {"type": "message_stop"}),
        ]
        data = "".join(f"event: {n}\ndata: {json.dumps(d)}\n\n" for n, d in events).encode()
        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def markers(body):
    out = []
    for i, m in enumerate(body["messages"]):
        for part in m["content"] if isinstance(m["content"], list) else []:
            if isinstance(part, dict) and "cache_control" in part:
                out.append(i)
    return out


def main() -> int:
    server = HTTPServer(("127.0.0.1", PORT), FakeAnthropic)
    threading.Thread(target=server.serve_forever, daemon=True).start()

    import agent.auxiliary_client as aux_client

    aux_client._is_anthropic_compatible_host = lambda url: url.startswith("http://127.0.0.1:")

    import model_tools
    from hermes_cli.plugins import _get_pre_tool_call_directive_details, get_plugin_manager

    get_plugin_manager().discover_and_load(force=True)
    sid = "sess-e2e"
    args = {"reason": "stuck on\nthe build", "question": "Why does `npm ci` fail with EINTEGRITY?"}

    checks: list[tuple[str, bool, str]] = []

    def check(name, ok, detail=""):
        checks.append((name, bool(ok), detail))

    d = _get_pre_tool_call_directive_details("advisor_call_open", args, session_id=sid)
    check(
        "approval directive",
        d.action == "approve"
        and d.rule_key == f"advisor_call:{ALIAS}:{sid}"
        and 'Reason given by the model: "stuck on the build"' in (d.message or ""),
        d.message or "",
    )

    def run(tool, a):
        return json.loads(model_tools.handle_function_call(tool, a, task_id="t", session_id=sid))

    opened = run("advisor_call_open", args)
    check("open", opened.get("reply") == "First advice: check the lockfile.", str(opened)[:200])
    said = run("advisor_call_say", {"message": "Lockfile v3, node 22."})
    check("say", said.get("reply") == "Second advice: pin the version.", str(said)[:200])
    hung = run("advisor_call_hangup", {})
    check("hangup", hung.get("end_reason") == "hangup" and len(SEEN) == 2, str(hung)[:200])
    if len(SEEN) == 2:
        first, second = SEEN
        check("model on the wire", first.get("model") == MODEL_ID, first.get("model"))
        check(
            "max_tokens on the wire",
            first.get("max_tokens") == 4000 == second.get("max_tokens"),
            f"{first.get('max_tokens')}, {second.get('max_tokens')}",
        )
        check("breakpoint A only on the opening", markers(first) == [0], str(markers(first)))
        check(
            "breakpoints A and B on the follow-up", markers(second) == [0, 1], str(markers(second))
        )
    bill = said.get("bill", {})
    check(
        "bill flagged as estimate (Hermes drops cache counts)",
        bill.get("estimate") is True,
        str(bill),
    )

    for name, ok, detail in checks:
        print(f"{'PASS' if ok else 'FAIL'}  {name}" + ("" if ok else f"  -> {detail}"))
    return 0 if all(ok for _, ok, _ in checks) else 1


if __name__ == "__main__":
    sys.exit(main())
