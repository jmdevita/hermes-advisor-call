"""U-12 transcript rendering, plus capture."""

from advisor_call.context import (
    OMITTED_PREFIX,
    ContextStore,
    fallback_redact,
    render_transcript,
)


def render(messages, limit=40_000, tool=2_000, redact=None):
    return render_transcript(messages, limit_chars=limit, tool_chars=tool, redact=redact)


def test_system_skipped_roles_rendered():
    text = render(
        [
            {"role": "system", "content": "SECRET SYSTEM PROMPT"},
            {"role": "user", "content": "fix the build"},
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "looking"},
                    {"type": "image_url", "image_url": {"url": "x"}},
                ],
            },
        ]
    )
    assert "SECRET SYSTEM PROMPT" not in text
    assert text == "### USER\nfix the build\n\n### ASSISTANT\nlooking\n[image]"


def test_tool_calls_and_results():
    long_args = '{"path": "' + "a" * 1000 + '"}'
    text = render(
        [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "t1",
                        "type": "function",
                        "function": {"name": "read_file", "arguments": long_args},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "t1", "name": "read_file", "content": "file body"},
        ]
    )
    assert "[tool call] read_file(" in text
    call_line = next(line for line in text.splitlines() if line.startswith("[tool call]"))
    assert len(call_line) < 450  # arguments cut to ~400 characters
    assert "### TOOL RESULT (read_file)\nfile body" in text


def test_tool_output_truncated_head_and_tail():
    body = "HEAD" + "x" * 5000 + "TAIL"
    text = render([{"role": "tool", "tool_call_id": "t1", "content": body}], tool=100)
    assert "HEAD" in text and "TAIL" in text
    assert "characters omitted" in text
    assert len(text) < 300


def test_tail_kept_at_limit():
    messages = [{"role": "user", "content": f"message {i} " + "y" * 200} for i in range(100)]
    text = render(messages, limit=2_000)
    assert text.startswith(OMITTED_PREFIX)
    assert len(text) <= 2_000
    assert "message 99" in text and "message 0 " not in text
    assert text[len(OMITTED_PREFIX) :].startswith("### USER")  # starts on a boundary


def test_in_flight_advisor_call_excluded():
    text = render(
        [
            {"role": "user", "content": "get a second opinion"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {"id": "old", "function": {"name": "advisor_call_open", "arguments": "{}"}}
                ],
            },
            {"role": "tool", "tool_call_id": "old", "content": "earlier advice"},
            {
                "role": "assistant",
                "content": "calling",
                "tool_calls": [
                    {
                        "id": "now",
                        "function": {
                            "name": "advisor_call_say",
                            "arguments": '{"message": "IN FLIGHT"}',
                        },
                    }
                ],
            },
        ]
    )
    assert "IN FLIGHT" not in text
    assert "earlier advice" in text  # a finished call stays in the transcript
    assert "calling" in text


def test_anthropic_style_blocks():
    text = render(
        [
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "hidden"},
                    {"type": "tool_use", "id": "u1", "name": "terminal", "input": {"cmd": "ls"}},
                    {"type": "tool_use", "id": "u2", "name": "advisor_call_open", "input": {}},
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "u1",
                        "content": [{"type": "text", "text": "README.md"}],
                    }
                ],
            },
        ]
    )
    assert "hidden" not in text
    assert '[tool call] terminal({"cmd": "ls"})' in text
    assert "[tool result]\nREADME.md" in text
    assert "advisor_call_open" not in text  # in flight: no result yet


def test_redaction_hook_applied_last():
    seen = []

    def redact(text):
        seen.append(text)
        return text.replace("hunter2", "[REDACTED]")

    text = render([{"role": "user", "content": "password is hunter2"}], redact=redact)
    assert "hunter2" not in text and seen  # each piece, then the final text


def test_secret_split_by_a_cut_is_still_redacted():
    # Review finding 4: redaction runs before any cut.
    secret = "ghp_" + "A1b2C3d4E5f6G7h8I9j0K1l2M3n4O5p6Q7r8"
    body = "x" * 995 + " " + secret + " " + "y" * 973  # the cut falls inside the token
    msgs = [
        {"role": "tool", "tool_call_id": "t", "content": body},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": "c", "function": {"name": "f", "arguments": "a" * 390 + " " + secret}}
            ],
        },
    ]
    out = render(msgs, tool=2_000, redact=fallback_redact)
    for i in range(len(secret) - 8):
        assert secret[i : i + 8] not in out


def test_fallback_redactor():
    # Built at runtime so the source holds nothing shaped like a real key
    # (secret scanners flag that, even for obvious fakes).
    fake_key = "-".join(["sk", "ant", "api03", "x" * 16])
    raw = f"key {fake_key} and Bearer abcdefghijklmnopqrstuvwxyz"
    out = fallback_redact(raw)
    assert fake_key not in out and "abcdefghijklmnopqrstuvwxyz" not in out


def test_junk_messages_tolerated():
    assert (
        render([None, "text", {"role": "user"}, {"content": "no role"}]) == "### UNKNOWN\nno role"
    )


# ------------------------------------------------------------------- capture


def test_capture_prefers_newest_snapshot():
    store = ContextStore()
    store.record_turn(
        "s1",
        conversation_history=[{"role": "user", "content": "old"}],
        user_message="new question",
        sender_id=42,
        platform="telegram",
    )
    assert store.messages("s1")[-1] == {"role": "user", "content": "new question"}
    mid_turn = [
        {"role": "user", "content": "new question"},
        {"role": "tool", "tool_call_id": "t", "content": "result"},
    ]
    store.record_request("s1", mid_turn)
    assert store.messages("s1") is mid_turn  # stored by reference
    store.record_turn("s1", conversation_history=[], user_message="next turn")
    assert store.messages("s1") == [{"role": "user", "content": "next turn"}]
    assert store.identity("s1") == ("42", "telegram")


def test_capture_does_not_duplicate_user_message():
    store = ContextStore()
    history = [{"role": "user", "content": "hello"}]
    store.record_turn("s1", conversation_history=history, user_message="hello")
    assert store.messages("s1") == history


def test_capture_isolated_and_bounded():
    store = ContextStore(max_sessions=2)
    for sid in ("a", "b", "c"):
        store.record_turn(sid, conversation_history=[], user_message=sid)
    assert store.messages("a") == []  # evicted
    assert store.messages("c") == [{"role": "user", "content": "c"}]
    assert store.messages(None) == [] and store.identity("missing") is None
    store.record_request(None, [])  # ignored, no error
