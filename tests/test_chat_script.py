from scripts.chat import parse_command, read_pasted_thread


def test_plain_text_is_a_message():
    assert parse_command("why is this urgent?") == ("message", {"text": "why is this urgent?"})


def test_quit_and_exit():
    assert parse_command("/quit") == ("quit", {})
    assert parse_command("/exit") == ("quit", {})


def test_history():
    assert parse_command("/history") == ("history", {})


def test_approve_with_reason():
    assert parse_command("/approve tc_abc123 verified in Stripe") == \
        ("approve", {"tool_call_id": "tc_abc123", "reason": "verified in Stripe"})


def test_reject_without_reason():
    assert parse_command("/reject tc_abc123") == ("reject", {"tool_call_id": "tc_abc123", "reason": None})


def test_unknown_slash_command_falls_back_to_message():
    # anything not matching a known command is just sent as a chat message, not an error --
    # matches the API's own advisory endpoint, which never rejects free text
    assert parse_command("/unknown foo") == ("message", {"text": "/unknown foo"})


def test_read_pasted_thread_stops_at_a_lone_dot():
    lines = ["My payment failed when I tried to upgrade.", "Still broken.", ".", "ignored after the dot"]
    assert read_pasted_thread(lines) == "My payment failed when I tried to upgrade.\nStill broken."


def test_read_pasted_thread_without_a_sentinel_consumes_everything():
    assert read_pasted_thread(["line one", "line two"]) == "line one\nline two"
