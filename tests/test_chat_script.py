import json
import sys

from fastapi.testclient import TestClient

import agents.orchestrator as orchestrator_module
import scripts.chat as chat_module
from config import settings
from scripts.chat import _print_decision, main, parse_command, read_pasted_thread

_DECISION = {
    "urgency": "critical", "action": "escalate_to_human", "confidence": 0.9,
    "draft_reply": None, "tool_calls": [],
}

_HOST_PREFIX = "http://127.0.0.1:8000"  # chat.py's default --base-url, minus the path

_VALID_CLASSIFICATION = {
    "product_area": "billing", "issue_type": "duplicate_charge", "sentiment": "angry",
    "language": "en", "confidence": 0.9, "rationale": "3 charges in 2h", "urgency": "critical",
}


def _mock_route_fn(state):
    if state.get("extracted") is None:
        return {"next": "triage_classifier", "reason": "new ticket"}
    if state.get("decision") is not None:
        return {"next": "end", "reason": "already decided"}
    return {"next": "action_decider", "reason": "test fixture"}


def _mock_decide_with_refund(state):
    return {"action": "escalate_to_human", "rationale": "needs approval", "confidence": 0.9,
            "cited_kb_doc_ids": [], "draft_reply": None,
            "pending_tool": {"tool_name": "issue_refund", "args": {"charge_ids": ["ch_1"], "amount_usd": 89.97}}}


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


def test_print_decision_shows_the_agent_reply(capsys):
    """Found live: an advisory POST .../messages response that actually carries a
    `reply` (now that api/schemas/decision.py's TicketResponse has the field at all)
    must get printed -- this was the whole point of typing a free-text consultation
    question into the REPL instead of a ticket-filing command."""
    _print_decision({"status": "completed", "decision": _DECISION, "reply": "It's critical because..."})
    assert "agent: It's critical because..." in capsys.readouterr().out


def test_print_decision_without_a_reply_prints_no_agent_line(capsys):
    _print_decision({"status": "completed", "decision": _DECISION})
    assert "agent:" not in capsys.readouterr().out


def test_chat_script_end_to_end_ingest_consult_approve(tmp_path, monkeypatch, capsys):
    """Actually runs scripts/chat.py's main(), not just its extracted pure helpers,
    against a real FastAPI app -- the exact "ingest a ticket, then type a free-text
    consultation question, then approve" flow that caught two real bugs when demoed
    manually: the dropped `reply` field and un-reconfigured stdin mangling non-ASCII
    input. The LLM boundary is mocked except the one advisory-reply call, same as
    test_api.py's test_full_lifecycle_ingest_message_approve_history."""
    monkeypatch.setattr(settings, "DATABASE_PATH", str(tmp_path / "chat_e2e.db"))
    monkeypatch.setattr(settings, "LOG_DIR", str(tmp_path / "logs"))
    monkeypatch.setattr(settings, "API_KEY", "")
    monkeypatch.setattr(orchestrator_module, "_default_classify_fn", lambda state: _VALID_CLASSIFICATION)
    monkeypatch.setattr(orchestrator_module, "_default_route_fn", _mock_route_fn)
    monkeypatch.setattr(orchestrator_module, "_default_decide_fn", _mock_decide_with_refund)

    from api.main import app
    with TestClient(app) as client:
        def fake_post(url, **kwargs):
            return client.post(url[len(_HOST_PREFIX):], **kwargs)

        def fake_get(url, **kwargs):
            return client.get(url[len(_HOST_PREFIX):], **kwargs)

        monkeypatch.setattr(chat_module.requests, "post", fake_post)
        monkeypatch.setattr(chat_module.requests, "get", fake_get)

        ticket_file = tmp_path / "ticket.json"
        ticket_file.write_text(json.dumps({
            "messages": [{"role": "customer", "content": "My payment failed, now I see three charges."}],
            "customer_metadata": {"plan": "free", "tenure_months": 4},
        }))
        monkeypatch.setattr(sys, "argv", ["chat.py", "--ticket", str(ticket_file)])

        captured = {}
        real_print_decision = chat_module._print_decision

        def spy_print_decision(data):
            captured["last"] = data
            real_print_decision(data)

        monkeypatch.setattr(chat_module, "_print_decision", spy_print_decision)

        def scripted_input():
            yield "why is this urgent?"
            tool_call_id = captured["last"]["decision"]["tool_calls"][0]["tool_call_id"]
            yield f"/approve {tool_call_id} verified in Stripe"
            yield "/quit"

        gen = scripted_input()
        monkeypatch.setattr("builtins.input", lambda *a: next(gen))

        main()

    out = capsys.readouterr().out
    assert "agent:" in out  # the consult reply actually got printed, not silently dropped
    assert captured["last"]["decision"]["tool_calls"][0]["status"] == "executed"
