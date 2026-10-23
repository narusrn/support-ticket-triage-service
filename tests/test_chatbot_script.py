import json
import sys

from fastapi.testclient import TestClient
from rich.console import Console

import agents.orchestrator as orchestrator_module
import scripts.chatbot as chatbot_module
from config import settings
from scripts.chatbot import main, render_decision

_DECISION = {
    "urgency": "critical", "action": "escalate_to_human", "confidence": 0.9,
    "draft_reply": None, "tool_calls": [
        {"tool_call_id": "tc_abc123", "tool_name": "issue_refund", "args": {"amount_usd": 89.97},
         "status": "pending_approval", "reason": None, "result": None},
    ],
}

_HOST_PREFIX = "http://127.0.0.1:8000"

_VALID_CLASSIFICATION = {
    "product_area": "billing", "issue_type": "duplicate_charge", "sentiment": "angry",
    "language": "en", "confidence": 0.9, "rationale": "3 charges in 2h", "urgency": "critical",
}


class _FakeResponse:
    def __init__(self, tool_calls=None, content=""):
        self.tool_calls = tool_calls or []
        self.content = content


def _handoff_call(next_: str, reason: str = "test fixture"):
    return {"name": "HandoffTool", "args": {"next": next_, "reason": reason}, "id": "call_1"}


class _FakeSupervisorLLM:
    """Mirrors the old route_fn's logic (classify first, then decide) by inspecting the
    conversation text instead of raw state -- a real supervisor_llm only ever sees the
    LangChain message list, not the state dict."""
    def invoke(self, messages):
        texts = " ".join(getattr(m, "content", "") or "" for m in messages)
        if "Triage Classifier:" not in texts:
            return _FakeResponse([_handoff_call("triage_classifier")])
        return _FakeResponse([_handoff_call("action_decider")])


def _mock_decide_with_refund(state, messages):
    return {"action": "escalate_to_human", "rationale": "needs approval", "confidence": 0.9,
            "cited_kb_doc_ids": [], "draft_reply": None,
            "pending_tool": {"tool_name": "issue_refund", "args": {"charge_ids": ["ch_1"], "amount_usd": 89.97}}}


def test_render_decision_shows_urgency_action_and_tool_call():
    console = Console(record=True, width=120)
    render_decision(console, {"status": "awaiting_approval", "decision": _DECISION})
    out = console.export_text()
    assert "CRITICAL" in out
    assert "escalate_to_human" in out
    assert "issue_refund" in out
    assert "pending_approval" in out


def test_render_decision_shows_agent_reply():
    console = Console(record=True, width=120)
    decision_no_tools = {**_DECISION, "tool_calls": []}
    render_decision(console, {"status": "completed", "decision": decision_no_tools, "reply": "It's critical because of three charges."})
    out = console.export_text()
    assert "three charges" in out


def test_chatbot_script_end_to_end_ingest_consult_approve(tmp_path, monkeypatch, capsys):
    """Mirrors tests/test_chat_script.py's end-to-end test, against the rich-based
    chatbot.py instead -- same real app, same ingest -> consult -> approve flow,
    just through requests.request() (what the rich version calls) instead of
    requests.post/get."""
    monkeypatch.setattr(settings, "DATABASE_PATH", str(tmp_path / "chatbot_e2e.db"))
    monkeypatch.setattr(settings, "LOG_DIR", str(tmp_path / "logs"))
    monkeypatch.setattr(settings, "API_KEY", "")
    monkeypatch.setattr(orchestrator_module, "_default_classify_fn", lambda state, messages: _VALID_CLASSIFICATION)
    monkeypatch.setattr(orchestrator_module, "_default_supervisor_llm", lambda: _FakeSupervisorLLM())
    monkeypatch.setattr(orchestrator_module, "_default_decide_fn", _mock_decide_with_refund)

    from api.main import app
    with TestClient(app) as client:
        def fake_request(method, url, **kwargs):
            path = url[len(_HOST_PREFIX):]
            return client.request(method, path, **kwargs)

        monkeypatch.setattr(chatbot_module.requests, "request", fake_request)

        ticket_file = tmp_path / "ticket.json"
        ticket_file.write_text(json.dumps({
            "messages": [{"role": "customer", "content": "My payment failed, now I see three charges."}],
            "customer_metadata": {"plan": "free", "tenure_months": 4},
        }))
        monkeypatch.setattr(sys, "argv", ["chatbot.py", "--ticket", str(ticket_file)])

        captured = {}
        real_render_decision = chatbot_module.render_decision

        def spy_render_decision(console, data):
            captured["last"] = data
            real_render_decision(console, data)

        monkeypatch.setattr(chatbot_module, "render_decision", spy_render_decision)

        def scripted_input(prompt=""):
            if "first" not in captured:
                captured["first"] = True
                return "why is this urgent?"
            if "second" not in captured:
                captured["second"] = True
                tool_call_id = captured["last"]["decision"]["tool_calls"][0]["tool_call_id"]
                return f"/approve {tool_call_id} verified in Stripe"
            return "/quit"

        monkeypatch.setattr("builtins.input", scripted_input)

        main()

    out = capsys.readouterr().out
    assert "Agent" in out  # the consult reply panel got rendered
    assert captured["last"]["decision"]["tool_calls"][0]["status"] == "executed"
