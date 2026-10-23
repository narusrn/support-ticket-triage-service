import pytest
from fastapi.testclient import TestClient

import agents.orchestrator as orchestrator_module
from config import settings

VALID_CLASSIFICATION = {
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


def _mock_decide_auto_respond(state, messages):
    return {"action": "auto_respond", "rationale": "FAQ answers this", "confidence": 0.9,
            "cited_kb_doc_ids": ["faq_dark_mode"], "draft_reply": "Dark mode is on the roadmap.",
            "pending_tool": None}


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "DATABASE_PATH", str(tmp_path / "test.db"))
    monkeypatch.setattr(settings, "LOG_DIR", str(tmp_path / "logs"))
    monkeypatch.setattr(settings, "API_KEY", "")  # auth disabled for the tests that aren't about auth itself
    monkeypatch.setattr(orchestrator_module, "_default_classify_fn", lambda state, messages: VALID_CLASSIFICATION)
    monkeypatch.setattr(orchestrator_module, "_default_supervisor_llm", lambda: _FakeSupervisorLLM())
    monkeypatch.setattr(orchestrator_module, "_default_decide_fn", _mock_decide_with_refund)

    from api.main import app
    with TestClient(app) as c:
        yield c


def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_ingest_ticket_returns_201_with_pending_approval(client):
    response = client.post("/api/v1/tickets", json={
        "messages": [{"role": "customer", "content": "My payment failed, now I see three charges."}],
        "customer_metadata": {"plan": "free", "tenure_months": 4},
    })
    assert response.status_code == 201
    body = response.json()
    assert body["status"] == "awaiting_approval"
    assert body["decision"]["action"] == "escalate_to_human"
    assert body["decision"]["tool_calls"][0]["status"] == "pending_approval"


def test_ingest_ticket_accepts_a_raw_text_plain_body_with_no_json_at_all(client):
    """On request: no JSON wrapping ceremony required at all -- POST the ticket text
    itself with Content-Type: text/plain and it's treated as one customer message,
    customer_metadata left for the classifier to infer from the text."""
    response = client.post(
        "/api/v1/tickets",
        content="My payment failed, now I see three charges.",
        headers={"Content-Type": "text/plain"},
    )
    assert response.status_code == 201
    body = response.json()
    assert body["status"] == "awaiting_approval"
    assert body["decision"]["action"] == "escalate_to_human"


def test_ingest_ticket_rejects_an_empty_text_plain_body(client):
    response = client.post("/api/v1/tickets", content="", headers={"Content-Type": "text/plain"})
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_error"


def test_ingest_ticket_rejects_empty_messages_with_422_and_error_shape(client):
    response = client.post("/api/v1/tickets", json={"messages": [], "customer_metadata": {"plan": "free", "tenure_months": 1}})
    assert response.status_code == 422
    body = response.json()
    assert "error" in body
    assert body["error"]["code"] == "validation_error"


def test_get_conversation_404_for_unknown_id(client):
    response = client.get("/api/v1/conversations/c_does_not_exist")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "conversation_not_found"


def test_full_lifecycle_ingest_message_approve_history(client):
    ingest = client.post("/api/v1/tickets", json={
        "messages": [{"role": "customer", "content": "My payment failed, now I see three charges."}],
        "customer_metadata": {"plan": "free", "tenure_months": 4},
    }).json()
    conversation_id = ingest["conversation_id"]
    tool_call_id = ingest["decision"]["tool_calls"][0]["tool_call_id"]

    # POST .../messages -- advisory turn, conversation already has a decision
    msg_response = client.post(f"/api/v1/conversations/{conversation_id}/messages", json={"message": "why is this urgent?"})
    assert msg_response.status_code == 200
    # found live: TicketResponse had no `reply` field, so FastAPI's response_model
    # silently dropped the text Runner.send_message() put there -- must actually come
    # back to the caller, not just land in storage.db
    assert msg_response.json()["reply"]

    # POST .../approve
    approve_response = client.post(
        f"/api/v1/conversations/{conversation_id}/tool-calls/{tool_call_id}/approve",
        json={"reason": "verified in Stripe"},
    )
    assert approve_response.status_code == 200
    approved = approve_response.json()
    assert approved["status"] == "completed"
    assert approved["decision"]["tool_calls"][0]["status"] == "executed"

    # duplicate approve -- idempotent, must not double-execute or error
    second_approve = client.post(f"/api/v1/conversations/{conversation_id}/tool-calls/{tool_call_id}/approve")
    assert second_approve.status_code == 200
    assert second_approve.json()["decision"]["tool_calls"][0]["result"]["refund_id"] == \
           approved["decision"]["tool_calls"][0]["result"]["refund_id"]

    # GET /conversations/{id} -- full audit trail
    history = client.get(f"/api/v1/conversations/{conversation_id}")
    assert history.status_code == 200
    body = history.json()
    assert body["status"] == "completed"
    assert len(body["messages"]) >= 1
    assert body["tool_calls"][0]["status"] == "executed"
    assert body["usage"]["total_runs"] >= 2

    # the approve resume didn't change any real decision field, so it must not have
    # inserted a duplicate decisions row -- and the one decision that IS there should
    # carry the real tool_calls, not a stray [] (the decisions table has no tool_calls
    # column of its own; GET attaches the real data to the latest decision)
    assert len(body["decisions"]) == 1
    assert body["decisions"][-1]["tool_calls"][0]["status"] == "executed"


def test_reject_tool_call(client, monkeypatch):
    ingest = client.post("/api/v1/tickets", json={
        "messages": [{"role": "customer", "content": "My payment failed, now I see three charges."}],
        "customer_metadata": {"plan": "free", "tenure_months": 4},
    }).json()
    conversation_id = ingest["conversation_id"]
    tool_call_id = ingest["decision"]["tool_calls"][0]["tool_call_id"]

    response = client.post(
        f"/api/v1/conversations/{conversation_id}/tool-calls/{tool_call_id}/reject",
        json={"reason": "amount looks wrong"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["decision"]["tool_calls"][0]["status"] == "rejected"
    assert body["decision"]["tool_calls"][0]["result"] is None


def test_approve_unknown_tool_call_id_returns_404(client):
    ingest = client.post("/api/v1/tickets", json={
        "messages": [{"role": "customer", "content": "My payment failed, now I see three charges."}],
        "customer_metadata": {"plan": "free", "tenure_months": 4},
    }).json()
    conversation_id = ingest["conversation_id"]

    response = client.post(f"/api/v1/conversations/{conversation_id}/tool-calls/tc_doesnotexist/approve")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "tool_call_not_found"


def test_ticket_with_no_pending_tool_completes_immediately(tmp_path, monkeypatch):
    # decide_fn is captured as a closure variable inside build_graph() at the moment the
    # graph compiles (i.e. when the Runner is constructed during lifespan startup) -- it is
    # NOT re-read from the module global on every call. The mock must be set BEFORE the
    # TestClient (and therefore the app's lifespan) starts, not after, or it has no effect
    # on the already-built graph.
    monkeypatch.setattr(settings, "DATABASE_PATH", str(tmp_path / "test2.db"))
    monkeypatch.setattr(settings, "LOG_DIR", str(tmp_path / "logs2"))
    monkeypatch.setattr(settings, "API_KEY", "")
    monkeypatch.setattr(orchestrator_module, "_default_classify_fn", lambda state, messages: VALID_CLASSIFICATION)
    monkeypatch.setattr(orchestrator_module, "_default_supervisor_llm", lambda: _FakeSupervisorLLM())
    monkeypatch.setattr(orchestrator_module, "_default_decide_fn", _mock_decide_auto_respond)

    from api.main import app
    with TestClient(app) as local_client:
        response = local_client.post("/api/v1/tickets", json={
            "messages": [{"role": "customer", "content": "Do you support dark mode?"}],
            "customer_metadata": {"plan": "pro", "tenure_months": 5},
        })
    assert response.status_code == 201
    body = response.json()
    assert body["status"] == "completed"
    assert body["decision"]["tool_calls"] == []


# --- auth: these build their own client with API_KEY actually set, unlike the shared fixture ---

@pytest.fixture
def secured_client(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "DATABASE_PATH", str(tmp_path / "secured.db"))
    monkeypatch.setattr(settings, "LOG_DIR", str(tmp_path / "logs3"))
    monkeypatch.setattr(settings, "API_KEY", "test-secret-key")
    monkeypatch.setattr(orchestrator_module, "_default_classify_fn", lambda state, messages: VALID_CLASSIFICATION)
    monkeypatch.setattr(orchestrator_module, "_default_supervisor_llm", lambda: _FakeSupervisorLLM())
    monkeypatch.setattr(orchestrator_module, "_default_decide_fn", _mock_decide_auto_respond)

    from api.main import app
    with TestClient(app) as c:
        yield c


def _ticket_body():
    return {"messages": [{"role": "customer", "content": "Do you support dark mode?"}],
            "customer_metadata": {"plan": "pro", "tenure_months": 5}}


def test_request_without_api_key_is_rejected(secured_client):
    response = secured_client.post("/api/v1/tickets", json=_ticket_body())
    assert response.status_code == 401
    assert response.json()["error"]["code"] == "unauthorized"


def test_request_with_wrong_api_key_is_rejected(secured_client):
    response = secured_client.post("/api/v1/tickets", json=_ticket_body(), headers={"X-API-Key": "wrong-key"})
    assert response.status_code == 401


def test_request_with_correct_api_key_succeeds(secured_client):
    response = secured_client.post("/api/v1/tickets", json=_ticket_body(), headers={"X-API-Key": "test-secret-key"})
    assert response.status_code == 201


def test_health_check_does_not_require_api_key(secured_client):
    # infra/monitoring checks shouldn't need a key
    response = secured_client.get("/health")
    assert response.status_code == 200
