from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from agents.runner import Runner
from storage import db

MESSAGES = [
    {"role": "customer", "content": "My payment failed when I tried to upgrade."},
    {"role": "customer", "content": "Now I see THREE charges of $29.99."},
]


def _classify_billing(state):
    return {"product_area": "billing", "issue_type": "duplicate_charge", "sentiment": "angry",
            "language": "en", "confidence": 0.9, "rationale": "3 charges", "urgency": "critical"}


def _decide_with_refund(state):
    return {"action": "escalate_to_human", "rationale": "needs approval", "confidence": 0.9,
            "cited_kb_doc_ids": [], "draft_reply": None,
            "pending_tool": {"tool_name": "issue_refund", "args": {"charge_ids": ["ch_1"], "amount_usd": 89.97}}}


def _decide_auto_respond(state):
    return {"action": "auto_respond", "rationale": "FAQ answers this", "confidence": 0.9,
            "cited_kb_doc_ids": ["faq_dark_mode"], "draft_reply": "Dark mode is on the roadmap.",
            "pending_tool": None}


def _route_to_action_decider(state):
    if state.get("extracted") is None:
        return {"next": "triage_classifier"}
    if state.get("decision") is not None:
        return {"next": "end"}
    return {"next": "action_decider"}


def _runner(tmp_path, **brain_fns) -> Runner:
    db_path = str(tmp_path / "test.db")
    db.init_db(db_path)
    conn = db.get_connection(db_path)
    return Runner(conn, InMemorySaver(), **brain_fns)


async def test_ingest_ticket_persists_conversation_messages_and_decision(tmp_path):
    runner = _runner(tmp_path, classify_fn=_classify_billing, decide_fn=_decide_with_refund,
                      route_fn=_route_to_action_decider)

    result = await runner.ingest_ticket("c_1", MESSAGES, {"plan": "free", "tenure_months": 4})

    assert result["status"] == "awaiting_approval"
    assert result["decision"]["action"] == "escalate_to_human"
    assert len(result["decision"]["tool_calls"]) == 1
    assert result["decision"]["tool_calls"][0]["status"] == "pending_approval"

    stored_messages = db.get_messages(runner.conn, "c_1")
    assert len(stored_messages) == 2
    stored_conv = db.get_conversation(runner.conn, "c_1")
    assert stored_conv["status"] == "awaiting_approval"


async def test_ingest_ticket_no_pending_tool_is_immediately_completed(tmp_path):
    runner = _runner(tmp_path, classify_fn=_classify_billing, decide_fn=_decide_auto_respond,
                      route_fn=_route_to_action_decider)

    result = await runner.ingest_ticket("c_2", MESSAGES, {"plan": "pro", "tenure_months": 5})

    assert result["status"] == "completed"
    assert result["decision"]["tool_calls"] == []


async def test_approve_executes_tool_and_marks_completed(tmp_path):
    runner = _runner(tmp_path, classify_fn=_classify_billing, decide_fn=_decide_with_refund,
                      route_fn=_route_to_action_decider)
    ingested = await runner.ingest_ticket("c_3", MESSAGES, {})
    tool_call_id = ingested["decision"]["tool_calls"][0]["tool_call_id"]

    result = await runner.approve("c_3", tool_call_id, reason="verified in Stripe")

    assert result["status"] == "completed"
    assert result["replayed"] is False
    tc = result["decision"]["tool_calls"][0]
    assert tc["status"] == "executed"
    assert tc["reason"] == "verified in Stripe"
    assert tc["result"]["status"] == "succeeded"


async def test_approve_twice_is_idempotent_second_call_is_replayed(tmp_path):
    runner = _runner(tmp_path, classify_fn=_classify_billing, decide_fn=_decide_with_refund,
                      route_fn=_route_to_action_decider)
    ingested = await runner.ingest_ticket("c_4", MESSAGES, {})
    tool_call_id = ingested["decision"]["tool_calls"][0]["tool_call_id"]

    first = await runner.approve("c_4", tool_call_id)
    second = await runner.approve("c_4", tool_call_id)

    assert first["replayed"] is False
    assert second["replayed"] is True
    # same refund_id both times -- proves the mock tool only actually ran once
    assert first["decision"]["tool_calls"][0]["result"]["refund_id"] == \
           second["decision"]["tool_calls"][0]["result"]["refund_id"]


async def test_reject_never_executes(tmp_path):
    runner = _runner(tmp_path, classify_fn=_classify_billing, decide_fn=_decide_with_refund,
                      route_fn=_route_to_action_decider)
    ingested = await runner.ingest_ticket("c_5", MESSAGES, {})
    tool_call_id = ingested["decision"]["tool_calls"][0]["tool_call_id"]

    result = await runner.reject("c_5", tool_call_id, reason="amount looks wrong")

    assert result["decision"]["tool_calls"][0]["status"] == "rejected"
    assert result["decision"]["tool_calls"][0]["result"] is None


async def test_send_message_on_decided_conversation_uses_advisory_path(tmp_path):
    runner = _runner(tmp_path, classify_fn=_classify_billing, decide_fn=_decide_auto_respond,
                      route_fn=_route_to_action_decider)
    await runner.ingest_ticket("c_6", MESSAGES, {})

    advisory_calls = []

    def fake_advisory(state, message_text):
        advisory_calls.append((state["decision"]["action"], message_text))
        return "Hi! Your ticket is already resolved -- let me know if you need anything else."

    runner.advisory_fn = fake_advisory
    result = await runner.send_message("c_6", "Hi, thanks!")

    assert advisory_calls == [("auto_respond", "Hi, thanks!")]
    assert result["reply"].startswith("Hi!")
    # decision/status unchanged by a chit-chat follow-up -- the graph was never re-entered
    assert result["status"] == "completed"
    assert result["decision"]["action"] == "auto_respond"

    stored_messages = db.get_messages(runner.conn, "c_6")
    assert stored_messages[-2]["role"] == "operator"
    assert stored_messages[-2]["content"] == "Hi, thanks!"
    assert stored_messages[-1]["role"] == "agent"


async def test_run_row_persisted_with_matching_trace_event_count(tmp_path):
    runner = _runner(tmp_path, classify_fn=_classify_billing, decide_fn=_decide_auto_respond,
                      route_fn=_route_to_action_decider)
    await runner.ingest_ticket("c_7", MESSAGES, {})

    runs = db.get_runs(runner.conn, "c_7")
    assert len(runs) == 1
    run = runs[0]
    assert run["completed"] is True
    assert run["final_action"] == "auto_respond"

    trace_events = db.get_trace_events(runner.conn, "c_7")
    assert run["steps"] == max(e["step_index"] for e in trace_events)


async def test_restart_survival_resume_works_with_a_fresh_runner_and_real_sqlite_checkpointer(tmp_path):
    db_path = str(tmp_path / "restart_test.db")
    db.init_db(db_path)
    conn1 = db.get_connection(db_path)

    async with AsyncSqliteSaver.from_conn_string(str(tmp_path / "checkpoint.sqlite")) as checkpointer1:
        runner1 = Runner(conn1, checkpointer1, classify_fn=_classify_billing, decide_fn=_decide_with_refund,
                          route_fn=_route_to_action_decider)
        ingested = await runner1.ingest_ticket("c_8", MESSAGES, {})
        tool_call_id = ingested["decision"]["tool_calls"][0]["tool_call_id"]
        assert ingested["status"] == "awaiting_approval"
    # checkpointer1's connection is now closed -- simulates the process exiting entirely

    conn2 = db.get_connection(db_path)  # reopen the same storage DB file
    async with AsyncSqliteSaver.from_conn_string(str(tmp_path / "checkpoint.sqlite")) as checkpointer2:
        runner2 = Runner(conn2, checkpointer2, classify_fn=_classify_billing, decide_fn=_decide_with_refund,
                          route_fn=_route_to_action_decider)
        result = await runner2.approve("c_8", tool_call_id, reason="verified after restart")

    assert result["status"] == "completed"
    assert result["decision"]["tool_calls"][0]["status"] == "executed"
    assert result["decision"]["tool_calls"][0]["reason"] == "verified after restart"
