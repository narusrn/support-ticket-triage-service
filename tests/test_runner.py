from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from agents.runner import Runner
from storage import db

MESSAGES = [
    {"role": "customer", "content": "My payment failed when I tried to upgrade."},
    {"role": "customer", "content": "Now I see THREE charges of $29.99."},
]


def _classify_billing(state, messages):
    return {"product_area": "billing", "issue_type": "duplicate_charge", "sentiment": "angry",
            "language": "en", "confidence": 0.9, "rationale": "3 charges", "urgency": "critical"}


def _decide_with_refund(state, messages):
    return {"action": "escalate_to_human", "rationale": "needs approval", "confidence": 0.9,
            "cited_kb_doc_ids": [], "draft_reply": None,
            "pending_tool": {"tool_name": "issue_refund", "args": {"charge_ids": ["ch_1"], "amount_usd": 89.97}}}


def _decide_auto_respond(state, messages):
    return {"action": "auto_respond", "rationale": "FAQ answers this", "confidence": 0.9,
            "cited_kb_doc_ids": ["faq_dark_mode"], "draft_reply": "Dark mode is on the roadmap.",
            "pending_tool": None}


class _FakeResponse:
    def __init__(self, tool_calls=None, content=""):
        self.tool_calls = tool_calls or []
        self.content = content


def _handoff_call(next_: str, reason: str = "test fixture"):
    return {"name": "HandoffTool", "args": {"next": next_, "reason": reason}, "id": "call_1"}


class _FakeSupervisorLLM:
    """Mirrors the old route_fn's logic (classify first, then decide) by inspecting the
    conversation text instead of raw state -- a real supervisor_llm only ever sees the
    LangChain message list, not the state dict. Stateless, so one instance is reusable
    across every test and every Runner it's handed to."""
    def invoke(self, messages):
        texts = " ".join(getattr(m, "content", "") or "" for m in messages)
        if "Triage Classifier:" not in texts:
            return _FakeResponse([_handoff_call("triage_classifier")])
        return _FakeResponse([_handoff_call("action_decider")])


def _runner(tmp_path, **brain_fns) -> Runner:
    db_path = str(tmp_path / "test.db")
    db.init_db(db_path)
    conn = db.get_connection(db_path)
    brain_fns.setdefault("supervisor_llm", _FakeSupervisorLLM())
    return Runner(conn, InMemorySaver(), **brain_fns)


async def test_customer_metadata_reaches_classify_fn(tmp_path):
    """Found live: customer_metadata was only ever written to storage.db, never visible to
    the LLM -- classify_fn/decide_fn/supervisor_llm all build their prompt purely from
    state["messages"]. An enterprise ticket and a free-plan ticket with identical wording
    would have classified identically, contradicting the spec's "metadata matters" and the
    whole point of designing CustomerMetadata with plan/tenure/seats in the first place."""
    seen_messages = []

    def classify_fn(state, messages):
        seen_messages.append(messages)
        return {"product_area": "billing", "issue_type": "x", "sentiment": "neutral",
                "language": "en", "confidence": 0.9, "rationale": "x", "urgency": "low"}

    runner = _runner(tmp_path, classify_fn=classify_fn, decide_fn=_decide_auto_respond)
    await runner.ingest_ticket("c_meta", [{"role": "customer", "content": "test"}],
                                {"plan": "enterprise", "tenure_months": 8, "seats": 45})

    assert len(seen_messages) == 1
    thread_text = seen_messages[0][-1]["content"]  # the explicit "messages" param's user message
    assert "plan=enterprise" in thread_text
    assert "seats=45" in thread_text


async def test_ingest_ticket_persists_conversation_messages_and_decision(tmp_path):
    runner = _runner(tmp_path, classify_fn=_classify_billing, decide_fn=_decide_with_refund)

    result = await runner.ingest_ticket("c_1", MESSAGES, {"plan": "free", "tenure_months": 4})

    assert result["status"] == "awaiting_approval"
    assert result["decision"]["action"] == "escalate_to_human"
    assert len(result["decision"]["tool_calls"]) == 1
    assert result["decision"]["tool_calls"][0]["status"] == "pending_approval"

    stored_messages = db.get_messages(runner.conn, "c_1")
    # 2 original customer messages + the graph's own handoff/status narrative, now synced
    assert [m["content"] for m in stored_messages[:2]] == [m["content"] for m in MESSAGES]
    assert any(m["content"].startswith("Triage Classifier:") for m in stored_messages)
    assert any(m["content"].startswith("Action Decider:") for m in stored_messages)
    stored_conv = db.get_conversation(runner.conn, "c_1")
    assert stored_conv["status"] == "awaiting_approval"


async def test_ingest_ticket_no_pending_tool_is_immediately_completed(tmp_path):
    runner = _runner(tmp_path, classify_fn=_classify_billing, decide_fn=_decide_auto_respond)

    result = await runner.ingest_ticket("c_2", MESSAGES, {"plan": "pro", "tenure_months": 5})

    assert result["status"] == "completed"
    assert result["decision"]["tool_calls"] == []


async def test_approve_executes_tool_and_marks_completed(tmp_path):
    runner = _runner(tmp_path, classify_fn=_classify_billing, decide_fn=_decide_with_refund)
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
    runner = _runner(tmp_path, classify_fn=_classify_billing, decide_fn=_decide_with_refund)
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
    runner = _runner(tmp_path, classify_fn=_classify_billing, decide_fn=_decide_with_refund)
    ingested = await runner.ingest_ticket("c_5", MESSAGES, {})
    tool_call_id = ingested["decision"]["tool_calls"][0]["tool_call_id"]

    result = await runner.reject("c_5", tool_call_id, reason="amount looks wrong")

    assert result["decision"]["tool_calls"][0]["status"] == "rejected"
    assert result["decision"]["tool_calls"][0]["result"] is None


async def test_send_message_on_decided_conversation_uses_advisory_path(tmp_path):
    runner = _runner(tmp_path, classify_fn=_classify_billing, decide_fn=_decide_auto_respond)
    await runner.ingest_ticket("c_6", MESSAGES, {})

    advisory_calls = []

    def fake_advisory(state, history, message_text):
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


async def test_send_message_remembers_earlier_advisory_turns(tmp_path):
    """Found live: a second consult question had no idea what the first one was just
    told, because advisory turns never touched the graph's checkpoint. send_message()
    must hand the advisory_fn everything said so far (including its own prior replies),
    not just the ticket's original decision context."""
    runner = _runner(tmp_path, classify_fn=_classify_billing, decide_fn=_decide_auto_respond)
    await runner.ingest_ticket("c_memory", MESSAGES, {})

    seen_histories = []

    def fake_advisory(state, history, message_text):
        seen_histories.append([(m["role"], m["content"]) for m in history])
        return f"reply #{len(seen_histories)}"

    runner.advisory_fn = fake_advisory
    await runner.send_message("c_memory", "first question")
    await runner.send_message("c_memory", "second question, refers to the first")

    # the second call's history must include the operator's first question AND the
    # agent's first reply -- neither existed yet when the first call's history was built
    assert ("operator", "first question") not in seen_histories[0]
    assert ("operator", "first question") in seen_histories[1]
    assert ("agent", "reply #1") in seen_histories[1]


async def test_graph_message_sync_survives_an_advisory_message_in_between(tmp_path):
    """The tricky ordering case: send_message()'s advisory path inserts messages straight
    into storage.db without ever touching the graph. If syncing were count-based instead of
    content-based, this advisory insert would throw off the count and the later approval's
    outcome message could get silently skipped."""
    runner = _runner(tmp_path, classify_fn=_classify_billing, decide_fn=_decide_with_refund)
    ingested = await runner.ingest_ticket("c_order", MESSAGES, {})
    tool_call_id = ingested["decision"]["tool_calls"][0]["tool_call_id"]

    await runner.send_message("c_order", "why is this urgent?")  # advisory -- 2 direct DB inserts, graph untouched
    await runner.approve("c_order", tool_call_id, reason="verified")

    stored = db.get_messages(runner.conn, "c_order")
    contents = [m["content"] for m in stored]
    assert "why is this urgent?" in contents  # the advisory operator turn
    assert any(c.startswith("System:") and "approved" in c for c in contents)  # the outcome message, not skipped
    # no duplicates -- every (role, content) pair appears exactly once
    assert len(contents) == len(set((m["role"], m["content"]) for m in stored))


async def test_approve_does_not_insert_a_duplicate_decision_row(tmp_path):
    """_persist_run() runs again on the approve resume, but the resume doesn't change any
    actual decision field (urgency/action/rationale/...) -- only the tool call's status.
    Without the dedup check, this would silently double the decisions table on every
    approve/reject."""
    runner = _runner(tmp_path, classify_fn=_classify_billing, decide_fn=_decide_with_refund)
    ingested = await runner.ingest_ticket("c_dedup", MESSAGES, {})
    tool_call_id = ingested["decision"]["tool_calls"][0]["tool_call_id"]

    await runner.approve("c_dedup", tool_call_id, reason="verified in Stripe")

    assert len(db.get_decisions(runner.conn, "c_dedup")) == 1


async def test_run_row_persisted_with_matching_trace_event_count(tmp_path):
    runner = _runner(tmp_path, classify_fn=_classify_billing, decide_fn=_decide_auto_respond)
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
                          supervisor_llm=_FakeSupervisorLLM())
        ingested = await runner1.ingest_ticket("c_8", MESSAGES, {})
        tool_call_id = ingested["decision"]["tool_calls"][0]["tool_call_id"]
        assert ingested["status"] == "awaiting_approval"
    # checkpointer1's connection is now closed -- simulates the process exiting entirely

    conn2 = db.get_connection(db_path)  # reopen the same storage DB file
    async with AsyncSqliteSaver.from_conn_string(str(tmp_path / "checkpoint.sqlite")) as checkpointer2:
        runner2 = Runner(conn2, checkpointer2, classify_fn=_classify_billing, decide_fn=_decide_with_refund,
                          supervisor_llm=_FakeSupervisorLLM())
        result = await runner2.approve("c_8", tool_call_id, reason="verified after restart")

    assert result["status"] == "completed"
    assert result["decision"]["tool_calls"][0]["status"] == "executed"
    assert result["decision"]["tool_calls"][0]["reason"] == "verified after restart"
