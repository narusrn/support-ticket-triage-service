import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from agents.orchestrator import GraphState, build_graph, maybe_trim_messages
from config import settings
from storage import db


# --- shared fixtures ---

def _conn(tmp_path):
    db_path = str(tmp_path / "test.db")
    db.init_db(db_path)
    conn = db.get_connection(db_path)
    db.insert_conversation(conn, "c_1", "awaiting_approval", {})
    return conn


def _initial_state(conversation_id: str = "c_1") -> GraphState:
    return {
        "conversation_id": conversation_id,
        "messages": [{"role": "system", "content": "you are a triage agent"},
                     {"role": "customer", "content": "my payment failed, now I see three charges"}],
        "urgency": None,
        "extracted": None,
        "kb_results": None,
        "decision": None,
        "pending_tool": None,
        "tool_calls": [],
        "step": 0,
        "trace": [],
        "next_hop": None,
    }


def _route_classify_then(next_after_classify: str):
    """route_fn is now consulted on every supervisor visit, including the very first one --
    this mirrors what a real LLM supervisor (guided by _supervisor.md) decides each turn:
    classify first if nothing is classified yet, then take the test's chosen next hop."""
    def route_fn(state):
        if state.get("extracted") is None:
            return {"next": "triage_classifier"}
        if state.get("decision") is not None:
            return {"next": "end"}
        return {"next": next_after_classify}
    return route_fn


VALID_CLASSIFICATION = {
    "product_area": "billing", "issue_type": "duplicate_charge", "sentiment": "angry",
    "language": "en", "confidence": 0.9, "rationale": "3 charges in 2h", "urgency": "critical",
}

VALID_FAQ_CLASSIFICATION = {
    "product_area": "ui/appearance", "issue_type": "feature_question", "sentiment": "curious",
    "language": "en", "confidence": 0.85, "rationale": "asking about dark mode", "urgency": "low",
}


# --- test 1: handoff routes to the right specialist in order, no KB for a live-incident ticket ---

def test_direct_reply_never_enters_triage_classifier_for_a_non_ticket_message(tmp_path):
    conn = _conn(tmp_path)
    classify_calls = []

    def classify_fn(state):
        classify_calls.append(1)  # should never be called for this test
        return VALID_CLASSIFICATION

    def route_fn(state):
        return {"next": "direct_reply",
                "reply": "I'm a support ticket triage assistant -- I classify tickets, search FAQs, "
                          "and decide whether to auto-respond, route, or escalate."}

    graph = build_graph(conn, InMemorySaver(), classify_fn=classify_fn, route_fn=route_fn)
    state = _initial_state()
    state["messages"] = [{"role": "customer", "content": "who are you, what can you do?"}]
    result = graph.invoke(state, {"configurable": {"thread_id": "c_1"}})

    assert classify_calls == []  # triage_classifier was never entered
    assert "__interrupt__" not in result
    assert result["decision"]["action"] == "auto_respond"
    assert "triage assistant" in result["decision"]["draft_reply"]
    assert result["messages"][-1]["role"] == "agent"
    assert "triage assistant" in result["messages"][-1]["content"]


def test_handoff_routes_in_order_and_skips_kb_for_live_incident(tmp_path):
    conn = _conn(tmp_path)
    calls = []

    def classify_fn(state):
        calls.append("classify")
        return VALID_CLASSIFICATION

    def route_fn(state):
        calls.append("route")
        if state.get("extracted") is None:
            return {"next": "triage_classifier"}
        return {"next": "action_decider"}  # live incident, not FAQ-answerable -> skip kb_retriever

    def decide_fn(state):
        calls.append("decide")
        return {"action": "escalate_to_human", "rationale": "triple charge, needs human approval",
                "confidence": 0.9, "cited_kb_doc_ids": [], "draft_reply": None,
                "pending_tool": {"tool_name": "issue_refund", "args": {"charge_ids": ["ch_1"], "amount_usd": 89.97}}}

    graph = build_graph(conn, InMemorySaver(), classify_fn=classify_fn, decide_fn=decide_fn, route_fn=route_fn)
    result = graph.invoke(_initial_state(), {"configurable": {"thread_id": "c_1"}})

    # route_fn is consulted on turn 1 too now (deciding to classify at all), not just post-classification
    assert calls == ["route", "classify", "route", "decide"]
    assert result["decision"]["action"] == "escalate_to_human"
    assert result["kb_results"] is None  # kb_retriever was never called


# --- test 2: side-effecting tool request suspends the graph; search_knowledge_base never does ---

def test_side_effecting_tool_request_suspends_graph(tmp_path):
    conn = _conn(tmp_path)

    def classify_fn(state):
        return VALID_CLASSIFICATION

    def decide_fn(state):
        return {"action": "escalate_to_human", "rationale": "needs approval", "confidence": 0.9,
                "cited_kb_doc_ids": [], "draft_reply": None,
                "pending_tool": {"tool_name": "issue_refund", "args": {"charge_ids": ["ch_1"], "amount_usd": 89.97}}}

    graph = build_graph(conn, InMemorySaver(), classify_fn=classify_fn, decide_fn=decide_fn,
                         route_fn=_route_classify_then("action_decider"))
    result = graph.invoke(_initial_state(), {"configurable": {"thread_id": "c_2"}})

    assert "__interrupt__" in result


def test_search_knowledge_base_never_suspends_the_graph(tmp_path):
    conn = _conn(tmp_path)

    def classify_fn(state):
        return VALID_FAQ_CLASSIFICATION

    def decide_fn(state):
        return {"action": "auto_respond", "rationale": "FAQ fully answers this", "confidence": 0.9,
                "cited_kb_doc_ids": ["faq_dark_mode"], "draft_reply": "Dark mode is on the roadmap...",
                "pending_tool": None}

    graph = build_graph(conn, InMemorySaver(), classify_fn=classify_fn, decide_fn=decide_fn,
                         route_fn=lambda state: {"next": "triage_classifier"} if state.get("extracted") is None
                         else {"next": "kb_retriever"} if state.get("kb_results") is None
                         else {"next": "action_decider"})
    result = graph.invoke(_initial_state(), {"configurable": {"thread_id": "c_3"}})

    assert "__interrupt__" not in result
    assert result["kb_results"] is not None
    assert result["decision"]["action"] == "auto_respond"


# --- test 3 & 4: resuming approve executes exactly once and reaches END; reject never executes ---

def _suspended_refund_graph(conn, thread_id):
    def classify_fn(state):
        return VALID_CLASSIFICATION

    def decide_fn(state):
        return {"action": "escalate_to_human", "rationale": "needs approval", "confidence": 0.9,
                "cited_kb_doc_ids": [], "draft_reply": None,
                "pending_tool": {"tool_name": "issue_refund", "args": {"charge_ids": ["ch_1"], "amount_usd": 89.97}}}

    graph = build_graph(conn, InMemorySaver(), classify_fn=classify_fn, decide_fn=decide_fn,
                         route_fn=_route_classify_then("action_decider"))
    graph.invoke(_initial_state(), {"configurable": {"thread_id": thread_id}})
    return graph


def test_resume_approve_executes_tool_exactly_once_and_reaches_end(tmp_path):
    conn = _conn(tmp_path)
    graph = _suspended_refund_graph(conn, "c_4")

    result = graph.invoke(Command(resume={"decision": "approve", "reason": "verified in Stripe"}),
                           {"configurable": {"thread_id": "c_4"}})

    assert "__interrupt__" not in result
    assert result["decision"]["tool_calls"][0]["status"] == "executed"
    assert result["decision"]["tool_calls"][0]["reason"] == "verified in Stripe"
    assert result["decision"]["tool_calls"][0]["result"]["status"] == "succeeded"


def test_resume_approve_appends_outcome_message_without_losing_prior_history(tmp_path):
    conn = _conn(tmp_path)
    graph = _suspended_refund_graph(conn, "c_4b")

    before = graph.get_state({"configurable": {"thread_id": "c_4b"}}).values
    original_message_count = len(before["messages"])

    result = graph.invoke(Command(resume={"decision": "approve", "reason": "verified"}),
                           {"configurable": {"thread_id": "c_4b"}})

    # prior messages (system + customer thread) must still be there, not wiped out by the append
    assert len(result["messages"]) == original_message_count + 1
    assert result["messages"][:original_message_count] == before["messages"]
    assert "approved" in result["messages"][-1]["content"]
    assert result["messages"][-1]["role"] == "agent"


def test_resume_reject_never_executes(tmp_path):
    conn = _conn(tmp_path)
    graph = _suspended_refund_graph(conn, "c_5")

    result = graph.invoke(Command(resume={"decision": "reject", "reason": "amount looks wrong"}),
                           {"configurable": {"thread_id": "c_5"}})

    assert result["decision"]["tool_calls"][0]["status"] == "rejected"
    assert result["decision"]["tool_calls"][0]["result"] is None

    stored = db.get_tool_call(conn, result["decision"]["tool_calls"][0]["tool_call_id"])
    assert stored["status"] == "rejected"


# --- test 5: context trim collapses the array correctly ---

def test_trimming_collapses_the_message_array():
    long_text = "x" * 4000  # ~1000 approx tokens per message at our 4-chars-per-token estimate
    messages = (
        [{"role": "system", "content": "system prompt"}]
        + [{"role": "customer", "content": long_text} for _ in range(5)]
    )
    assert len(messages) == 6

    trimmed = maybe_trim_messages(messages, summarize_fn=lambda msgs: "summary of " + str(len(msgs)) + " messages")

    assert trimmed[0] == messages[0]  # system message preserved, always first
    assert trimmed[1]["role"] == "history_summary"
    assert len(trimmed) < len(messages)


def test_trimming_is_a_no_op_under_threshold():
    messages = [{"role": "system", "content": "short"}, {"role": "customer", "content": "also short"}]
    trimmed = maybe_trim_messages(messages, summarize_fn=lambda msgs: "unused")
    assert trimmed == messages


# --- test 6: malformed specialist output retries once, then falls back to escalate_to_human ---

def test_decide_fn_raising_llm_unavailable_falls_back_instead_of_crashing(tmp_path):
    """Found live against the real API: a classify_fn/decide_fn that raises (not just
    returns malformed data -- e.g. LLMUnavailableError once call_with_retry's own
    budget is exhausted) used to propagate straight out of ingest_ticket() uncaught,
    instead of degrading to escalate_to_human like a schema-validation failure does."""
    from agents.llm import LLMUnavailableError

    conn = _conn(tmp_path)
    attempts = {"count": 0}

    def classify_fn(state):
        return VALID_CLASSIFICATION

    def decide_fn(state):
        attempts["count"] += 1
        raise LLMUnavailableError("simulated: retries exhausted")

    graph = build_graph(conn, InMemorySaver(), classify_fn=classify_fn, decide_fn=decide_fn,
                         route_fn=_route_classify_then("action_decider"))
    result = graph.invoke(_initial_state(), {"configurable": {"thread_id": "c_llm_down"}})

    assert attempts["count"] == 2  # 1 initial try + exactly 1 retry
    assert result["decision"]["action"] == "escalate_to_human"
    assert result["decision"]["rationale"] == "llm_unavailable"


def test_malformed_decision_output_retries_once_then_falls_back(tmp_path):
    conn = _conn(tmp_path)
    attempts = {"count": 0}

    def classify_fn(state):
        return VALID_CLASSIFICATION

    def decide_fn(state):
        attempts["count"] += 1
        return {"action": "not_a_real_action", "rationale": "x", "confidence": 0.9}  # always invalid

    graph = build_graph(conn, InMemorySaver(), classify_fn=classify_fn, decide_fn=decide_fn,
                         route_fn=_route_classify_then("action_decider"))
    result = graph.invoke(_initial_state(), {"configurable": {"thread_id": "c_6"}})

    assert attempts["count"] == 2  # 1 initial try + exactly 1 retry, not more
    assert result["decision"]["action"] == "escalate_to_human"
    assert result["decision"]["rationale"] == "schema_validation_failed"


def test_malformed_output_that_succeeds_on_retry_is_used(tmp_path):
    conn = _conn(tmp_path)
    attempts = {"count": 0}

    def classify_fn(state):
        return VALID_CLASSIFICATION

    def decide_fn(state):
        attempts["count"] += 1
        if attempts["count"] == 1:
            return {"action": "not_a_real_action", "rationale": "x", "confidence": 0.9}
        return {"action": "escalate_to_human", "rationale": "recovered on retry", "confidence": 0.8,
                "cited_kb_doc_ids": [], "draft_reply": None, "pending_tool": None}

    graph = build_graph(conn, InMemorySaver(), classify_fn=classify_fn, decide_fn=decide_fn,
                         route_fn=_route_classify_then("action_decider"))
    result = graph.invoke(_initial_state(), {"configurable": {"thread_id": "c_6b"}})

    assert attempts["count"] == 2
    assert result["decision"]["rationale"] == "recovered on retry"


# --- test 7: low confidence blocks auto_respond ---

def test_low_confidence_blocks_auto_respond(tmp_path):
    conn = _conn(tmp_path)

    def classify_fn(state):
        return VALID_FAQ_CLASSIFICATION

    def decide_fn(state):
        return {"action": "auto_respond", "rationale": "looks answerable", "confidence": 0.3,
                "cited_kb_doc_ids": ["faq_dark_mode"], "draft_reply": "here's the answer", "pending_tool": None}

    graph = build_graph(conn, InMemorySaver(), classify_fn=classify_fn, decide_fn=decide_fn,
                         route_fn=_route_classify_then("action_decider"))
    result = graph.invoke(_initial_state(), {"configurable": {"thread_id": "c_7"}})

    assert result["decision"]["action"] == "route_to_specialist"  # downgraded, not auto_respond
    assert result["decision"]["draft_reply"] is None


# --- test 8: MAX_STEPS halts an oscillating supervisor ---

def test_max_steps_halts_an_oscillating_supervisor(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "MAX_STEPS", 5)
    conn = _conn(tmp_path)

    def classify_fn(state):
        return VALID_CLASSIFICATION

    # never moves on to action_decider -- always sends it back to kb_retriever
    graph = build_graph(conn, InMemorySaver(), classify_fn=classify_fn,
                         route_fn=lambda state: {"next": "kb_retriever"})
    result = graph.invoke(_initial_state(), {"configurable": {"thread_id": "c_8"}})

    assert "__interrupt__" not in result
    assert result["decision"]["rationale"] == "max_steps_exceeded"
    assert result["step"] >= 5
