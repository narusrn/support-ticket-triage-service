"""Tests for the ReAct/handoff supervisor (agents/orchestrator.py). LLM calls are
mocked at the brain-function boundary -- classify_fn/decide_fn are injectable, and
supervisor_llm is a fake object with .invoke() returning something tool_calls/.content
-shaped, exactly like a real LangChain AIMessage -- so no network call, no API key
needed, for any test in this file.
"""
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from agents.orchestrator import GraphState, _REACT_LANGUAGE_RULE, _language_rule, build_graph, maybe_trim_messages
from agents.llm import LLMUnavailableError
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
        "messages": [{"role": "customer", "content": "my payment failed, now I see three charges"}],
        "urgency": None, "extracted": None, "kb_results": None, "decision": None,
        "pending_tool": None, "tool_calls": [], "step": 0, "trace": [],
    }


VALID_CLASSIFICATION = {
    "product_area": "billing", "issue_type": "duplicate_charge", "sentiment": "angry",
    "language": "en", "confidence": 0.9, "rationale": "3 charges in 2h", "urgency": "critical",
}

VALID_FAQ_CLASSIFICATION = {
    "product_area": "ui/appearance", "issue_type": "feature_question", "sentiment": "curious",
    "language": "en", "confidence": 0.85, "rationale": "asking about dark mode", "urgency": "low",
}


class _FakeResponse:
    """Stands in for a LangChain AIMessage: a real tool-calling response's .tool_calls
    is a list of {"name", "args", "id"} dicts -- a real bind_tools() response works
    the same way."""
    def __init__(self, tool_calls=None, content=""):
        self.tool_calls = tool_calls or []
        self.content = content


def _handoff_call(next_: str, reason: str = "test", reply=None):
    args = {"next": next_, "reason": reason}
    if reply is not None:
        args["reply"] = reply
    return {"name": "HandoffTool", "args": args, "id": "call_1"}


class _ScriptedLLM:
    """Returns one canned response per supervisor turn, in order."""
    def __init__(self, responses):
        self._responses = list(responses)

    def invoke(self, messages):
        return self._responses.pop(0)


class _RepeatingLLM:
    """Returns the SAME canned response every turn -- for simulating a supervisor
    that never progresses (the MAX_STEPS test)."""
    def __init__(self, response):
        self._response = response

    def invoke(self, messages):
        return self._response


# --- routing order, in-order handoffs, skipping kb_retriever for a live incident ---

def test_supervisor_classifies_then_escalates(tmp_path):
    conn = _conn(tmp_path)
    classify_calls = []

    def classify_fn(state, messages):
        classify_calls.append(1)
        return VALID_CLASSIFICATION

    def decide_fn(state, messages):
        return {"action": "escalate_to_human", "rationale": "needs a human", "confidence": 0.9,
                "cited_kb_doc_ids": [], "draft_reply": None, "pending_tool": None}

    llm = _ScriptedLLM([
        _FakeResponse([_handoff_call("triage_classifier")]),
        _FakeResponse([_handoff_call("action_decider")]),
    ])
    graph = build_graph(conn, InMemorySaver(), classify_fn=classify_fn, decide_fn=decide_fn, supervisor_llm=llm)
    result = graph.invoke(_initial_state(), {"configurable": {"thread_id": "c_1"}})

    assert classify_calls == [1]
    assert result["decision"]["action"] == "escalate_to_human"
    assert any(m["content"].startswith("Triage Classifier:") for m in result["messages"])
    assert any(m["content"].startswith("Supervisor: This ticket's been decided") for m in result["messages"])


def test_search_knowledge_base_never_suspends_the_graph(tmp_path):
    conn = _conn(tmp_path)

    def classify_fn(state, messages):
        return VALID_FAQ_CLASSIFICATION

    def decide_fn(state, messages):
        return {"action": "auto_respond", "rationale": "FAQ fully answers this", "confidence": 0.9,
                "cited_kb_doc_ids": ["faq_dark_mode"], "draft_reply": "Dark mode is on the roadmap...",
                "pending_tool": None}

    llm = _ScriptedLLM([
        _FakeResponse([_handoff_call("triage_classifier")]),
        _FakeResponse([_handoff_call("kb_retriever")]),
        _FakeResponse([_handoff_call("action_decider")]),
    ])
    graph = build_graph(conn, InMemorySaver(), classify_fn=classify_fn, decide_fn=decide_fn, supervisor_llm=llm)
    result = graph.invoke(_initial_state(), {"configurable": {"thread_id": "c_3"}})

    assert "__interrupt__" not in result
    assert result["kb_results"] is not None
    assert result["decision"]["action"] == "auto_respond"
    assert any(m["content"].startswith("KB Retriever:") for m in result["messages"])


def test_kb_retriever_reports_back_so_supervisor_does_not_loop_on_it(tmp_path):
    """Found live (ticket_3_dark_mode against the real model): without a status message
    from kb_retriever, every subsequent supervisor turn looked identical to the first
    ("still an unanswered FAQ question"), so it kept re-handing-off to kb_retriever
    until MAX_STEPS forced an escalation, instead of moving on to action_decider."""
    conn = _conn(tmp_path)

    def classify_fn(state, messages):
        return VALID_FAQ_CLASSIFICATION

    def decide_fn(state, messages):
        return {"action": "auto_respond", "rationale": "FAQ answers this", "confidence": 0.9,
                "cited_kb_doc_ids": ["faq_1"], "draft_reply": "here's the answer", "pending_tool": None}

    llm = _ScriptedLLM([
        _FakeResponse([_handoff_call("triage_classifier")]),
        _FakeResponse([_handoff_call("kb_retriever")]),
        _FakeResponse([_handoff_call("action_decider")]),
    ])
    graph = build_graph(conn, InMemorySaver(), classify_fn=classify_fn, decide_fn=decide_fn, supervisor_llm=llm)
    result = graph.invoke(_initial_state(), {"configurable": {"thread_id": "c_kb"}})

    status_msgs = [m for m in result["messages"] if m["content"].startswith("KB Retriever:")]
    assert len(status_msgs) == 1
    assert result["decision"]["action"] == "auto_respond"


# --- the bug this architecture is known for: >1 tool call in one supervisor turn ---

def test_supervisor_drops_every_tool_call_but_the_first_and_logs_it(tmp_path):
    """The exact failure mode from the ORIGINAL build that made a ReAct supervisor risky
    in the first place: a model emitting two tool calls in the same turn, racing on the
    shared state dict. _single_tool_call() keeps only the first and records the drop as
    a trace event -- this is a real code-level guard, not just a prompt instruction."""
    conn = _conn(tmp_path)

    def classify_fn(state, messages):
        return VALID_CLASSIFICATION

    def decide_fn(state, messages):
        return {"action": "escalate_to_human", "rationale": "x", "confidence": 0.9,
                "cited_kb_doc_ids": [], "draft_reply": None, "pending_tool": None}

    llm = _ScriptedLLM([
        _FakeResponse([_handoff_call("triage_classifier"), _handoff_call("kb_retriever")]),
        _FakeResponse([_handoff_call("action_decider")]),
    ])
    graph = build_graph(conn, InMemorySaver(), classify_fn=classify_fn, decide_fn=decide_fn, supervisor_llm=llm)
    result = graph.invoke(_initial_state(), {"configurable": {"thread_id": "c_2"}})

    assert result["kb_results"] is None  # kb_retriever was never entered -- its call was dropped
    drop_events = [t for t in result["trace"] if t["event_type"] == "multiple_tool_calls_dropped"]
    assert len(drop_events) == 1
    assert drop_events[0]["payload"]["kept"] == "HandoffTool"
    assert drop_events[0]["payload"]["dropped"] == ["HandoffTool"]


# --- direct_reply: both the explicit tool-call path and the no-tool-call fallback ---

def test_direct_reply_via_explicit_handoff_call_never_enters_triage_classifier(tmp_path):
    conn = _conn(tmp_path)
    classify_calls = []

    def classify_fn(state, messages):
        classify_calls.append(1)
        return VALID_CLASSIFICATION

    llm = _ScriptedLLM([_FakeResponse([_handoff_call(
        "direct_reply", reply="I'm a support ticket triage assistant.")])])
    graph = build_graph(conn, InMemorySaver(), classify_fn=classify_fn, supervisor_llm=llm)

    state = _initial_state()
    state["messages"] = [{"role": "customer", "content": "who are you, what can you do?"}]
    result = graph.invoke(state, {"configurable": {"thread_id": "c_1"}})

    assert classify_calls == []  # triage_classifier was never entered
    assert "__interrupt__" not in result
    assert result["decision"]["action"] == "auto_respond"
    assert "triage assistant" in result["decision"]["draft_reply"]


def test_direct_reply_with_no_tool_call_at_all(tmp_path):
    conn = _conn(tmp_path)
    llm = _ScriptedLLM([_FakeResponse([], content="I'm a support triage assistant.")])
    graph = build_graph(conn, InMemorySaver(), supervisor_llm=llm)

    state = _initial_state()
    state["messages"] = [{"role": "customer", "content": "who are you?"}]
    result = graph.invoke(state, {"configurable": {"thread_id": "c_3"}})

    assert result["decision"]["action"] == "auto_respond"
    assert result["decision"]["draft_reply"] == "I'm a support triage assistant."


def test_narrated_handoff_with_no_tool_call_mid_pipeline_is_not_trusted_as_a_reply(tmp_path):
    """Found live (ticket_3_dark_mode against the real model): mid-pipeline, the model
    sometimes narrates its next move in plain content ("I'm now handing this to Action
    Decider...") instead of actually calling HandoffTool. A no-tool-call response is
    only a legitimate direct_reply when nothing is classified yet -- once extracted is
    set, it must fall back to the safe default instead of leaking that narration into
    draft_reply."""
    conn = _conn(tmp_path)

    def classify_fn(state, messages):
        return VALID_CLASSIFICATION

    def decide_fn(state, messages):
        return {"action": "escalate_to_human", "rationale": "x", "confidence": 0.9,
                "cited_kb_doc_ids": [], "draft_reply": None, "pending_tool": None}

    llm = _ScriptedLLM([
        _FakeResponse([_handoff_call("triage_classifier")]),
        _FakeResponse([], content="I'm now handing this to Action Decider -- let's see what we should do."),
    ])
    graph = build_graph(conn, InMemorySaver(), classify_fn=classify_fn, decide_fn=decide_fn, supervisor_llm=llm)
    result = graph.invoke(_initial_state(), {"configurable": {"thread_id": "c_narrated"}})

    assert result["decision"]["action"] == "escalate_to_human"  # from decide_fn, not treated as a direct_reply
    assert "Action Decider" not in (result["decision"].get("draft_reply") or "")


def test_handoff_tool_call_missing_reason_still_keeps_the_models_real_choice(tmp_path):
    """Found live (edge_case_greeting against the real model): the tool call omitted
    "reason" entirely. reason was a required field with no default, so that call failed
    model_validate() -- and the except-branch discarded the model's actual
    next="direct_reply" choice, routing a plain greeting into triage_classifier instead
    of answering it directly. reason now defaults instead of being required."""
    conn = _conn(tmp_path)
    classify_calls = []

    def classify_fn(state, messages):
        classify_calls.append(1)
        return VALID_CLASSIFICATION

    call_missing_reason = {"name": "HandoffTool",
                            "args": {"next": "direct_reply", "reply": "I'm a support triage assistant."},
                            "id": "call_1"}
    llm = _ScriptedLLM([_FakeResponse([call_missing_reason])])
    graph = build_graph(conn, InMemorySaver(), classify_fn=classify_fn, supervisor_llm=llm)
    result = graph.invoke(_initial_state(), {"configurable": {"thread_id": "c_missing_reason"}})

    assert classify_calls == []
    assert result["decision"]["action"] == "auto_respond"
    assert result["decision"]["draft_reply"] == "I'm a support triage assistant."


def test_supervisor_cannot_skip_classification(tmp_path):
    """Found live against the real model: the supervisor sometimes tries to jump
    straight to action_decider on a real ticket, leaving extracted=None. action_decider
    would then produce a decision missing a REQUIRED Decision.extracted field. The
    supervisor must override this regardless of what the model's tool call says."""
    conn = _conn(tmp_path)
    classify_calls = []

    def classify_fn(state, messages):
        classify_calls.append(1)
        return VALID_CLASSIFICATION

    def decide_fn(state, messages):
        return {"action": "escalate_to_human", "rationale": "x", "confidence": 0.9,
                "cited_kb_doc_ids": [], "draft_reply": None, "pending_tool": None}

    # model misbehaves on purpose -- always tries to jump straight to action_decider
    llm = _ScriptedLLM([
        _FakeResponse([_handoff_call("action_decider", reason="trying to skip classification")]),
        _FakeResponse([_handoff_call("action_decider")]),
    ])
    graph = build_graph(conn, InMemorySaver(), classify_fn=classify_fn, decide_fn=decide_fn, supervisor_llm=llm)
    result = graph.invoke(_initial_state(), {"configurable": {"thread_id": "c_5"}})

    assert classify_calls == [1]
    assert result["extracted"] is not None


# --- side-effecting tool request suspends the graph; approve/reject resolve it ---

def _suspended_refund_graph(conn, thread_id):
    def classify_fn(state, messages):
        return VALID_CLASSIFICATION

    def decide_fn(state, messages):
        return {"action": "escalate_to_human", "rationale": "needs approval", "confidence": 0.9,
                "cited_kb_doc_ids": [], "draft_reply": None,
                "pending_tool": {"tool_name": "issue_refund", "args": {"charge_ids": ["ch_1"], "amount_usd": 89.97}}}

    llm = _ScriptedLLM([
        _FakeResponse([_handoff_call("triage_classifier")]),
        _FakeResponse([_handoff_call("action_decider")]),
    ])
    graph = build_graph(conn, InMemorySaver(), classify_fn=classify_fn, decide_fn=decide_fn, supervisor_llm=llm)
    graph.invoke(_initial_state(), {"configurable": {"thread_id": thread_id}})
    return graph


def test_side_effecting_tool_request_suspends_graph(tmp_path):
    conn = _conn(tmp_path)
    graph = _suspended_refund_graph(conn, "c_4")
    state = graph.get_state({"configurable": {"thread_id": "c_4"}}).values
    assert state.get("pending_tool") is not None


def test_resume_approve_executes_tool_exactly_once_and_reaches_end(tmp_path):
    conn = _conn(tmp_path)
    graph = _suspended_refund_graph(conn, "c_4")

    result = graph.invoke(Command(resume={"decision": "approve", "reason": "verified in Stripe"}),
                           {"configurable": {"thread_id": "c_4"}})

    assert "__interrupt__" not in result
    assert result["decision"]["tool_calls"][0]["status"] == "executed"
    assert result["decision"]["tool_calls"][0]["reason"] == "verified in Stripe"
    assert result["decision"]["tool_calls"][0]["result"]["status"] == "succeeded"
    # the "every specialist reports back to the supervisor" invariant: await_tool_approval
    # routes back to supervisor, which appends a closing line before reaching END
    assert any(m["content"].startswith("Supervisor: This ticket's been decided") for m in result["messages"])


def test_resume_approve_appends_outcome_message_without_losing_prior_history(tmp_path):
    conn = _conn(tmp_path)
    graph = _suspended_refund_graph(conn, "c_4b")

    before = graph.get_state({"configurable": {"thread_id": "c_4b"}}).values
    original_message_count = len(before["messages"])

    result = graph.invoke(Command(resume={"decision": "approve", "reason": "verified"}),
                           {"configurable": {"thread_id": "c_4b"}})

    # prior messages (system + customer thread) must still be there, not wiped out by the append
    assert len(result["messages"]) > original_message_count
    assert result["messages"][:original_message_count] == before["messages"]
    assert "approved" in result["messages"][original_message_count]["content"]


def test_resume_reject_never_executes(tmp_path):
    conn = _conn(tmp_path)
    graph = _suspended_refund_graph(conn, "c_5")

    result = graph.invoke(Command(resume={"decision": "reject", "reason": "amount looks wrong"}),
                           {"configurable": {"thread_id": "c_5"}})

    assert result["decision"]["tool_calls"][0]["status"] == "rejected"
    assert result["decision"]["tool_calls"][0]["result"] is None

    stored = db.get_tool_call(conn, result["decision"]["tool_calls"][0]["tool_call_id"])
    assert stored["status"] == "rejected"


# --- context trim collapses the array correctly (pure function, no graph involved) ---

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


# --- malformed/missing specialist output retries once, then falls back ---

def test_decide_fn_raising_llm_unavailable_falls_back_instead_of_crashing(tmp_path):
    """Found live against the real API: a classify_fn/decide_fn that raises (not just
    returns malformed data -- e.g. LLMUnavailableError once call_with_retry's own
    budget is exhausted) used to propagate straight out of ingest_ticket() uncaught,
    instead of degrading to escalate_to_human like a schema-validation failure does."""
    conn = _conn(tmp_path)
    attempts = {"count": 0}

    def classify_fn(state, messages):
        return VALID_CLASSIFICATION

    def decide_fn(state, messages):
        attempts["count"] += 1
        raise LLMUnavailableError("simulated: retries exhausted")

    llm = _ScriptedLLM([
        _FakeResponse([_handoff_call("triage_classifier")]),
        _FakeResponse([_handoff_call("action_decider")]),
    ])
    graph = build_graph(conn, InMemorySaver(), classify_fn=classify_fn, decide_fn=decide_fn, supervisor_llm=llm)
    result = graph.invoke(_initial_state(), {"configurable": {"thread_id": "c_llm_down"}})

    assert attempts["count"] == 2  # 1 initial try + exactly 1 retry
    assert result["decision"]["action"] == "escalate_to_human"
    assert result["decision"]["rationale"] == "llm_unavailable"


def test_malformed_decision_output_retries_once_then_falls_back(tmp_path):
    conn = _conn(tmp_path)
    attempts = {"count": 0}

    def classify_fn(state, messages):
        return VALID_CLASSIFICATION

    def decide_fn(state, messages):
        attempts["count"] += 1
        return {"action": "not_a_real_action", "rationale": "x", "confidence": 0.9}  # always invalid

    llm = _ScriptedLLM([
        _FakeResponse([_handoff_call("triage_classifier")]),
        _FakeResponse([_handoff_call("action_decider")]),
    ])
    graph = build_graph(conn, InMemorySaver(), classify_fn=classify_fn, decide_fn=decide_fn, supervisor_llm=llm)
    result = graph.invoke(_initial_state(), {"configurable": {"thread_id": "c_6"}})

    assert attempts["count"] == 2  # 1 initial try + exactly 1 retry, not more
    assert result["decision"]["action"] == "escalate_to_human"
    assert result["decision"]["rationale"] == "schema_validation_failed"


def test_malformed_output_that_succeeds_on_retry_is_used(tmp_path):
    conn = _conn(tmp_path)
    attempts = {"count": 0}

    def classify_fn(state, messages):
        return VALID_CLASSIFICATION

    def decide_fn(state, messages):
        attempts["count"] += 1
        if attempts["count"] == 1:
            return {"action": "not_a_real_action", "rationale": "x", "confidence": 0.9}
        return {"action": "escalate_to_human", "rationale": "recovered on retry", "confidence": 0.8,
                "cited_kb_doc_ids": [], "draft_reply": None, "pending_tool": None}

    llm = _ScriptedLLM([
        _FakeResponse([_handoff_call("triage_classifier")]),
        _FakeResponse([_handoff_call("action_decider")]),
    ])
    graph = build_graph(conn, InMemorySaver(), classify_fn=classify_fn, decide_fn=decide_fn, supervisor_llm=llm)
    result = graph.invoke(_initial_state(), {"configurable": {"thread_id": "c_6b"}})

    assert attempts["count"] == 2
    assert result["decision"]["rationale"] == "recovered on retry"


# --- low confidence blocks auto_respond ---

def test_low_confidence_blocks_auto_respond(tmp_path):
    conn = _conn(tmp_path)

    def classify_fn(state, messages):
        return VALID_FAQ_CLASSIFICATION

    def decide_fn(state, messages):
        return {"action": "auto_respond", "rationale": "looks answerable", "confidence": 0.3,
                "cited_kb_doc_ids": ["faq_dark_mode"], "draft_reply": "here's the answer", "pending_tool": None}

    llm = _ScriptedLLM([
        _FakeResponse([_handoff_call("triage_classifier")]),
        _FakeResponse([_handoff_call("action_decider")]),
    ])
    graph = build_graph(conn, InMemorySaver(), classify_fn=classify_fn, decide_fn=decide_fn, supervisor_llm=llm)
    result = graph.invoke(_initial_state(), {"configurable": {"thread_id": "c_7"}})

    assert result["decision"]["action"] == "route_to_specialist"  # downgraded, not auto_respond
    assert result["decision"]["draft_reply"] is None


# --- MAX_STEPS halts an oscillating supervisor ---

def test_max_steps_halts_an_oscillating_supervisor(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "MAX_STEPS", 5)
    conn = _conn(tmp_path)

    def classify_fn(state, messages):
        return VALID_CLASSIFICATION

    # never moves on to action_decider -- always wants kb_retriever, simulating an
    # oscillating supervisor that never converges
    llm = _RepeatingLLM(_FakeResponse([_handoff_call("kb_retriever", reason="stuck on purpose")]))
    graph = build_graph(conn, InMemorySaver(), classify_fn=classify_fn, supervisor_llm=llm)
    result = graph.invoke(_initial_state(), {"configurable": {"thread_id": "c_8"}})

    assert "__interrupt__" not in result
    assert result["decision"]["rationale"] == "max_steps_exceeded"
    assert result["step"] >= 5


# --- language-matching: explicit once known, inferred only when it has to be ---

def test_language_rule_is_explicit_once_language_is_known(tmp_path):
    """Found live: with only a generic "same language as the customer" instruction,
    action_decider's rationale came out in Spanish on an all-English ticket, once the
    context had enough OTHER English text in it (prior dialogue, the prompt itself) to
    make "the customer's language" ambiguous to infer. Once triage_classifier has run,
    the language is a known fact -- state the ISO code explicitly instead of asking
    the model to re-derive it from a noisier context."""
    assert _language_rule({"extracted": None}) == _REACT_LANGUAGE_RULE  # nothing classified yet -- must infer
    rule = _language_rule({"extracted": {"language": "th"}})
    assert "'th'" in rule
    assert rule != _REACT_LANGUAGE_RULE
