"""The supervisor/specialist LangGraph: routing, the await_tool_approval
autonomy-boundary gate, rolling-summary context trim, the MAX_STEPS loop
guard, and schema-validation retry on specialist output.

Every "brain" call (classify_fn, decide_fn, route_fn, summarize_fn) is
injectable. build_graph() defaults each to a real OpenAI call via
agents/llm.py when not given -- that's the production path, and it needs
OPENAI_API_KEY. Tests pass canned mock functions instead, so none of this
module's own tests make a network call.
"""
import hashlib
import json
import sqlite3
from typing import Any, Callable, Literal, Optional, TypedDict
from typing_extensions import Annotated
import operator

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, StateGraph
from langgraph.types import interrupt
from pydantic import BaseModel, Field, ValidationError

from agents import tools
from agents.llm import LLMUnavailableError, call_with_retry, get_agents, get_supervisor_prompt, make_llm_client
from api.schemas.decision import ExtractedInfo
from config import settings
from storage import db


# --- state ---

class GraphState(TypedDict):
    conversation_id: str
    messages: list[dict]  # plain (replace) field -- context trim needs wholesale replacement, not append
    urgency: Optional[str]
    extracted: Optional[dict]
    kb_results: Optional[list[dict]]
    decision: Optional[dict]
    pending_tool: Optional[dict]
    tool_calls: Annotated[list[dict], operator.add]
    step: int
    trace: Annotated[list[dict], operator.add]
    next_hop: Optional[str]  # set by supervisor() itself, read by the conditional edge -- route_fn's
                              # decision (including an optional direct_reply) is computed inside the node,
                              # since a conditional edge function can only return a string, not state updates


# --- internal validation schemas (what a specialist's raw LLM output must match) ---

class RawClassification(ExtractedInfo):
    urgency: Literal["critical", "high", "medium", "low"]


class RawDecision(BaseModel):
    action: Literal["auto_respond", "route_to_specialist", "escalate_to_human"]
    rationale: str
    confidence: float = Field(..., ge=0, le=1)
    cited_kb_doc_ids: list[str] = Field(default_factory=list)
    draft_reply: Optional[str] = None
    pending_tool: Optional[dict] = None  # {"tool_name": ..., "args": {...}}


class RouteDecision(BaseModel):
    next: Literal["triage_classifier", "kb_retriever", "action_decider", "direct_reply", "end"]
    reply: Optional[str] = Field(None, description="Only set when next='direct_reply' -- the answer to give "
                                                     "directly, with no classification or decision pipeline "
                                                     "run at all (e.g. 'who are you', 'what can you do').")


class ApprovalDecision(BaseModel):
    decision: Literal["approve", "reject"]
    reason: Optional[str] = None


_SCHEMA_FAILURE_DECISION = {
    "action": "escalate_to_human",
    "rationale": "schema_validation_failed",
    "confidence": 0.0,
    "cited_kb_doc_ids": [],
    "draft_reply": None,
}


def _trace(step: int, node: str, event_type: str, payload: Optional[dict] = None,
           model: Optional[str] = None, latency_ms: Optional[int] = None,
           tokens: Optional[dict] = None) -> dict:
    return {"step_index": step, "node": node, "event_type": event_type, "payload": payload,
            "model": model, "latency_ms": latency_ms, "tokens": tokens}


def _call_with_validation_retry(fn: Callable[[], dict], model_cls, *, max_retries: int = 1):
    """Calls fn(), validates the result against model_cls. Retries once on EITHER a
    validation failure (malformed-but-present output) OR fn() raising at all (e.g.
    LLMUnavailableError once call_with_retry's own budget is exhausted) -- either
    way, the caller falls back to escalate_to_human instead of crashing or guessing.

    Returns (validated_model_or_None, failure_reason_or_None) -- the reason lets the
    caller set an accurate decision rationale ("schema_validation_failed" vs
    "llm_unavailable") instead of a single hardcoded one regardless of what failed.
    """
    last_failure_reason = None
    for attempt in range(max_retries + 1):
        try:
            raw = fn()
        except LLMUnavailableError:
            last_failure_reason = "llm_unavailable"
            if attempt == max_retries:
                return None, last_failure_reason
            continue
        try:
            return model_cls.model_validate(raw), None
        except ValidationError:
            last_failure_reason = "schema_validation_failed"
            if attempt == max_retries:
                return None, last_failure_reason
    return None, last_failure_reason


# --- context engineering ---

def _approx_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def _total_tokens(messages: list[dict]) -> int:
    return sum(_approx_tokens(m.get("content", "")) for m in messages)


def _default_summarize_fn(messages_to_summarize: list[dict]) -> str:
    """Placeholder summarizer -- concatenates and truncates rather than calling
    an LLM. Swap for a real call_with_retry(...) + LLM summary once wired to
    OpenAI; the trim mechanics around it don't change either way."""
    joined = " / ".join(m["content"] for m in messages_to_summarize)
    return joined[:500]


def maybe_trim_messages(messages: list[dict], summarize_fn: Callable[[list[dict]], str]) -> list[dict]:
    if not messages or _total_tokens(messages) <= settings.CONTEXT_TOKEN_THRESHOLD:
        return messages

    system = messages[0]
    rest = messages[1:]
    has_summary = bool(rest) and rest[0].get("role") == "history_summary"
    old_summary = rest[0]["content"] if has_summary else None
    raw = rest[1:] if has_summary else rest

    kept: list[dict] = []
    kept_tokens = 0
    to_summarize: list[dict] = []
    for m in reversed(raw):
        t = _approx_tokens(m.get("content", ""))
        if kept_tokens + t <= settings.CONTEXT_KEEP_TOKENS:
            kept.insert(0, m)
            kept_tokens += t
        else:
            to_summarize.insert(0, m)

    if old_summary:
        to_summarize.insert(0, {"role": "history_summary", "content": old_summary})

    new_summary_text = summarize_fn(to_summarize) if to_summarize else (old_summary or "")

    new_messages = [system]
    if new_summary_text:
        new_messages.append({"role": "history_summary", "content": new_summary_text})
    new_messages.extend(kept)
    return new_messages


# --- default (real LLM) brain functions -- used when build_graph() isn't given mocks ---

def _default_classify_fn(state: GraphState) -> dict:
    # method="function_calling": OpenAI's default strict JSON-schema mode rejects any
    # free-form dict field (requires additionalProperties:false on every nested object,
    # including ones we don't control the shape of, like ToolCall.args) -- function-calling
    # mode doesn't have that restriction and is just as reliable for our schemas
    llm = make_llm_client(settings.OPENAI_MODEL).with_structured_output(RawClassification, method="function_calling")
    prompt = get_agents()["triage_classifier"].prompt
    thread_text = "\n".join(f"{m['role']}: {m['content']}" for m in state["messages"] if m["role"] != "history_summary")
    result = call_with_retry(lambda: llm.invoke([{"role": "system", "content": prompt},
                                                  {"role": "user", "content": thread_text}]))
    return result.model_dump()


def _default_decide_fn(state: GraphState) -> dict:
    llm = make_llm_client(settings.OPENAI_MODEL).with_structured_output(RawDecision, method="function_calling")
    prompt = get_agents()["action_decider"].prompt
    context = (f"extracted: {state.get('extracted')}\nkb_results: {state.get('kb_results')}\n"
               f"thread: {[m['content'] for m in state['messages'] if m['role'] != 'history_summary']}")
    result = call_with_retry(lambda: llm.invoke([{"role": "system", "content": prompt},
                                                  {"role": "user", "content": context}]))
    return result.model_dump()


def _default_route_fn(state: GraphState) -> dict:
    llm = make_llm_client(settings.SUPERVISOR_MODEL).with_structured_output(RouteDecision, method="function_calling")
    prompt = get_supervisor_prompt()
    thread = "\n".join(f"{m['role']}: {m['content']}" for m in state["messages"] if m["role"] != "history_summary")
    context = (f"thread so far:\n{thread}\n\n"
               f"extracted: {state.get('extracted')}\nkb_results: {state.get('kb_results')}\n"
               f"decision: {state.get('decision')}")
    result = call_with_retry(lambda: llm.invoke([{"role": "system", "content": prompt},
                                                  {"role": "user", "content": context}]))
    return result.model_dump()


# --- graph factory ---

def build_graph(
    conn: sqlite3.Connection,
    checkpointer: BaseCheckpointSaver,
    classify_fn: Optional[Callable[[GraphState], dict]] = None,
    decide_fn: Optional[Callable[[GraphState], dict]] = None,
    route_fn: Optional[Callable[[GraphState], str]] = None,
    summarize_fn: Optional[Callable[[list[dict]], str]] = None,
):
    classify_fn = classify_fn or _default_classify_fn
    decide_fn = decide_fn or _default_decide_fn
    route_fn = route_fn or _default_route_fn
    summarize_fn = summarize_fn or _default_summarize_fn

    def supervisor(state: GraphState) -> dict:
        step = state["step"] + 1
        trimmed = maybe_trim_messages(state["messages"], summarize_fn)
        trim_happened = len(trimmed) != len(state["messages"])
        trace_entries = [_trace(step, "supervisor", "turn_start")]
        updates: dict[str, Any] = {"step": step, "trace": trace_entries}
        if trim_happened:
            updates["messages"] = trimmed  # messages is a plain field: this replaces, not appends
            trace_entries.append(_trace(step, "supervisor", "context_trimmed", payload={"new_length": len(trimmed)}))

        if step >= settings.MAX_STEPS:
            updates["next_hop"] = "force_escalate"
            return updates

        if state.get("decision") is not None:
            # already fully decided this turn (e.g. resumed from await_tool_approval) -- nothing left to route
            updates["next_hop"] = "end"
            return updates

        # the supervisor decides EVERY turn, including the very first one -- it is not forced into
        # triage_classifier just because nothing is classified yet. A plain "who are you" should never
        # get mis-classified as if it were a real ticket; route_fn (the LLM, guided by _supervisor.md)
        # makes that call itself, same as it decides whether to skip kb_retriever for a live incident.
        route_view = {**state, **updates}
        raw = route_fn(route_view)
        try:
            routed = raw if isinstance(raw, RouteDecision) else RouteDecision.model_validate(raw)
        except ValidationError:
            # a routing failure is not a reason to give up on the ticket -- fall back to the one safe
            # default: classify if not classified yet, otherwise hand off to the decider
            routed = RouteDecision(next="triage_classifier" if state.get("extracted") is None else "action_decider")

        updates["next_hop"] = routed.next
        trace_entries.append(_trace(step, "supervisor", "handoff", payload={"to": routed.next}))

        if routed.next == "direct_reply" and routed.reply:
            updates["messages"] = updates.get("messages", state["messages"]) + [{"role": "agent", "content": routed.reply}]
            trace_entries.append(_trace(step, "supervisor", "direct_reply", payload={"reply": routed.reply}))
            # represented as a lightweight auto_respond, not a schema carve-out -- every ticket's
            # decision has the same shape (Decision requires urgency/extracted/action/rationale),
            # so a greeting or meta-question doesn't need a special-cased response contract
            updates["decision"] = {
                "urgency": "low",
                "extracted": {"product_area": "general", "issue_type": "general_inquiry", "sentiment": "neutral",
                              "language": "en", "confidence": 1.0, "rationale": "not a support ticket"},
                "action": "auto_respond",
                "draft_reply": routed.reply,
                "rationale": "direct conversational reply -- not a ticket requiring triage",
                "confidence": 1.0,
                "cited_kb_doc_ids": [],
            }

        return updates

    def _route_from_supervisor(state: GraphState) -> str:
        return state["next_hop"]

    def call_triage_classifier(state: GraphState) -> dict:
        step = state["step"] + 1
        validated, failure_reason = _call_with_validation_retry(lambda: classify_fn(state), RawClassification)
        if validated is None:
            return {
                "step": step,
                "decision": {**_SCHEMA_FAILURE_DECISION, "rationale": failure_reason},
                "trace": [_trace(step, "call_triage_classifier", failure_reason)],
            }
        extracted = validated.model_dump(exclude={"urgency"})
        # a short status message, not the full rationale -- kept deliberately brief so it
        # doesn't bloat every subsequent LLM call's context the way the full reasoning would
        status_message = {
            "role": "agent",
            "content": f"[triage_classifier] urgency={validated.urgency}, product_area={extracted['product_area']}, "
                       f"issue_type={extracted['issue_type']}, confidence={extracted['confidence']}",
        }
        return {
            "step": step,
            "extracted": extracted,
            "urgency": validated.urgency,
            "messages": state["messages"] + [status_message],
            "trace": [_trace(step, "call_triage_classifier", "decision",
                              payload={"urgency": validated.urgency, **extracted})],
        }

    def call_kb_retriever(state: GraphState) -> dict:
        step = state["step"] + 1
        query = state.get("extracted", {}).get("issue_type", "") if state.get("extracted") else ""
        results = tools.search_knowledge_base(query or "general question")
        return {
            "step": step,
            "kb_results": results,
            "trace": [_trace(step, "call_kb_retriever", "tool_call",
                              payload={"tool_name": "search_knowledge_base", "query": query, "results": results})],
        }

    def call_action_decider(state: GraphState) -> dict:
        step = state["step"] + 1
        validated, failure_reason = _call_with_validation_retry(lambda: decide_fn(state), RawDecision)
        if validated is None:
            return {
                "step": step,
                "decision": {**_SCHEMA_FAILURE_DECISION, "rationale": failure_reason},
                "trace": [_trace(step, "call_action_decider", failure_reason)],
            }

        decision = validated.model_dump(exclude={"pending_tool"})
        decision["urgency"] = state.get("urgency") or "medium"
        decision["extracted"] = state.get("extracted")

        # confidence gate: auto_respond is not allowed below the configured threshold
        if decision["action"] == "auto_respond" and decision["confidence"] < settings.AUTO_RESPOND_CONFIDENCE_THRESHOLD:
            decision["action"] = "route_to_specialist"
            decision["draft_reply"] = None

        updates: dict[str, Any] = {
            "step": step,
            "decision": decision,
            "trace": [_trace(step, "call_action_decider", "decision", payload=decision)],
        }
        if validated.pending_tool and validated.pending_tool.get("tool_name") in tools.SIDE_EFFECTING_TOOL_IMPLS:
            updates["pending_tool"] = validated.pending_tool
        return updates

    def force_escalate(state: GraphState) -> dict:
        step = state["step"] + 1
        decision = {**_SCHEMA_FAILURE_DECISION, "rationale": "max_steps_exceeded"}
        return {"step": step, "decision": decision,
                "trace": [_trace(step, "force_escalate", "loop_guard_triggered")]}

    def await_tool_approval(state: GraphState) -> dict:
        step = state["step"] + 1
        tool = state["pending_tool"]
        # sha256, not Python's hash() -- hash() is salted per-process (PYTHONHASHSEED), so it would
        # mint a DIFFERENT tool_call_id after a server restart, breaking resume. This must be the
        # same id before and after suspending, including across a real process restart.
        arg_digest = hashlib.sha256(
            f"{state['conversation_id']}:{tool['tool_name']}:{json.dumps(tool['args'], sort_keys=True)}".encode()
        ).hexdigest()[:10]
        tool_call_id = f"tc_{arg_digest}"
        idempotency_key = f"{state['conversation_id']}:{tool['tool_name']}:{arg_digest}"

        existing = db.get_tool_call(conn, tool_call_id)
        if existing is None:
            db.insert_tool_call(conn, tool_call_id, state["conversation_id"], tool["tool_name"],
                                 tool["args"], "pending_approval", idempotency_key=idempotency_key)

        raw = interrupt({"tool_call_id": tool_call_id, "tool_name": tool["tool_name"], "args": tool["args"]},
                         response_schema=ApprovalDecision)
        human_decision = raw if isinstance(raw, ApprovalDecision) else ApprovalDecision.model_validate(raw)

        if human_decision.decision == "approve":
            try:
                result = tools.dispatch_tool_call(conn, tool_call_id, reason=human_decision.reason)
                status = "executed"
            except tools.ToolExecutionError:
                result = None
                status = "rejected"
        else:
            db.update_tool_call(conn, tool_call_id, "rejected", reason=human_decision.reason)
            result = None
            status = "rejected"

        tool_call_record = {"tool_call_id": tool_call_id, "tool_name": tool["tool_name"],
                             "status": status, "reason": human_decision.reason, "result": result}

        decision = dict(state["decision"] or {})
        decision["tool_calls"] = [tool_call_record]

        if status == "executed":
            outcome_text = f"[System] {tool['tool_name']} ({tool_call_id}) approved"
        else:
            outcome_text = f"[System] {tool['tool_name']} ({tool_call_id}) rejected"
        if human_decision.reason:
            outcome_text += f" ({human_decision.reason})"
        outcome_text += f": {result}" if result is not None else " — not executed"
        outcome_message = {"role": "agent", "content": outcome_text}

        return {
            "step": step,
            "decision": decision,
            "tool_calls": [tool_call_record],
            # messages is a plain field -- must append explicitly, a bare [outcome_message]
            # here would wipe out the rest of the thread instead of adding to it
            "messages": state["messages"] + [outcome_message],
            "trace": [_trace(step, "await_tool_approval", "resumed", payload=tool_call_record)],
        }

    def _route_from_action_decider(state: GraphState) -> str:
        if state.get("pending_tool"):
            return "await_tool_approval"
        return "end"

    graph = StateGraph(GraphState)
    graph.add_node("supervisor", supervisor)
    graph.add_node("triage_classifier", call_triage_classifier)
    graph.add_node("kb_retriever", call_kb_retriever)
    graph.add_node("action_decider", call_action_decider)
    graph.add_node("force_escalate", force_escalate)
    graph.add_node("await_tool_approval", await_tool_approval)

    graph.set_entry_point("supervisor")
    graph.add_conditional_edges("supervisor", _route_from_supervisor, {
        "triage_classifier": "triage_classifier",
        "kb_retriever": "kb_retriever",
        "action_decider": "action_decider",
        "force_escalate": "force_escalate",
        "direct_reply": END,  # reply was already appended to messages inside supervisor() itself
        "end": END,
    })
    graph.add_edge("triage_classifier", "supervisor")
    graph.add_edge("kb_retriever", "supervisor")
    graph.add_conditional_edges("action_decider", _route_from_action_decider, {
        "await_tool_approval": "await_tool_approval",
        "end": END,
    })
    graph.add_edge("force_escalate", END)
    graph.add_edge("await_tool_approval", END)

    return graph.compile(checkpointer=checkpointer)
