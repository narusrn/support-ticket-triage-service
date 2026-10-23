"""The supervisor/specialist LangGraph: a ReAct-style, tool-calling supervisor
(HandoffTool + Command(goto=...)) that routes to triage_classifier/kb_retriever/
action_decider, the await_tool_approval autonomy-boundary gate, rolling-summary
context trim, the MAX_STEPS loop guard, and schema-validation retry on specialist
output.

This replaces an earlier structured-output router (the supervisor produced a
RouteDecision object instead of calling a tool). That design was dropped after
live testing against the real model turned up four real bugs, each fixed here:
a kb_retriever loop (it never reported back to the supervisor, so every turn
looked identical to the first), a "narrated handoff" (the model sometimes
described its next move in plain text instead of calling the tool, which would
have leaked straight into draft_reply), a required `reason` field with no
default crashing validation and discarding the model's real routing choice, and
action_decider/await_tool_approval never reporting back to the supervisor at all
-- the only two nodes in the whole graph that didn't, breaking the invariant
every other node follows. `_single_tool_call()` guards against a fifth risk
that was caught during the ORIGINAL build, before any of the above: a ReAct
supervisor can legally emit more than one tool call in a single turn, and when
it did, two tool results got written into the same shared state dict and one
silently clobbered the other.

Every "brain" call (classify_fn, decide_fn, supervisor_llm, summarize_fn) is
injectable. build_graph() defaults each to a real OpenAI call via agents/llm.py
when not given -- that's the production path, and it needs OPENAI_API_KEY.
Tests pass canned mock functions instead, so none of this module's own tests
make a network call.
"""
import hashlib
import json
import sqlite3
from typing import Any, Callable, Literal, Optional, TypedDict
from typing_extensions import Annotated
import operator

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, StateGraph
from langgraph.types import Command, interrupt
from pydantic import BaseModel, Field, ValidationError

from agents import retriever, tools
from agents.llm import LLMUnavailableError, call_with_retry, extract_usage, get_agents, get_supervisor_prompt, make_llm_client
from api.schemas.decision import ExtractedInfo
from config import settings
from storage import db


# --- state ---

class GraphState(TypedDict):
    conversation_id: str
    messages: list[dict]  # plain (replace) field -- context trim needs wholesale replacement, not append
    urgency: Optional[str]
    extracted: Optional[dict]
    query: Optional[str]  # built by the kb_retriever node, shared-name with agents.retriever.py's
                            # RetrieverState.query so LangGraph auto-transfers it into that embedded subgraph
    kb_results: Optional[list[dict]]  # shared-name with RetrieverState.kb_results -- auto-transferred back out
    decision: Optional[dict]
    pending_tool: Optional[dict]
    tool_calls: Annotated[list[dict], operator.add]
    step: int
    trace: Annotated[list[dict], operator.add]


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


class HandoffTool(BaseModel):
    """Hand control to exactly one specialist, or answer directly. Call this at most
    once per turn -- the supervisor cannot run two specialists at the same time."""
    next: Literal["triage_classifier", "kb_retriever", "action_decider", "direct_reply", "end"]
    # Found live (edge_case_greeting): the model sometimes calls HandoffTool with only
    # next/reply and omits reason entirely. reason was required (no default), so that
    # call failed model_validate() -- and the except-branch threw away the model's real
    # next="direct_reply" choice along with it, routing a plain greeting into the ticket
    # pipeline instead of answering it. A default here means a missing reason only loses
    # a trace label, never the actual routing decision.
    reason: str = Field("(no reason given)", description="One short phrase for why this hop was chosen.")
    reply: Optional[str] = Field(None, description="Only set when next='direct_reply'.")


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

# Every dialogue message the graph itself writes is phrased as "Speaker: sentence"
# (via _say() below) so GET /conversations/{id}'s message history reads as an actual
# conversation, not a log of "[node] key=value" tags. A genuine customer-facing
# direct_reply answer is the one exception -- it's appended unprefixed, since it IS
# the answer, not a line of internal commentary. agents/runner.py's _is_narration()
# uses this set to tell the two apart when rebuilding advisory chat history.
_DIALOGUE_SPEAKERS = {"Supervisor", "Triage Classifier", "KB Retriever", "Action Decider", "System"}


def _say(speaker: str, text: str) -> dict:
    return {"role": "agent", "content": f"{speaker}: {text}"}


# _supervisor.md/triage_classifier.md/action_decider.md are left untouched -- these
# two rules are appended at runtime instead, so they're easy to find/change together
# and don't require a version bump on prompt files whose own content hasn't changed.
_REACT_SUPERVISOR_RULES = (
    "You act ONLY by calling the HandoffTool function -- there is no other way to move "
    "this ticket forward. Decide using exactly this order, every turn:\n"
    "1. Not actually a ticket at all (a greeting, \"who are you\", \"what can you do\")? "
    "Call HandoffTool(next=\"direct_reply\", reply=<your short honest answer>). Do not "
    "classify it -- it was never a ticket.\n"
    "2. A real ticket with nothing classified yet? Call HandoffTool(next=\"triage_classifier\"). "
    "Nothing else may happen before this.\n"
    "3. Already classified, and this looks like an FAQ-answerable question (not a live "
    "incident already in progress)? Call HandoffTool(next=\"kb_retriever\"). A live outage "
    "or an in-progress billing error is NOT an FAQ lookup -- skip straight to step 4.\n"
    "4. Classified (and KB checked, if step 3 ran)? Call HandoffTool(next=\"action_decider\") "
    "to produce the final decision.\n"
    "Hard rule on top of that logic: call HandoffTool at most once per turn. A second "
    "call in the same turn does not mean \"do two things at once\" -- it is simply "
    "ignored, so making one only wastes the turn. The only time you make NO tool call "
    "at all is a direct conversational answer in your own words, with no handoff."
)

# Applies to the live conversation (messages) AND the API-facing Decision.rationale
# field, on request -- an operator reviewing a Thai ticket reads a Thai rationale,
# same language the customer used throughout.
_REACT_LANGUAGE_RULE = (
    "Write your `reason`/`rationale` text in the SAME language the customer used in "
    "their messages (not necessarily English) -- it gets read aloud as a status update "
    "to a colleague, in that language."
)

_KB_RETRIEVER_PHRASES = {
    "en": ("I searched the knowledge base and found {n} relevant article(s): {docs}.",
           "I searched the knowledge base but didn't find anything relevant to this question."),
    "th": ("ฉันค้นฐานความรู้แล้วเจอบทความที่เกี่ยวข้อง {n} รายการ: {docs}",
           "ฉันค้นฐานความรู้แล้ว แต่ไม่เจออะไรที่เกี่ยวข้องกับคำถามนี้เลย"),
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

    Returns (validated_model_or_None, failure_reason_or_None, raw_dict_or_None) --
    the reason lets the caller set an accurate decision rationale
    ("schema_validation_failed" vs "llm_unavailable") instead of a single
    hardcoded one regardless of what failed. `raw` is handed back too (even
    though it's also what validated the model from) so a caller can read an
    extra key validation itself doesn't care about -- see classify_fn/decide_fn's
    "_usage" key, used for real token/cost tracking without widening their
    public (state, messages) -> dict contract that test mocks everywhere rely on.
    """
    last_failure_reason = None
    for attempt in range(max_retries + 1):
        try:
            raw = fn()
        except LLMUnavailableError:
            last_failure_reason = "llm_unavailable"
            if attempt == max_retries:
                return None, last_failure_reason, None
            continue
        try:
            return model_cls.model_validate(raw), None, raw
        except ValidationError:
            last_failure_reason = "schema_validation_failed"
            if attempt == max_retries:
                return None, last_failure_reason, None
    return None, last_failure_reason, None


def _single_tool_call(response, step: int) -> tuple[Optional[dict], list[dict]]:
    """response.tool_calls can legally hold more than one entry -- bind_tools doesn't
    stop the model from doing that, and nothing in the prompt can fully guarantee it
    either. Keep only the first; log the drop as a trace event instead of silently
    ignoring (or silently racing on) the rest. This guard was dropped for a
    controlled experiment (to test whether the prompt alone was enough) and never
    actually triggered in that testing -- but it's user-facing now, not a controlled
    experiment, so it's back."""
    calls = getattr(response, "tool_calls", None) or []
    trace_entries = []
    if len(calls) > 1:
        trace_entries.append(_trace(step, "supervisor", "multiple_tool_calls_dropped",
                                     payload={"kept": calls[0]["name"], "dropped": [c["name"] for c in calls[1:]]}))
    return (calls[0] if calls else None), trace_entries


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

def _language_rule(state: GraphState) -> str:
    """Found live: with the generic "same language as the customer" instruction, the
    model sometimes picked the wrong language entirely (an all-English ticket got a
    Spanish rationale from action_decider) once the context had enough OTHER text in
    it (prior dialogue lines, the prompt itself) to make "the customer's language"
    ambiguous to infer. Once triage_classifier has already run, the language is a
    known fact, not something to infer -- state the ISO code explicitly instead of
    asking the model to re-derive it from a now-noisier context."""
    language = (state.get("extracted") or {}).get("language")
    if language:
        return (f"Write your `reason`/`rationale` text in the language with ISO code '{language}' "
                f"-- the language the customer used -- regardless of what language anything else "
                f"in this context happens to be written in.")
    return _REACT_LANGUAGE_RULE


def _classification_messages(state: GraphState) -> list[dict]:
    prompt = get_agents()["triage_classifier"].prompt
    thread_text = "\n".join(f"{m['role']}: {m['content']}" for m in state["messages"] if m["role"] != "history_summary")
    # language isn't known yet here -- it's literally what's being extracted -- so this
    # is the one call that still has to infer it from the thread text itself.
    return [{"role": "system", "content": prompt}, {"role": "system", "content": _REACT_LANGUAGE_RULE},
            {"role": "user", "content": thread_text}]


def _decision_messages(state: GraphState) -> list[dict]:
    prompt = get_agents()["action_decider"].prompt
    context = (f"extracted: {state.get('extracted')}\nkb_results: {state.get('kb_results')}\n"
               f"thread: {[m['content'] for m in state['messages'] if m['role'] != 'history_summary']}")
    return [{"role": "system", "content": prompt}, {"role": "system", "content": _language_rule(state)},
            {"role": "user", "content": context}]


def _default_classify_fn(state: GraphState, messages: list[dict]) -> dict:
    # method="function_calling": OpenAI's default strict JSON-schema mode rejects any
    # free-form dict field (requires additionalProperties:false on every nested object,
    # including ones we don't control the shape of, like ToolCall.args) -- function-calling
    # mode doesn't have that restriction and is just as reliable for our schemas
    #
    # include_raw=True: without it, with_structured_output() hands back only the parsed
    # Pydantic object, discarding the raw AIMessage -- and with it, usage_metadata, the
    # only place real token counts come from. Known gap noted in WRITEUP.md #7 until now.
    llm = make_llm_client(settings.OPENAI_MODEL).with_structured_output(
        RawClassification, method="function_calling", include_raw=True)
    result = call_with_retry(lambda: llm.invoke(messages))
    data = result["parsed"].model_dump() if result["parsed"] is not None else {}
    # "_usage" is read back out by _call_with_validation_retry's caller, then discarded --
    # Pydantic v2 ignores unknown fields by default, so RawClassification.model_validate()
    # never sees or rejects it.
    data["_usage"] = extract_usage(result["raw"], settings.OPENAI_MODEL)
    return data


def _default_decide_fn(state: GraphState, messages: list[dict]) -> dict:
    llm = make_llm_client(settings.OPENAI_MODEL).with_structured_output(
        RawDecision, method="function_calling", include_raw=True)
    result = call_with_retry(lambda: llm.invoke(messages))
    data = result["parsed"].model_dump() if result["parsed"] is not None else {}
    data["_usage"] = extract_usage(result["raw"], settings.OPENAI_MODEL)
    return data


def _default_supervisor_llm():
    return make_llm_client(settings.SUPERVISOR_MODEL).bind_tools([HandoffTool])


# --- graph factory ---

def build_graph(
    conn: sqlite3.Connection,
    checkpointer: BaseCheckpointSaver,
    classify_fn: Optional[Callable[[GraphState, list[dict]], dict]] = None,
    decide_fn: Optional[Callable[[GraphState, list[dict]], dict]] = None,
    summarize_fn: Optional[Callable[[list[dict]], str]] = None,
    supervisor_llm=None,
    retriever_graph=None,
):
    classify_fn = classify_fn or _default_classify_fn
    decide_fn = decide_fn or _default_decide_fn
    summarize_fn = summarize_fn or _default_summarize_fn
    llm = supervisor_llm or _default_supervisor_llm()
    # defaults to agents.retriever's real compiled graph (embeddings + vector search +
    # grade/rewrite loop), embedded directly as the "kb_retriever_search" node below --
    # not called through a plain function -- so get_graph(xray=True) can expand it into
    # its own box in the diagram, the same way the reference embeds its retriever_graph.
    # Injectable the same way classify_fn/decide_fn are: tests pass
    # retriever.build_stub_graph(fake_fn) so nothing here needs a real OpenAI embeddings call.
    retriever_graph = retriever_graph or retriever._default_graph()

    def supervisor(state: GraphState) -> Command:
        step = state["step"] + 1
        trimmed = maybe_trim_messages(state["messages"], summarize_fn)
        trim_happened = len(trimmed) != len(state["messages"])
        trace_entries = [_trace(step, "supervisor", "turn_start")]
        updates: dict[str, Any] = {"step": step, "trace": trace_entries}
        if trim_happened:
            updates["messages"] = trimmed  # messages is a plain field: this replaces, not appends
            trace_entries.append(_trace(step, "supervisor", "context_trimmed", payload={"new_length": len(trimmed)}))

        if step >= settings.MAX_STEPS:
            return Command(goto="force_escalate", update=updates)

        if state.get("decision") is not None:
            # Found live: action_decider and await_tool_approval used to go straight to
            # END without ever routing back through here, breaking the "every specialist
            # reports back to the supervisor" invariant every OTHER node follows
            # (triage_classifier/kb_retriever both edge back to supervisor). Routing them
            # back here too costs nothing -- no LLM call happens on this path -- and this
            # closing line is what makes that last hop visible in the transcript.
            updates["messages"] = updates.get("messages", state["messages"]) + [
                _say("Supervisor", "This ticket's been decided -- wrapping up, nothing further to route.")]
            return Command(goto=END, update=updates)

        current_messages = updates.get("messages", state["messages"])
        lc_messages = [SystemMessage(
            get_supervisor_prompt() + "\n\n" + _REACT_SUPERVISOR_RULES + "\n\n" + _language_rule(state))]
        for m in current_messages:
            if m["role"] in ("system", "history_summary"):
                continue
            cls = AIMessage if m["role"] == "agent" else HumanMessage
            lc_messages.append(cls(m["content"]))

        response = call_with_retry(lambda: llm.invoke(lc_messages))
        # bind_tools() (unlike with_structured_output()) hands back the raw AIMessage
        # directly -- usage_metadata is already right there, no include_raw needed.
        usage = extract_usage(response, settings.SUPERVISOR_MODEL)
        call, drop_trace = _single_tool_call(response, step)
        trace_entries += drop_trace

        if call is None:
            if state.get("extracted") is None:
                # legitimate direct_reply case: nothing classified yet, so this might
                # genuinely not be a ticket at all (step 1 of _REACT_SUPERVISOR_RULES).
                reply = response.content or "Sorry, could you rephrase that?"
                handoff = HandoffTool(next="direct_reply", reason="model answered directly, no handoff tool call",
                                      reply=reply)
            else:
                # Found live (ticket_3_dark_mode): mid-pipeline, the model sometimes
                # narrates its next move in plain content ("I'm now handing this to
                # Action Decider...") instead of actually calling HandoffTool. Treating
                # that content as the direct_reply text would put that narration in
                # front of the customer as draft_reply -- never valid once a ticket is
                # already classified, so fall back to the safe default instead.
                handoff = HandoffTool(next="action_decider",
                                      reason=f"no handoff tool call mid-pipeline, falling back to the safe "
                                             f"default (model said: {response.content!r})")
        else:
            try:
                handoff = HandoffTool.model_validate(call["args"])
            except ValidationError:
                handoff = HandoffTool(next="triage_classifier" if state.get("extracted") is None else "action_decider",
                                       reason="malformed handoff call, falling back to the safe default")

        # same safety net as before: a ticket can't reach action_decider without
        # extracted being set, regardless of what the model's tool call says.
        if handoff.next == "action_decider" and state.get("extracted") is None:
            handoff = HandoffTool(next="triage_classifier",
                                   reason=f"override: action_decider requires extracted first "
                                          f"(model wanted: {handoff.reason})")

        updates["trace"] = trace_entries + [_trace(step, "supervisor", "handoff",
                                                     payload={"to": handoff.next, "reason": handoff.reason},
                                                     model=usage["model"], tokens=usage)]

        _SPECIALIST_LABELS = {"triage_classifier": "Triage Classifier", "kb_retriever": "KB Retriever",
                              "action_decider": "Action Decider"}
        if handoff.next in ("triage_classifier", "kb_retriever", "action_decider"):
            updates["messages"] = current_messages + [
                _say("Supervisor", f"I'm handing this to {_SPECIALIST_LABELS[handoff.next]} -- {handoff.reason}")]

        if handoff.next == "direct_reply" and handoff.reply:
            updates["messages"] = current_messages + [{"role": "agent", "content": handoff.reply}]
            # represented as a lightweight auto_respond, not a schema carve-out -- every
            # ticket's decision has the same shape (Decision requires urgency/extracted/
            # action/rationale), so a greeting or meta-question doesn't need a
            # special-cased response contract
            updates["decision"] = {
                "urgency": "low",
                "extracted": {"product_area": "general", "issue_type": "general_inquiry", "sentiment": "neutral",
                              "language": "en", "confidence": 1.0, "rationale": "not a support ticket"},
                "action": "auto_respond",
                "draft_reply": handoff.reply,
                "rationale": "direct conversational reply -- not a ticket requiring triage",
                "confidence": 1.0,
                "cited_kb_doc_ids": [],
            }

        # the handoff tool call IS the routing decision -- Command(goto=...) branches
        # straight to the chosen node, no separate next_hop field + conditional edge
        # needed the way an earlier structured-output version used one.
        goto = END if handoff.next in ("direct_reply", "end") else handoff.next
        return Command(goto=goto, update=updates)

    def call_triage_classifier(state: GraphState) -> dict:
        step = state["step"] + 1
        messages = _classification_messages(state)
        validated, failure_reason, raw = _call_with_validation_retry(
            lambda: classify_fn(state, messages), RawClassification)
        usage = (raw or {}).get("_usage")
        if validated is None:
            # a validation failure still cost a real API call (raw may carry "_usage"
            # even when "parsed" came back None) -- track that spend too, not just the
            # calls that happened to succeed.
            return {
                "step": step,
                "decision": {**_SCHEMA_FAILURE_DECISION, "rationale": failure_reason},
                "trace": [_trace(step, "call_triage_classifier", failure_reason,
                                  model=usage.get("model") if usage else None, tokens=usage)],
            }
        extracted = validated.model_dump(exclude={"urgency"})
        # validated.rationale is free-form prose the model writes in the customer's own
        # language (per _REACT_LANGUAGE_RULE) -- stitching English scaffolding around
        # the structured fields (issue_type, product_area, etc, short English-style
        # taxonomy tags by design) could never actually read as Thai no matter what's
        # translated, so this uses the rationale directly instead of rebuilding a sentence.
        status_message = _say("Triage Classifier", validated.rationale)
        return {
            "step": step,
            "extracted": extracted,
            "urgency": validated.urgency,
            "messages": state["messages"] + [status_message],
            "trace": [_trace(step, "call_triage_classifier", "decision",
                              payload={"urgency": validated.urgency, **extracted},
                              model=usage.get("model") if usage else None, tokens=usage)],
        }

    # kb_retriever is now a 3-node chain, not one function -- call_kb_retriever (below)
    # builds the query, then control passes to "kb_retriever_search" (agents/retriever.py's
    # compiled graph, embedded directly as that node -- see build_graph()'s retriever_graph
    # param) via a static edge, then to after_kb_retriever (below) via another. Split this
    # way, instead of one function calling retriever.retrieve() directly, specifically so
    # get_graph(xray=True) can expand "kb_retriever_search" into its own box in the
    # architecture diagram, the same way the reference project embeds its retriever graph.
    # query/kb_results are shared-name fields with RetrieverState (see retriever.py's module
    # docstring) -- LangGraph auto-transfers them into/out of the embedded subgraph, so
    # neither node here has to pass them through by hand.
    def call_kb_retriever(state: GraphState) -> dict:
        # Found live (golden-set tickets 6/7): using extracted["issue_type"] (a short
        # taxonomy tag like "feature_question") as the search query matched nothing in
        # the FAQ corpus for anything but a lucky literal substring hit -- vector/TF-IDF
        # search alike need the customer's own words, not a category label, to have any
        # real chance of matching the docs' vocabulary or embedding space.
        customer_text = " ".join(m["content"] for m in state["messages"] if m["role"] == "customer")
        query = customer_text or (state.get("extracted") or {}).get("issue_type", "")
        return {"query": query or "general question"}

    def after_kb_retriever(state: GraphState) -> dict:
        step = state["step"] + 1
        results = state.get("kb_results") or []
        # Found live: without a status message here, the supervisor's only input is
        # state["messages"] -- so with no record that kb_retriever already ran, every
        # subsequent turn looked identical to the first ("still an unanswered FAQ
        # question") and it kept re-handing-off to kb_retriever until MAX_STEPS forced
        # an escalation. triage_classifier already reports back this way; kb_retriever
        # must too, for the same reason.
        #
        # kb_retriever_search's own agentic loop (agents/retriever.py) already decides
        # relevance and retries internally -- this node can't just tell IT to match the
        # customer's language the way an LLM-call node can, so _KB_RETRIEVER_PHRASES covers
        # the languages this project actually tests (en/th); anything else falls back to English.
        language = (state.get("extracted") or {}).get("language", "en")
        found, not_found = _KB_RETRIEVER_PHRASES.get(language, _KB_RETRIEVER_PHRASES["en"])
        doc_ids = list(dict.fromkeys(r["doc_id"] for r in results))  # dedupe, preserve order --
                                                                       # a topic split into >1 chunk
                                                                       # can otherwise show up twice
        if doc_ids:
            doc_list = ", ".join(doc_ids)
            status_message = _say("KB Retriever", found.format(n=len(doc_ids), docs=doc_list))
        else:
            status_message = _say("KB Retriever", not_found)
        return {
            "step": step,
            "messages": state["messages"] + [status_message],
            "trace": [_trace(step, "call_kb_retriever", "tool_call",
                              payload={"tool_name": "search_knowledge_base", "query": state.get("query"), "results": results})],
        }

    def call_action_decider(state: GraphState) -> dict:
        step = state["step"] + 1
        messages = _decision_messages(state)
        validated, failure_reason, raw = _call_with_validation_retry(
            lambda: decide_fn(state, messages), RawDecision)
        usage = (raw or {}).get("_usage")
        if validated is None:
            return {
                "step": step,
                "decision": {**_SCHEMA_FAILURE_DECISION, "rationale": failure_reason},
                "trace": [_trace(step, "call_action_decider", failure_reason,
                                  model=usage.get("model") if usage else None, tokens=usage)],
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
            "trace": [_trace(step, "call_action_decider", "decision", payload=decision,
                              model=usage.get("model") if usage else None, tokens=usage)],
        }
        if validated.pending_tool and validated.pending_tool.get("tool_name") in tools.SIDE_EFFECTING_TOOL_IMPLS:
            updates["pending_tool"] = validated.pending_tool
            tool_label = validated.pending_tool["tool_name"].replace("_", " ")
            # Leads with validated.rationale (language-matched, like triage_classifier's
            # status message) -- the "needs human approval" clause stays a fixed English
            # connector naming the tool.
            updates["messages"] = state["messages"] + [_say(
                "Action Decider", f"{validated.rationale} ({tool_label} needs human approval before I can proceed.)")]
        return updates

    def force_escalate(state: GraphState) -> dict:
        step = state["step"] + 1
        decision = {**_SCHEMA_FAILURE_DECISION, "rationale": "max_steps_exceeded"}
        msg = _say("System", "I've gone back and forth too many times without reaching a decision, "
                              "so I'm escalating this to a human instead of continuing.")
        return {"step": step, "decision": decision, "messages": state["messages"] + [msg],
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

        tool_label = tool["tool_name"].replace("_", " ")
        if status == "executed":
            outcome_text = f"The {tool_label} was approved, and I've executed it. Result: {result}."
        else:
            outcome_text = f"The {tool_label} request was rejected -- I won't be executing it."
        if human_decision.reason:
            outcome_text += f" Reason given: {human_decision.reason}"

        return {
            "step": step,
            "decision": decision,
            "tool_calls": [tool_call_record],
            # messages is a plain field -- must append explicitly, a bare [outcome_message]
            # here would wipe out the rest of the thread instead of adding to it
            "messages": state["messages"] + [_say("System", outcome_text)],
            "trace": [_trace(step, "await_tool_approval", "resumed", payload=tool_call_record)],
        }

    def _route_from_action_decider(state: GraphState) -> str:
        # action_decider and await_tool_approval are the only two nodes that must route
        # back to the supervisor conditionally rather than unconditionally -- everything
        # else either always continues (triage_classifier/kb_retriever -> supervisor,
        # await_tool_approval -> supervisor) or always ends (force_escalate -> END, see
        # the comment on that edge below for why it's the one exception).
        return "await_tool_approval" if state.get("pending_tool") else "supervisor"

    graph = StateGraph(GraphState)
    # destinations=... is declarative only (used for graph visualization/validation) --
    # supervisor() returning Command(goto=...) is what actually branches at runtime,
    # there is no conditional-edge router to also list these targets in.
    graph.add_node("supervisor", supervisor,
                    destinations=("triage_classifier", "kb_retriever", "action_decider", "force_escalate", END))
    graph.add_node("triage_classifier", call_triage_classifier)
    graph.add_node("kb_retriever", call_kb_retriever)
    # retriever_graph embedded directly (not called from inside a function) -- see
    # after_kb_retriever's comment above for why, and retriever.py's module docstring
    # for how its internal state stays isolated from this graph's.
    graph.add_node("kb_retriever_search", retriever_graph)
    graph.add_node("after_kb_retriever", after_kb_retriever)
    graph.add_node("action_decider", call_action_decider)
    graph.add_node("force_escalate", force_escalate)
    graph.add_node("await_tool_approval", await_tool_approval)

    graph.set_entry_point("supervisor")
    graph.add_edge("triage_classifier", "supervisor")
    graph.add_edge("kb_retriever", "kb_retriever_search")
    graph.add_edge("kb_retriever_search", "after_kb_retriever")
    graph.add_edge("after_kb_retriever", "supervisor")
    graph.add_conditional_edges("action_decider", _route_from_action_decider, {
        "await_tool_approval": "await_tool_approval",
        "supervisor": "supervisor",
    })
    # force_escalate -> END stays a direct edge, NOT back to supervisor: the
    # supervisor's MAX_STEPS check runs before its decision-is-not-None check, so
    # looping back here would just keep re-triggering force_escalate forever instead
    # of ending.
    graph.add_edge("force_escalate", END)
    graph.add_edge("await_tool_approval", "supervisor")

    return graph.compile(checkpointer=checkpointer)
