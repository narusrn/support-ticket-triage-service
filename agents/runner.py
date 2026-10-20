"""Ties agents/orchestrator.py execution to storage/db.py persistence, and
owns the checkpointer-backed compiled graph for a process's lifetime.

Async throughout -- production uses AsyncSqliteSaver (verified against the
real package during planning), which requires async calls. Tests use
InMemorySaver instead, which supports the same async interface, so no test
needs a real OpenAI key or a real SQLite file to exercise this module.
"""
import time
import uuid
from datetime import datetime, timezone
from typing import Callable, Optional

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.types import Command

from agents.llm import call_with_retry, get_supervisor_prompt, make_llm_client
from agents.orchestrator import build_graph
from config import settings
from storage import db


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()


def _default_advisory_reply(state: dict, message_text: str) -> str:
    """Used for a follow-up turn on an already-decided conversation -- never
    invokes the graph, just reads the current state and answers. Cannot
    approve/reject/execute anything regardless of what the operator asks;
    only the /approve and /reject endpoints can do that."""
    client = make_llm_client(settings.OPENAI_MODEL)
    context = f"Current decision: {state.get('decision')}\nPending tool (if any): {state.get('pending_tool')}"
    response = call_with_retry(lambda: client.invoke([
        {"role": "system", "content": "You are a support triage assistant discussing an already-triaged "
                                       "ticket with an operator. Answer their question using the context "
                                       "below. You cannot approve, reject, or execute any tool call from "
                                       "this conversation -- only the dedicated approve/reject endpoints can."},
        {"role": "system", "content": context},
        {"role": "user", "content": message_text},
    ]))
    return response.content


class Runner:
    def __init__(
        self,
        conn,
        checkpointer: BaseCheckpointSaver,
        classify_fn: Optional[Callable] = None,
        decide_fn: Optional[Callable] = None,
        route_fn: Optional[Callable] = None,
        summarize_fn: Optional[Callable] = None,
        advisory_fn: Callable[[dict, str], str] = _default_advisory_reply,
    ):
        self.conn = conn
        self.graph = build_graph(conn, checkpointer, classify_fn=classify_fn, decide_fn=decide_fn,
                                  route_fn=route_fn, summarize_fn=summarize_fn)
        self.advisory_fn = advisory_fn

    # --- public API ---

    async def ingest_ticket(self, conversation_id: str, messages: list[dict], customer_metadata: dict) -> dict:
        db.insert_conversation(self.conn, conversation_id, "completed", customer_metadata)
        for m in messages:
            db.insert_message(self.conn, conversation_id, m["role"], m["content"], m.get("sent_at"))

        # customer_metadata matters for triage (per spec) but was previously only ever
        # written to storage.db -- never visible to classify_fn/decide_fn/route_fn, since
        # those all build their prompt from state["messages"] alone. A second system
        # message surfaces it the same way the main system prompt does.
        metadata_line = ", ".join(f"{k}={v}" for k, v in customer_metadata.items())
        initial_state = {
            "conversation_id": conversation_id,
            "messages": [
                {"role": "system", "content": get_supervisor_prompt()},
                {"role": "system", "content": f"Customer metadata: {metadata_line}"},
            ] + [{"role": m["role"], "content": m["content"]} for m in messages],
            "urgency": None, "extracted": None, "kb_results": None, "decision": None,
            "pending_tool": None, "tool_calls": [], "step": 0, "trace": [], "next_hop": None,
        }
        config = {"configurable": {"thread_id": conversation_id}}
        started = time.time()
        result = await self.graph.ainvoke(initial_state, config)

        status = "awaiting_approval" if "__interrupt__" in result else "completed"
        db.update_conversation_status(self.conn, conversation_id, status)
        self._persist_run(conversation_id, started, result)
        return self._build_response(conversation_id)

    async def send_message(self, conversation_id: str, message_text: str) -> dict:
        db.insert_message(self.conn, conversation_id, "operator", message_text)

        config = {"configurable": {"thread_id": conversation_id}}
        snapshot = await self.graph.aget_state(config)
        state = snapshot.values

        # already triaged (whether awaiting_approval or completed) -- lightweight advisory
        # reply, never touches the graph/checkpoint. See PLAN.md "Resuming" for why.
        reply = self.advisory_fn(state, message_text)
        db.insert_message(self.conn, conversation_id, "agent", reply)

        response = self._build_response(conversation_id)
        response["reply"] = reply
        return response

    async def approve(self, conversation_id: str, tool_call_id: str, reason: Optional[str] = None) -> dict:
        return await self._resolve(conversation_id, tool_call_id, "approve", reason)

    async def reject(self, conversation_id: str, tool_call_id: str, reason: Optional[str] = None) -> dict:
        return await self._resolve(conversation_id, tool_call_id, "reject", reason)

    # --- internal ---

    async def _resolve(self, conversation_id: str, tool_call_id: str, decision: str, reason: Optional[str]) -> dict:
        existing = db.get_tool_call(self.conn, tool_call_id)
        if existing is None:
            raise ValueError(f"no such tool_call_id {tool_call_id!r}")

        if existing["status"] in ("executed", "rejected"):
            # idempotent replay: the graph already resumed from an earlier call, or never needs
            # to be touched for this one -- return the stored result without invoking it again
            response = self._build_response(conversation_id)
            response["replayed"] = True
            return response

        config = {"configurable": {"thread_id": conversation_id}}
        started = time.time()
        result = await self.graph.ainvoke(Command(resume={"decision": decision, "reason": reason}), config)

        status = "awaiting_approval" if "__interrupt__" in result else "completed"
        db.update_conversation_status(self.conn, conversation_id, status)
        self._persist_run(conversation_id, started, result)
        response = self._build_response(conversation_id)
        response["replayed"] = False
        return response

    def _sync_graph_messages(self, conversation_id: str, result_state: dict) -> None:
        """Copies any new entries from the graph's own state["messages"] (the supervisor's
        handoff/status/outcome lines) into storage.db's messages table, so GET
        /conversations/{id} shows the full conversation, not just the raw customer thread.

        Content-based dedup, not count-based: send_message()'s advisory path inserts
        operator/agent messages straight into the DB without ever touching the graph, so
        db.get_messages()'s row count can run ahead of what's reflected in state["messages"].
        Comparing (role, content) pairs instead of lengths stays correct regardless of which
        path added which message, and is cheap at this scale (a handful of messages per
        conversation). The system prompt / metadata line are skipped -- internal setup
        context, not part of the conversation.
        """
        graph_messages = [m for m in result_state.get("messages", []) if m["role"] != "system"]
        existing = {(m["role"], m["content"]) for m in db.get_messages(self.conn, conversation_id)}
        for m in graph_messages:
            key = (m["role"], m["content"])
            if key not in existing:
                db.insert_message(self.conn, conversation_id, m["role"], m["content"])
                existing.add(key)

    def _persist_run(self, conversation_id: str, started_at: float, result_state: dict,
                      stopped_reason: str = "end") -> None:
        run_id = f"run_{uuid.uuid4().hex[:8]}"
        finished_at = time.time()

        self._sync_graph_messages(conversation_id, result_state)

        for entry in result_state.get("trace", []):
            db.insert_trace_event(self.conn, run_id, conversation_id, entry["step_index"], entry["node"],
                                   entry["event_type"], payload=entry.get("payload"), model=entry.get("model"),
                                   latency_ms=entry.get("latency_ms"), tokens=entry.get("tokens"))

        decision = result_state.get("decision")
        if decision is not None:
            db.insert_decision(self.conn, conversation_id, decision)

        pending = [tc["tool_call_id"] for tc in db.get_tool_calls(self.conn, conversation_id)
                   if tc["status"] == "pending_approval"]

        db.insert_run(self.conn, {
            "run_id": run_id, "conversation_id": conversation_id,
            "started_at": _iso(started_at), "finished_at": _iso(finished_at),
            "duration_ms": int((finished_at - started_at) * 1000),
            "steps": result_state.get("step", 0),
            "final_urgency": decision.get("urgency") if decision else None,
            "final_action": decision.get("action") if decision else None,
            "pending_approvals": pending,
            "completed": True,
            "stopped_reason": stopped_reason,
        })

    def _build_response(self, conversation_id: str) -> dict:
        conversation = db.get_conversation(self.conn, conversation_id)
        decisions = db.get_decisions(self.conn, conversation_id)
        latest_decision = dict(decisions[-1]) if decisions else {}
        # tool_calls is assembled fresh from the DB (the single source of truth for persisted
        # state), not trusted from the graph's in-memory decision dict, which may not reflect
        # a tool_call inserted by await_tool_approval right before it suspended.
        latest_decision["tool_calls"] = db.get_tool_calls(self.conn, conversation_id)
        return {"conversation_id": conversation_id, "status": conversation["status"], "decision": latest_decision}
