"""Tool implementations and the idempotency-guarded dispatcher.

search_knowledge_base (now agents/retriever.py's agentic RAG retriever,
called directly from agents/orchestrator.py's call_kb_retriever -- not
re-exported here anymore) is read-only and executes immediately -- never
goes through the approval gate. issue_refund and open_incident_or_page_oncall
ARE side-effecting: the graph never calls them directly -- it only ever
reaches them through dispatch_tool_call(), after a human has approved via
the /approve endpoint. That boundary lives in the graph's routing, not here;
this module only guarantees that once a tool_call row exists, re-dispatching
it never re-executes the side effect.
"""
import random
from typing import Any, Optional

from storage import db


class ToolExecutionError(Exception):
    """Raised by a mocked tool to simulate a transient failure."""


def issue_refund(charge_ids: list[str], amount_usd: float, *, simulate_failure: bool = False) -> dict[str, Any]:
    if simulate_failure:
        raise ToolExecutionError("payment_gateway_timeout")
    return {
        "refund_id": f"re_{random.randint(100000, 999999)}",
        "charge_ids": charge_ids,
        "amount_usd": amount_usd,
        "status": "succeeded",
    }


def open_incident_or_page_oncall(summary: str, severity: str, *, simulate_failure: bool = False) -> dict[str, Any]:
    if simulate_failure:
        raise ToolExecutionError("paging_service_unavailable")
    return {
        "incident_id": f"inc_{random.randint(1000, 9999)}",
        "summary": summary,
        "severity": severity,
        "status": "paged",
    }


SIDE_EFFECTING_TOOL_IMPLS = {
    "issue_refund": issue_refund,
    "open_incident_or_page_oncall": open_incident_or_page_oncall,
}


def dispatch_tool_call(conn, tool_call_id: str, reason: Optional[str] = None) -> dict[str, Any]:
    """Executes a side-effecting tool call already recorded in storage.db. Safe to
    call twice with the same tool_call_id: an already-resolved call returns the
    stored result instead of re-running the mock tool."""
    tool_call = db.get_tool_call(conn, tool_call_id)
    if tool_call is None:
        raise ValueError(f"no tool_call row for {tool_call_id!r}")

    if tool_call["status"] in ("executed", "rejected"):
        return tool_call["result"]

    impl = SIDE_EFFECTING_TOOL_IMPLS[tool_call["tool_name"]]
    try:
        result = impl(**tool_call["args"])
    except ToolExecutionError as exc:
        db.update_tool_call(conn, tool_call_id, "rejected", reason=f"tool_execution_error: {exc}")
        raise

    db.update_tool_call(conn, tool_call_id, "executed", reason=reason, result=result)
    return result
