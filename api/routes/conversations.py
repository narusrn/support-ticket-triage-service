from typing import Optional

from fastapi import APIRouter, HTTPException, Path, Request, status

from api.schemas.api import ApprovalRequest, ErrorResponse, OperatorMessageRequest, UsageSummary
from api.schemas.decision import ConversationDetail, TicketResponse
from storage import db

router = APIRouter()

_CONVERSATION_ID = Path(..., description="The conversation_id returned by POST /tickets.",
                        examples=["c_a1b2c3d4"])
_TOOL_CALL_ID = Path(..., description="A tool_call_id from decision.tool_calls -- stable across retries and "
                                       "server restarts (derived from a hash of conversation_id, tool_name, "
                                       "and args), unlike Python's hash().", examples=["tc_a30ebab79e"])


def _require_conversation(conn, conversation_id: str) -> dict:
    conv = db.get_conversation(conn, conversation_id)
    if conv is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "conversation_not_found", "message": f"No conversation with id {conversation_id}"},
        )
    return conv


@router.post(
    "/conversations/{conversation_id}/messages",
    response_model=TicketResponse,
    summary="Operator turn -- discuss an already-triaged ticket",
    description=(
        "Free-text follow-up on a conversation that already has a decision (whether "
        "`awaiting_approval` or `completed`) -- e.g. an operator asking 'why is this urgent?' "
        "or 'what KB article did this cite?'.\n\n"
        "**Never re-enters the triage graph.** This is a lightweight advisory call over the "
        "current decision + the full prior message history (including earlier advisory turns "
        "in the same conversation, so follow-ups like 'and in Thai baht?' work). It cannot "
        "approve, reject, or execute anything regardless of what the message says or implies "
        "-- that boundary is enforced in code (only the dedicated approve/reject endpoints "
        "below can resume the graph), not by trusting the model to refuse.\n\n"
        "The reply is both returned in `reply` and persisted, so it shows up in the `messages` "
        "list from `GET /conversations/{conversation_id}` afterward."
    ),
    responses={
        404: {"model": ErrorResponse, "description": "Unknown conversation_id."},
        200: {"content": {"application/json": {"example": {
            "conversation_id": "c_a1b2c3d4", "status": "awaiting_approval",
            "decision": {"urgency": "critical", "action": "escalate_to_human", "confidence": 0.9,
                         "extracted": {"product_area": "billing", "issue_type": "duplicate_charge",
                                       "sentiment": "angry", "language": "en", "confidence": 0.9,
                                       "rationale": "3 charges in 2h"},
                         "draft_reply": None, "tool_calls": [], "rationale": "needs approval",
                         "cited_kb_doc_ids": []},
            "reply": "This is urgent because three separate charges were captured, not just pending holds.",
        }}}},
    },
)
async def send_message(
    body: OperatorMessageRequest, request: Request, conversation_id: str = _CONVERSATION_ID,
) -> TicketResponse:
    _require_conversation(request.app.state.conn, conversation_id)
    runner = request.app.state.runner
    result = await runner.send_message(conversation_id, body.message)
    return TicketResponse(conversation_id=result["conversation_id"], status=result["status"],
                           decision=result["decision"], reply=result["reply"])


@router.post(
    "/conversations/{conversation_id}/tool-calls/{tool_call_id}/approve",
    response_model=TicketResponse,
    summary="Approve a pending side-effecting tool call",
    description=(
        "**The only code path that actually executes `issue_refund` or "
        "`open_incident_or_page_oncall`.** There is no way for the agent to run either tool "
        "on its own -- `action_decider` can only ever propose one, which suspends the graph "
        "(`interrupt()`) until a human calls this endpoint. Resumes the graph from exactly "
        "where it paused, dispatches the real (mocked) side effect, and returns the updated "
        "decision with `tool_calls[].status == \"executed\"`.\n\n"
        "**Idempotent**: if this `tool_call_id` was already resolved by an earlier approve or "
        "reject call, returns the stored result (`replayed: true` internally) without "
        "touching the graph or re-running the side effect a second time -- safe to retry on a "
        "timeout or double-click."
    ),
    responses={
        404: {"model": ErrorResponse, "description": "Unknown conversation_id or tool_call_id."},
        200: {"content": {"application/json": {"example": {
            "conversation_id": "c_a1b2c3d4", "status": "completed",
            "decision": {"urgency": "critical", "action": "escalate_to_human", "confidence": 0.9,
                         "extracted": {"product_area": "billing", "issue_type": "duplicate_charge",
                                       "sentiment": "angry", "language": "en", "confidence": 0.9,
                                       "rationale": "3 charges in 2h"},
                         "draft_reply": None, "rationale": "needs approval", "cited_kb_doc_ids": [],
                         "tool_calls": [{"tool_call_id": "tc_a30ebab79e", "tool_name": "issue_refund",
                                          "args": {"charge_ids": ["ch_1", "ch_2", "ch_3"], "amount_usd": 89.97},
                                          "status": "executed", "reason": "verified in Stripe",
                                          "result": {"refund_id": "re_123456", "status": "succeeded"}}]},
        }}}},
    },
)
async def approve_tool_call(
    request: Request, conversation_id: str = _CONVERSATION_ID, tool_call_id: str = _TOOL_CALL_ID,
    body: Optional[ApprovalRequest] = None,
) -> TicketResponse:
    _require_conversation(request.app.state.conn, conversation_id)
    runner = request.app.state.runner
    reason = body.reason if body else None
    try:
        result = await runner.approve(conversation_id, tool_call_id, reason)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "tool_call_not_found", "message": str(exc)},
        )
    return TicketResponse(conversation_id=result["conversation_id"], status=result["status"], decision=result["decision"])


@router.post(
    "/conversations/{conversation_id}/tool-calls/{tool_call_id}/reject",
    response_model=TicketResponse,
    summary="Reject a pending side-effecting tool call",
    description=(
        "Same resume path as `/approve` -- the graph wakes up from the same `interrupt()` -- "
        "but the side effect is never dispatched. `tool_calls[].status` becomes `\"rejected\"` "
        "and `result` stays `null`. Also idempotent: resolving an already-resolved tool call "
        "again just returns the stored (rejected) result."
    ),
    responses={404: {"model": ErrorResponse, "description": "Unknown conversation_id or tool_call_id."}},
)
async def reject_tool_call(
    request: Request, conversation_id: str = _CONVERSATION_ID, tool_call_id: str = _TOOL_CALL_ID,
    body: Optional[ApprovalRequest] = None,
) -> TicketResponse:
    _require_conversation(request.app.state.conn, conversation_id)
    runner = request.app.state.runner
    reason = body.reason if body else None
    try:
        result = await runner.reject(conversation_id, tool_call_id, reason)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "tool_call_not_found", "message": str(exc)},
        )
    return TicketResponse(conversation_id=result["conversation_id"], status=result["status"], decision=result["decision"])


@router.get(
    "/conversations/{conversation_id}",
    response_model=ConversationDetail,
    summary="Full audit trail for a conversation",
    description=(
        "Everything recorded for one conversation: the raw customer thread, every internal "
        "agent dialogue line (Supervisor/Triage Classifier/KB Retriever/Action Decider/System), "
        "every operator follow-up, every decision ever reached, every tool call proposed "
        "(regardless of status), and a rolled-up usage summary (runs/tokens/cost/latency) for "
        "just this conversation. Works identically whether called right after ingest or after "
        "a server restart days later -- it's read straight from SQLite, not from any in-memory "
        "graph state."
    ),
    responses={404: {"model": ErrorResponse, "description": "Unknown conversation_id."}},
)
async def get_conversation(request: Request, conversation_id: str = _CONVERSATION_ID) -> ConversationDetail:
    conn = request.app.state.conn
    conv = _require_conversation(conn, conversation_id)
    usage = db.get_usage_summary(conn, conversation_id)
    decisions = db.get_decisions(conn, conversation_id)
    tool_calls = db.get_tool_calls(conn, conversation_id)
    # the decisions table has no per-decision tool_calls column -- the real data lives only
    # in the tool_calls table above. Attach it to the latest decision (same convention
    # Runner._build_response() uses for the other endpoints) so it isn't just a stray [].
    if decisions:
        decisions[-1]["tool_calls"] = tool_calls
    return ConversationDetail(
        conversation_id=conversation_id,
        status=conv["status"],
        messages=db.get_messages(conn, conversation_id),
        decisions=decisions,
        tool_calls=tool_calls,
        usage=UsageSummary(**usage),
    )
