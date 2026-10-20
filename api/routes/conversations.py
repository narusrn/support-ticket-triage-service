from typing import Optional

from fastapi import APIRouter, HTTPException, Request, status

from api.schemas.api import ApprovalRequest, ErrorResponse, OperatorMessageRequest, UsageSummary
from api.schemas.decision import ConversationDetail, TicketResponse
from storage import db

router = APIRouter()


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
        "awaiting_approval or completed). Never re-enters the triage graph -- answers from "
        "the current state via a lightweight advisory call. Cannot approve, reject, or "
        "execute anything regardless of what the message says; only the dedicated "
        "approve/reject endpoints can do that."
    ),
    responses={404: {"model": ErrorResponse, "description": "Unknown conversation_id."}},
)
async def send_message(conversation_id: str, body: OperatorMessageRequest, request: Request) -> TicketResponse:
    _require_conversation(request.app.state.conn, conversation_id)
    runner = request.app.state.runner
    result = await runner.send_message(conversation_id, body.message)
    return TicketResponse(conversation_id=result["conversation_id"], status=result["status"], decision=result["decision"])


@router.post(
    "/conversations/{conversation_id}/tool-calls/{tool_call_id}/approve",
    response_model=TicketResponse,
    summary="Approve a pending side-effecting tool call",
    description=(
        "The only path that actually executes issue_refund or open_incident_or_page_oncall. "
        "Idempotent: if this tool_call was already resolved (by an earlier approve/reject "
        "call), returns the stored result without touching the graph or re-executing anything."
    ),
    responses={404: {"model": ErrorResponse, "description": "Unknown conversation_id or tool_call_id."}},
)
async def approve_tool_call(
    conversation_id: str, tool_call_id: str, request: Request, body: Optional[ApprovalRequest] = None
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
    description="Same resume path as approve, but the side effect is never executed.",
    responses={404: {"model": ErrorResponse, "description": "Unknown conversation_id or tool_call_id."}},
)
async def reject_tool_call(
    conversation_id: str, tool_call_id: str, request: Request, body: Optional[ApprovalRequest] = None
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
    description="Messages, every decision, every tool call, and a rolled-up usage summary.",
    responses={404: {"model": ErrorResponse, "description": "Unknown conversation_id."}},
)
async def get_conversation(conversation_id: str, request: Request) -> ConversationDetail:
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
