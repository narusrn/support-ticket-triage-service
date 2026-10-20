from uuid import uuid4

from fastapi import APIRouter, Request, status

from api.schemas.api import ErrorResponse, TicketIngestRequest
from api.schemas.decision import TicketResponse

router = APIRouter()


@router.post(
    "/tickets",
    response_model=TicketResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Ingest a new support ticket",
    description=(
        "Accepts a full message thread that already exists -- the customer is offline, this "
        "system never talks to them directly. Creates a new conversation, runs the triage "
        "agent once, and returns the initial decision. If the agent requests a side-effecting "
        "tool, the response's `status` will be 'awaiting_approval' and the pending tool_call "
        "must be resolved via `POST .../tool-calls/{id}/approve` or `.../reject` before "
        "anything executes."
    ),
    responses={422: {"model": ErrorResponse, "description": "Validation failed, e.g. an empty messages list."}},
)
async def ingest_ticket(body: TicketIngestRequest, request: Request) -> TicketResponse:
    conversation_id = f"c_{uuid4().hex[:8]}"
    runner = request.app.state.runner
    result = await runner.ingest_ticket(
        conversation_id,
        [m.model_dump(mode="json") for m in body.messages],
        body.customer_metadata.model_dump(),
    )
    return TicketResponse(**result)
