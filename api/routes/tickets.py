from uuid import uuid4

from fastapi import APIRouter, HTTPException, Request, status
from pydantic import ValidationError

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
        "anything executes.\n\n"
        "Two ways to call this:\n"
        "- `Content-Type: application/json` -- `{\"messages\": [...] | \"...\", \"customer_metadata\": {...}}` "
        "(`messages` itself can also just be a plain string instead of a list).\n"
        "- `Content-Type: text/plain` -- no JSON wrapping at all, the body IS the ticket text. "
        "`customer_metadata` is then left for the classifier to infer from the text itself."
    ),
    responses={422: {"model": ErrorResponse, "description": "Validation failed, e.g. an empty messages list."}},
    # Found live: once this handler stopped taking `body: TicketIngestRequest` as a typed
    # parameter (needed to read the raw body for the text/plain branch below), Swagger UI
    # lost the editable request-body box entirely -- it had nothing to generate one from.
    # openapi_extra manually restores it, as a content-type dropdown with BOTH shapes.
    # text/plain is listed FIRST -- Swagger UI shows whichever content type comes first
    # as the default, and plain text is the easier/obvious path to lead with, not JSON.
    openapi_extra={
        "requestBody": {
            "required": True,
            "content": {
                "text/plain": {
                    "schema": {"type": "string"},
                    # the spec's own sample Ticket 1 (billing), verbatim -- same thread as
                    # eval/labeled_tickets.json's ticket_1_billing.
                    "example": (
                        "Customer: Free plan (upgrade attempted), 4 months on free tier, first "
                        "time contacting support.\n"
                        "3h ago: \"My payment failed when I tried to upgrade to Pro. Can you check "
                        "what's wrong?\"\n"
                        "2h ago: \"I tried again with a different card. Now I see TWO pending "
                        "charges but my account still shows Free plan??\"\n"
                        "1h ago: \"Okay this is getting ridiculous. Just checked my bank app — I "
                        "have THREE charges of $29.99 now. None refunded. And I STILL don't have "
                        "Pro access.\"\n"
                        "just now: \"HELLO?? Is anyone there??? I need this fixed NOW. I have a "
                        "presentation in 2 hours and need the Pro export features. If these "
                        "charges aren't reversed by end of day I'm disputing all of them with my "
                        "bank.\""
                    ),
                },
                "application/json": {"schema": TicketIngestRequest.model_json_schema()},
            },
        }
    },
)
async def ingest_ticket(request: Request) -> TicketResponse:
    content_type = request.headers.get("content-type", "")
    if content_type.startswith("text/plain"):
        # On request: no JSON wrapping ceremony at all -- same zero-friction shape as
        # scripts/react_playground.py's --paste flow, which also leaves customer_metadata
        # for the classifier to infer from the text rather than asking for it up front.
        raw_text = (await request.body()).decode("utf-8").strip()
        if not raw_text:
            raise HTTPException(status_code=422, detail={
                "code": "validation_error", "message": "request body is empty"})
        messages = [{"role": "customer", "content": raw_text}]
        customer_metadata = {}
    else:
        try:
            body = TicketIngestRequest.model_validate(await request.json())
        except ValidationError as exc:
            raise HTTPException(status_code=422, detail={
                "code": "validation_error", "message": str(exc.errors())})
        messages = [m.model_dump(mode="json") for m in body.messages]
        customer_metadata = body.customer_metadata.model_dump()

    conversation_id = f"c_{uuid4().hex[:8]}"
    runner = request.app.state.runner
    result = await runner.ingest_ticket(conversation_id, messages, customer_metadata)
    return TicketResponse(**result)
