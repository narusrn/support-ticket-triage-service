from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class Message(BaseModel):
    role: Literal["customer", "operator", "agent"] = Field(..., description="Who sent this message in the thread.")
    content: str = Field(..., min_length=1, description="The raw message text.")
    sent_at: Optional[datetime] = Field(None, description="Original send time — metadata that matters for triage per the spec.")


class CustomerMetadata(BaseModel):
    plan: Literal["free", "pro", "enterprise"] = Field(..., description="Customer's subscription tier.")
    tenure_months: int = Field(..., ge=0, description="How many months they've been a customer.")
    prior_tickets: int = Field(0, ge=0, description="Number of previous support tickets filed.")
    region: Optional[str] = Field(None, description="Customer's region, e.g. 'thailand' — relevant for outage scope.")
    seats: Optional[int] = Field(None, ge=1, description="Licensed seat count, for enterprise accounts.")


class TicketIngestRequest(BaseModel):
    messages: list[Message] = Field(
        ..., min_length=1,
        description="The full thread exactly as it already existed before this system saw it -- "
                    "either a list of {role, content} messages, or a single plain-text string "
                    "(e.g. a raw pasted email/chat thread), treated as one customer message."
    )
    customer_metadata: CustomerMetadata

    @field_validator("messages", mode="before")
    @classmethod
    def _coerce_plain_text(cls, value):
        # scripts/chat.py --paste already does this client-side (wraps raw pasted text
        # into one {"role": "customer", "content": ...} message before POSTing) -- moved
        # into the schema itself so any caller can just send a string, not only that script.
        if isinstance(value, str):
            return [{"role": "customer", "content": value}]
        return value

    model_config = ConfigDict(json_schema_extra={
        "example": {
            "messages": [
                {"role": "customer", "content": "My payment failed when I tried to upgrade to Pro. Can you check what's wrong?"},
                {"role": "customer", "content": "I tried again with a different card. Now I see TWO pending charges but my account still shows Free plan??"},
                {"role": "customer", "content": "Okay this is getting ridiculous. Just checked my bank app — I have THREE charges of $29.99 now. None refunded. And I STILL don't have Pro access."},
                {"role": "customer", "content": "HELLO?? Is anyone there??? I need this fixed NOW. I have a presentation in 2 hours and need the Pro export features. If these charges aren't reversed by end of day I'm disputing all of them with my bank."},
            ],
            "customer_metadata": {"plan": "free", "tenure_months": 4, "prior_tickets": 0},
        }
    })


class OperatorMessageRequest(BaseModel):
    message: str = Field(..., min_length=1, description="Free-text operator turn, e.g. a follow-up question.")

    model_config = ConfigDict(json_schema_extra={"example": {"message": "why is this urgent?"}})


class ApprovalRequest(BaseModel):
    reason: Optional[str] = Field(None, description="Why the operator approved or rejected this tool call — stored in the audit trail, surfaced via GET.")

    model_config = ConfigDict(json_schema_extra={"example": {"reason": "verified in Stripe dashboard, confirmed duplicate"}})


class ErrorDetail(BaseModel):
    code: str = Field(..., description="Machine-readable error code, e.g. 'conversation_not_found'.")
    message: str = Field(..., description="Human-readable explanation.")


class ErrorResponse(BaseModel):
    error: ErrorDetail


class UsageSummary(BaseModel):
    total_runs: int
    total_tokens: int
    cost_usd: float
    avg_latency_ms: float
