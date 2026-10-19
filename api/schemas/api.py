from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field


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
    messages: list[Message] = Field(..., min_length=1, description="The full thread exactly as it already existed before this system saw it.")
    customer_metadata: CustomerMetadata


class OperatorMessageRequest(BaseModel):
    message: str = Field(..., min_length=1, description="Free-text operator turn, e.g. a follow-up question.")


class ApprovalRequest(BaseModel):
    reason: Optional[str] = Field(None, description="Why the operator approved or rejected this tool call — stored in the audit trail, surfaced via GET.")


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
