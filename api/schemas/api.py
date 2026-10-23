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
    total_runs: int = Field(..., description="One run per graph invocation for this conversation -- the initial "
                                               "ingest, plus one more for every approve/reject resume.")
    total_tokens: int = Field(..., description="Summed across every LLM call (supervisor + specialists) in every "
                                                  "run, real usage from the OpenAI response, not an estimate.")
    cost_usd: float = Field(..., description="Priced from total_tokens against config.py's MODEL_PRICES_USD_PER_1K.")
    avg_latency_ms: float = Field(..., description="Average wall-clock duration per run (ingest or resume), in ms.")


class TimeseriesPoint(BaseModel):
    bucket: str = Field(..., description="Hour bucket, ISO 8601 (e.g. '2026-10-23T14:00:00').")
    runs: int = Field(..., description="Number of graph invocations that started in this hour.")
    tokens: int = Field(..., description="Total tokens consumed by runs that started in this hour.")


class MetricsResponse(BaseModel):
    """Service-wide rollup for GET /metrics and the dashboard built on it --
    mirrors storage.db.get_global_metrics()'s shape exactly. All totals are
    per RUN (one graph invocation), not per conversation -- a ticket that
    gets approved later contributes 2 runs (ingest + resume) to these totals,
    not 1, since each is a real invocation with its own cost/latency."""
    total_conversations: int = Field(..., description="Distinct tickets ever ingested via POST /tickets.")
    total_runs: int = Field(..., description="Every graph invocation across every conversation -- "
                                               "ingests plus approve/reject resumes.")
    total_tokens: int = Field(..., description="Real token usage summed across every run, service-wide.")
    total_cost_usd: float = Field(..., description="Priced from total_tokens against MODEL_PRICES_USD_PER_1K.")
    avg_latency_ms: float = Field(..., description="Average wall-clock duration per run, service-wide.")
    action_breakdown: dict[str, int] = Field(..., description="Final decision.action, counted per run -- "
                                                                 "e.g. {'auto_respond': 3, 'escalate_to_human': 2}.")
    urgency_breakdown: dict[str, int] = Field(..., description="Final decision.urgency, counted per run -- "
                                                                  "e.g. {'critical': 2, 'low': 3}.")
    tool_call_status_breakdown: dict[str, int] = Field(..., description="Current status of every tool call ever "
                                                                          "proposed, service-wide -- e.g. "
                                                                          "{'pending_approval': 1, 'executed': 4}.")
    timeseries: list[TimeseriesPoint] = Field(..., description="Requests/tokens per hour, for the dashboard's graph.")
