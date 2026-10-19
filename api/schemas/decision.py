from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

from api.schemas.api import Message, UsageSummary


class ExtractedInfo(BaseModel):
    product_area: str = Field(..., description="Which part of the product this concerns, e.g. 'billing'.")
    issue_type: str = Field(..., description="Specific issue category, e.g. 'duplicate_charge'.")
    sentiment: str = Field(..., description="Customer's emotional tone, e.g. 'angry', 'curious'.")
    language: str = Field(..., description="ISO language code of the customer's messages, e.g. 'th'.")
    confidence: float = Field(..., ge=0, le=1, description="triage_classifier's confidence in this classification.")
    rationale: str = Field(..., description="One sentence — *why* it classified the ticket this way.")


class ToolCall(BaseModel):
    tool_call_id: str
    tool_name: Literal["issue_refund", "open_incident_or_page_oncall", "search_knowledge_base"]
    args: dict[str, Any]
    status: Literal["pending_approval", "executed", "rejected"]
    reason: Optional[str] = Field(None, description="Human-provided reason, set on approve/reject.")
    result: Optional[dict[str, Any]] = Field(None, description="Mock tool's return value, once executed.")


class Decision(BaseModel):
    urgency: Literal["critical", "high", "medium", "low"]
    extracted: ExtractedInfo
    action: Literal["auto_respond", "route_to_specialist", "escalate_to_human"]
    draft_reply: Optional[str] = Field(None, description="Ready-to-send reply text — only set when action='auto_respond'.")
    tool_calls: list[ToolCall] = Field(default_factory=list)
    rationale: str = Field(..., description="action_decider's one-sentence reasoning for the chosen action.")
    confidence: float = Field(..., ge=0, le=1)
    cited_kb_doc_ids: list[str] = Field(default_factory=list, description="FAQ doc ids the draft_reply is grounded in — empty unless action='auto_respond'.")


# the two response envelopes every endpoint reuses -- one shape, not a bespoke response per route
class TicketResponse(BaseModel):
    conversation_id: str
    status: Literal["completed", "awaiting_approval"] = Field(
        ..., description="'completed' = graph reached END. 'awaiting_approval' = suspended at interrupt(); "
                          "a tool_call needs POST .../approve or .../reject before anything executes."
    )
    decision: Decision


class ConversationDetail(BaseModel):
    conversation_id: str
    status: Literal["completed", "awaiting_approval"]
    messages: list[Message]
    decisions: list[Decision]
    tool_calls: list[ToolCall]
    usage: UsageSummary
