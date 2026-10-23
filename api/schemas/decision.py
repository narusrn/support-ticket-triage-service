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
    tool_call_id: str = Field(..., description="Stable id derived from sha256(conversation_id:tool_name:sorted_args) -- "
                                                "same id before and after a server restart, so a resumed /approve or "
                                                "/reject call always resolves the SAME tool call, never a new one.")
    tool_name: Literal["issue_refund", "open_incident_or_page_oncall", "search_knowledge_base"] = Field(
        ..., description="Only issue_refund/open_incident_or_page_oncall ever reach pending_approval -- "
                          "search_knowledge_base is read-only and executes immediately, never gated.")
    args: dict[str, Any] = Field(..., description="Arguments the agent proposed, e.g. "
                                                    "{'charge_ids': ['ch_1'], 'amount_usd': 29.99} for issue_refund.")
    status: Literal["pending_approval", "executed", "rejected"] = Field(
        ..., description="pending_approval = waiting on POST .../approve or .../reject. executed = the mock side "
                          "effect actually ran (only reachable via approve). rejected = approve/reject resolved it "
                          "without running anything.")
    reason: Optional[str] = Field(None, description="Human-provided reason, set on approve/reject.")
    result: Optional[dict[str, Any]] = Field(None, description="Mock tool's return value, once executed -- "
                                                                  "e.g. {'refund_id': 're_123', 'status': 'succeeded'}.")


class Decision(BaseModel):
    urgency: Literal["critical", "high", "medium", "low"] = Field(
        ..., description="triage_classifier's call, weighing message content AND customer_metadata "
                          "(plan tier, seat count, tenure) -- not word choice alone.")
    extracted: ExtractedInfo
    action: Literal["auto_respond", "route_to_specialist", "escalate_to_human"] = Field(
        ..., description="auto_respond: draft_reply is ready to send, grounded in cited_kb_doc_ids. "
                          "route_to_specialist: not confident/groundable enough to answer directly, no tool "
                          "call either. escalate_to_human: needs a human now, often (not always) paired with "
                          "a pending tool_call.")
    draft_reply: Optional[str] = Field(None, description="Ready-to-send reply text — only set when action='auto_respond'.")
    tool_calls: list[ToolCall] = Field(default_factory=list, description="Empty unless action_decider proposed a "
                                                                           "side-effecting tool for this ticket.")
    rationale: str = Field(..., description="action_decider's one-sentence reasoning for the chosen action.")
    confidence: float = Field(..., ge=0, le=1, description="action_decider's self-reported confidence. Below "
                                                              "AUTO_RESPOND_CONFIDENCE_THRESHOLD (default 0.75), "
                                                              "auto_respond is downgraded to route_to_specialist "
                                                              "in code regardless of what the raw output said.")
    cited_kb_doc_ids: list[str] = Field(default_factory=list, description="FAQ doc ids the draft_reply is grounded in — empty unless action='auto_respond'.")


# the two response envelopes every endpoint reuses -- one shape, not a bespoke response per route
class TicketResponse(BaseModel):
    conversation_id: str
    status: Literal["completed", "awaiting_approval"] = Field(
        ..., description="'completed' = graph reached END. 'awaiting_approval' = suspended at interrupt(); "
                          "a tool_call needs POST .../approve or .../reject before anything executes."
    )
    decision: Decision
    reply: Optional[str] = Field(
        None, description="The agent's conversational reply text -- only set by "
                           "POST .../messages (the advisory endpoint); absent elsewhere."
    )


class ConversationDetail(BaseModel):
    conversation_id: str
    status: Literal["completed", "awaiting_approval"]
    messages: list[Message] = Field(..., description="Full thread in order: the original customer messages, "
                                                         "every agent dialogue line (Supervisor/Triage Classifier/"
                                                         "etc.), and any operator follow-ups via POST .../messages.")
    decisions: list[Decision] = Field(..., description="Usually just one entry -- a new row is only added when a "
                                                          "decision FIELD actually changes, not on every resume "
                                                          "(approving a tool call doesn't create a duplicate).")
    tool_calls: list[ToolCall] = Field(..., description="Every tool call ever proposed for this conversation, "
                                                           "regardless of which decision it belongs to.")
    usage: UsageSummary
