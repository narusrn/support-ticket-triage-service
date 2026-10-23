import pytest
from pydantic import ValidationError

from api.schemas.api import CustomerMetadata, Message, TicketIngestRequest
from api.schemas.decision import Decision, ExtractedInfo, ToolCall


VALID_EXTRACTED = {
    "product_area": "billing",
    "issue_type": "duplicate_charge",
    "sentiment": "angry",
    "language": "en",
    "confidence": 0.9,
    "rationale": "3 charges in 2h",
}


def _decision(**overrides):
    base = {
        "urgency": "critical",
        "extracted": VALID_EXTRACTED,
        "action": "escalate_to_human",
        "rationale": "triple charge, needs human approval",
        "confidence": 0.9,
    }
    base.update(overrides)
    return base


def test_decision_accepts_a_valid_payload():
    d = Decision(**_decision())
    assert d.urgency == "critical"
    assert d.tool_calls == []  # default_factory
    assert d.cited_kb_doc_ids == []


def test_decision_rejects_bad_urgency_enum():
    with pytest.raises(ValidationError):
        Decision(**_decision(urgency="super_urgent"))


def test_decision_rejects_bad_action_enum():
    with pytest.raises(ValidationError):
        Decision(**_decision(action="do_whatever"))


def test_decision_rejects_confidence_above_one():
    with pytest.raises(ValidationError):
        Decision(**_decision(confidence=1.5))


def test_decision_rejects_confidence_below_zero():
    with pytest.raises(ValidationError):
        Decision(**_decision(confidence=-0.1))


def test_decision_requires_rationale():
    base = _decision()
    del base["rationale"]
    with pytest.raises(ValidationError):
        Decision(**base)


def test_extracted_info_rejects_confidence_out_of_range():
    with pytest.raises(ValidationError):
        ExtractedInfo(**{**VALID_EXTRACTED, "confidence": 2.0})


def test_tool_call_rejects_unknown_tool_name():
    with pytest.raises(ValidationError):
        ToolCall(tool_call_id="tc_1", tool_name="delete_everything", args={}, status="pending_approval")


def test_tool_call_rejects_unknown_status():
    with pytest.raises(ValidationError):
        ToolCall(tool_call_id="tc_1", tool_name="issue_refund", args={}, status="maybe_later")


def test_message_rejects_empty_content():
    with pytest.raises(ValidationError):
        Message(role="customer", content="")


def test_message_rejects_unknown_role():
    with pytest.raises(ValidationError):
        Message(role="bot", content="hello")


def test_ticket_ingest_request_rejects_empty_messages_list():
    with pytest.raises(ValidationError):
        TicketIngestRequest(messages=[], customer_metadata=CustomerMetadata(plan="free", tenure_months=1))


def test_ticket_ingest_request_accepts_a_plain_text_string():
    """On request: a caller can POST messages as a single plain-text string -- a raw
    pasted thread -- instead of structuring it into a list of {role, content} objects."""
    req = TicketIngestRequest(messages="Hi, my payment failed and I was charged twice.",
                               customer_metadata=CustomerMetadata(plan="free", tenure_months=1))
    assert len(req.messages) == 1
    assert req.messages[0].role == "customer"
    assert req.messages[0].content == "Hi, my payment failed and I was charged twice."


def test_ticket_ingest_request_rejects_an_empty_plain_text_string():
    with pytest.raises(ValidationError):
        TicketIngestRequest(messages="", customer_metadata=CustomerMetadata(plan="free", tenure_months=1))


def test_customer_metadata_rejects_negative_tenure():
    with pytest.raises(ValidationError):
        CustomerMetadata(plan="free", tenure_months=-1)


def test_customer_metadata_rejects_unknown_plan():
    with pytest.raises(ValidationError):
        CustomerMetadata(plan="ultra", tenure_months=1)
