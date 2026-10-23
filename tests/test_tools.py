import pytest

from agents import tools
from storage import db


def _setup(tmp_path):
    db_path = str(tmp_path / "test.db")
    db.init_db(db_path)
    conn = db.get_connection(db_path)
    db.insert_conversation(conn, "c_1", "awaiting_approval", {})
    return conn


def test_dispatch_tool_call_executes_issue_refund_once(tmp_path):
    conn = _setup(tmp_path)
    db.insert_tool_call(conn, "tc_1", "c_1", "issue_refund", {"charge_ids": ["ch_1", "ch_2"], "amount_usd": 59.98},
                         "pending_approval", idempotency_key="c_1:issue_refund:abc")

    result = tools.dispatch_tool_call(conn, "tc_1", reason="verified in Stripe")
    assert result["status"] == "succeeded"
    assert result["amount_usd"] == 59.98

    stored = db.get_tool_call(conn, "tc_1")
    assert stored["status"] == "executed"
    assert stored["reason"] == "verified in Stripe"


def test_dispatch_tool_call_is_idempotent_no_double_refund(tmp_path):
    conn = _setup(tmp_path)
    db.insert_tool_call(conn, "tc_1", "c_1", "issue_refund", {"charge_ids": ["ch_1"], "amount_usd": 29.99},
                         "pending_approval", idempotency_key="c_1:issue_refund:xyz")

    first = tools.dispatch_tool_call(conn, "tc_1")
    second = tools.dispatch_tool_call(conn, "tc_1")

    # same refund_id both times -- proves the mock tool only actually ran once
    assert first["refund_id"] == second["refund_id"]


def test_dispatch_tool_call_executes_open_incident(tmp_path):
    conn = _setup(tmp_path)
    db.insert_tool_call(conn, "tc_2", "c_1", "open_incident_or_page_oncall",
                         {"summary": "Asia region outage", "severity": "critical"}, "pending_approval")

    result = tools.dispatch_tool_call(conn, "tc_2")
    assert result["status"] == "paged"
    assert result["severity"] == "critical"


def test_dispatch_tool_call_simulated_failure_marks_rejected(tmp_path):
    conn = _setup(tmp_path)
    db.insert_tool_call(conn, "tc_3", "c_1", "issue_refund",
                         {"charge_ids": ["ch_1"], "amount_usd": 10.0, "simulate_failure": True}, "pending_approval")

    with pytest.raises(tools.ToolExecutionError):
        tools.dispatch_tool_call(conn, "tc_3")

    stored = db.get_tool_call(conn, "tc_3")
    assert stored["status"] == "rejected"
    assert "payment_gateway_timeout" in stored["reason"]


def test_dispatch_tool_call_raises_for_unknown_tool_call_id(tmp_path):
    conn = _setup(tmp_path)
    with pytest.raises(ValueError):
        tools.dispatch_tool_call(conn, "tc_does_not_exist")
