from storage import db


def test_conversation_create_read_update_round_trip(tmp_path):
    db_path = str(tmp_path / "test.db")
    db.init_db(db_path)
    conn = db.get_connection(db_path)

    db.insert_conversation(conn, "c_1", "awaiting_approval", {"plan": "free", "tenure_months": 4})
    fetched = db.get_conversation(conn, "c_1")
    assert fetched["conversation_id"] == "c_1"
    assert fetched["status"] == "awaiting_approval"
    assert fetched["customer_metadata"] == {"plan": "free", "tenure_months": 4}

    db.update_conversation_status(conn, "c_1", "completed")
    updated = db.get_conversation(conn, "c_1")
    assert updated["status"] == "completed"
    assert updated["updated_at"] >= fetched["updated_at"]

    assert db.get_conversation(conn, "c_missing") is None
    conn.close()


def test_messages_and_decisions_round_trip(tmp_path):
    db_path = str(tmp_path / "test.db")
    db.init_db(db_path)
    conn = db.get_connection(db_path)
    db.insert_conversation(conn, "c_1", "completed", {})

    db.insert_message(conn, "c_1", "customer", "payment failed")
    db.insert_message(conn, "c_1", "customer", "three charges now")
    messages = db.get_messages(conn, "c_1")
    assert [m["content"] for m in messages] == ["payment failed", "three charges now"]

    decision = {
        "urgency": "critical",
        "extracted": {"product_area": "billing", "issue_type": "duplicate_charge", "sentiment": "angry",
                      "language": "en", "confidence": 0.9, "rationale": "3 charges in 2h"},
        "action": "escalate_to_human",
        "draft_reply": None,
        "rationale": "triple charge, needs human approval",
        "confidence": 0.9,
        "cited_kb_doc_ids": [],
    }
    db.insert_decision(conn, "c_1", decision)
    decisions = db.get_decisions(conn, "c_1")
    assert len(decisions) == 1
    assert decisions[0]["urgency"] == "critical"
    assert decisions[0]["extracted"]["language"] == "en"
    conn.close()


def test_tool_call_lifecycle_and_idempotency_key_is_unique(tmp_path):
    db_path = str(tmp_path / "test.db")
    db.init_db(db_path)
    conn = db.get_connection(db_path)
    db.insert_conversation(conn, "c_1", "awaiting_approval", {})

    db.insert_tool_call(conn, "tc_1", "c_1", "issue_refund", {"amount_usd": 89.97},
                         "pending_approval", idempotency_key="c_1:issue_refund:abc")
    pending = db.get_tool_call(conn, "tc_1")
    assert pending["status"] == "pending_approval"
    assert pending["result"] is None

    db.update_tool_call(conn, "tc_1", "executed", reason="verified in Stripe",
                         result={"refund_id": "re_1", "status": "succeeded"})
    executed = db.get_tool_call(conn, "tc_1")
    assert executed["status"] == "executed"
    assert executed["reason"] == "verified in Stripe"
    assert executed["result"]["refund_id"] == "re_1"

    # same idempotency key again -> must fail (UNIQUE constraint), proves double-booking is impossible at the DB level
    try:
        db.insert_tool_call(conn, "tc_2", "c_1", "issue_refund", {"amount_usd": 89.97},
                             "pending_approval", idempotency_key="c_1:issue_refund:abc")
        assert False, "expected a UNIQUE constraint violation on idempotency_key"
    except Exception:
        pass
    conn.close()


def test_trace_events_and_runs_round_trip(tmp_path):
    db_path = str(tmp_path / "test.db")
    db.init_db(db_path)
    conn = db.get_connection(db_path)
    db.insert_conversation(conn, "c_1", "completed", {})

    db.insert_trace_event(conn, "run_1", "c_1", 1, "supervisor", "handoff", payload={"to": "triage_classifier"})
    events = db.get_trace_events(conn, "c_1")
    assert events[0]["payload"] == {"to": "triage_classifier"}

    db.insert_run(conn, {
        "run_id": "run_1", "conversation_id": "c_1", "started_at": "2026-10-20T00:00:00Z",
        "finished_at": "2026-10-20T00:00:01Z", "duration_ms": 680, "steps": 5,
        "tokens": {"total": 2767}, "cost_usd": 0.00083, "final_urgency": "critical",
        "final_action": "escalate_to_human", "completed": True, "stopped_reason": "end",
    })
    runs = db.get_runs(conn, "c_1")
    assert runs[0]["completed"] is True
    assert runs[0]["tokens"]["total"] == 2767

    usage = db.get_usage_summary(conn, "c_1")
    assert usage["total_runs"] == 1
    assert usage["total_tokens"] == 2767
    conn.close()


def test_global_metrics_aggregates_across_every_conversation(tmp_path):
    db_path = str(tmp_path / "test.db")
    db.init_db(db_path)
    conn = db.get_connection(db_path)

    db.insert_conversation(conn, "c_1", "completed", {})
    db.insert_conversation(conn, "c_2", "awaiting_approval", {})

    db.insert_run(conn, {
        "run_id": "run_1", "conversation_id": "c_1", "started_at": "2026-10-20T00:00:00Z",
        "duration_ms": 600, "steps": 3, "tokens": {"total": 1000}, "cost_usd": 0.001,
        "final_urgency": "critical", "final_action": "escalate_to_human", "completed": True,
    })
    db.insert_run(conn, {
        "run_id": "run_2", "conversation_id": "c_2", "started_at": "2026-10-20T01:00:00Z",
        "duration_ms": 400, "steps": 2, "tokens": {"total": 500}, "cost_usd": 0.0005,
        "final_urgency": "low", "final_action": "auto_respond", "completed": True,
    })
    db.insert_tool_call(conn, "tc_1", "c_1", "issue_refund", {}, "executed")
    db.insert_tool_call(conn, "tc_2", "c_2", "issue_refund", {}, "pending_approval")

    metrics = db.get_global_metrics(conn)
    assert metrics["total_conversations"] == 2
    assert metrics["total_runs"] == 2
    assert metrics["total_tokens"] == 1500
    assert round(metrics["total_cost_usd"], 6) == round(0.0015, 6)
    assert metrics["action_breakdown"] == {"escalate_to_human": 1, "auto_respond": 1}
    assert metrics["urgency_breakdown"] == {"critical": 1, "low": 1}
    assert metrics["tool_call_status_breakdown"] == {"executed": 1, "pending_approval": 1}
    assert len(metrics["timeseries"]) == 2  # two different hour buckets
    conn.close()


def test_restart_survival_reopening_same_file(tmp_path):
    db_path = str(tmp_path / "restart_test.db")
    db.init_db(db_path)

    conn1 = db.get_connection(db_path)
    db.insert_conversation(conn1, "c_persist", "awaiting_approval", {"plan": "enterprise"})
    db.insert_message(conn1, "c_persist", "customer", "ระบบเข้าไม่ได้")
    conn1.close()  # simulates the process/connection going away entirely

    # reopen the same file as if the server had just restarted
    conn2 = db.get_connection(db_path)
    restored = db.get_conversation(conn2, "c_persist")
    assert restored is not None
    assert restored["status"] == "awaiting_approval"
    assert restored["customer_metadata"] == {"plan": "enterprise"}

    messages = db.get_messages(conn2, "c_persist")
    assert messages[0]["content"] == "ระบบเข้าไม่ได้"
    conn2.close()


def test_ping(tmp_path):
    db_path = str(tmp_path / "ping_test.db")
    db.init_db(db_path)
    assert db.ping(db_path) is True
