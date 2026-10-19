import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from config import settings

SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (
    conversation_id TEXT PRIMARY KEY,
    status TEXT NOT NULL,
    customer_metadata TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id TEXT NOT NULL REFERENCES conversations(conversation_id),
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    sent_at TEXT
);

CREATE TABLE IF NOT EXISTS decisions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id TEXT NOT NULL REFERENCES conversations(conversation_id),
    urgency TEXT NOT NULL,
    extracted TEXT NOT NULL,
    action TEXT NOT NULL,
    draft_reply TEXT,
    rationale TEXT NOT NULL,
    confidence REAL NOT NULL,
    cited_kb_doc_ids TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS tool_calls (
    tool_call_id TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL REFERENCES conversations(conversation_id),
    tool_name TEXT NOT NULL,
    args TEXT NOT NULL,
    status TEXT NOT NULL,
    idempotency_key TEXT UNIQUE,
    reason TEXT,
    result TEXT,
    created_at TEXT NOT NULL,
    resolved_at TEXT
);

CREATE TABLE IF NOT EXISTS trace_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    conversation_id TEXT NOT NULL,
    step_index INTEGER NOT NULL,
    node TEXT NOT NULL,
    event_type TEXT NOT NULL,
    payload TEXT,
    model TEXT,
    latency_ms INTEGER,
    tokens TEXT,
    timestamp TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    duration_ms INTEGER,
    steps INTEGER,
    specialists_called TEXT,
    tokens TEXT,
    cost_usd REAL,
    models_used TEXT,
    final_urgency TEXT,
    final_action TEXT,
    pending_approvals TEXT,
    completed INTEGER NOT NULL DEFAULT 0,
    stopped_reason TEXT,
    error TEXT
);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def get_connection(db_path: Optional[str] = None) -> sqlite3.Connection:
    path = db_path or settings.DATABASE_PATH
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    return conn


def init_db(db_path: Optional[str] = None) -> None:
    conn = get_connection(db_path)
    try:
        conn.executescript(SCHEMA)
        conn.commit()
    finally:
        conn.close()


def ping(db_path: Optional[str] = None) -> bool:
    try:
        conn = get_connection(db_path)
        conn.execute("SELECT 1")
        conn.close()
        return True
    except sqlite3.Error:
        return False


# --- conversations ---

def insert_conversation(conn: sqlite3.Connection, conversation_id: str, status: str, customer_metadata: dict) -> None:
    now = _now()
    conn.execute(
        "INSERT INTO conversations (conversation_id, status, customer_metadata, created_at, updated_at) "
        "VALUES (?, ?, ?, ?, ?)",
        (conversation_id, status, json.dumps(customer_metadata), now, now),
    )
    conn.commit()


def get_conversation(conn: sqlite3.Connection, conversation_id: str) -> Optional[dict]:
    row = conn.execute("SELECT * FROM conversations WHERE conversation_id = ?", (conversation_id,)).fetchone()
    if row is None:
        return None
    d = dict(row)
    d["customer_metadata"] = json.loads(d["customer_metadata"]) if d["customer_metadata"] else {}
    return d


def update_conversation_status(conn: sqlite3.Connection, conversation_id: str, status: str) -> None:
    conn.execute(
        "UPDATE conversations SET status = ?, updated_at = ? WHERE conversation_id = ?",
        (status, _now(), conversation_id),
    )
    conn.commit()


# --- messages ---

def insert_message(conn: sqlite3.Connection, conversation_id: str, role: str, content: str,
                    sent_at: Optional[str] = None) -> int:
    cur = conn.execute(
        "INSERT INTO messages (conversation_id, role, content, sent_at) VALUES (?, ?, ?, ?)",
        (conversation_id, role, content, sent_at or _now()),
    )
    conn.commit()
    return cur.lastrowid


def get_messages(conn: sqlite3.Connection, conversation_id: str) -> list[dict]:
    rows = conn.execute("SELECT * FROM messages WHERE conversation_id = ? ORDER BY id", (conversation_id,)).fetchall()
    return [dict(r) for r in rows]


# --- decisions ---

def insert_decision(conn: sqlite3.Connection, conversation_id: str, decision: dict) -> int:
    cur = conn.execute(
        "INSERT INTO decisions (conversation_id, urgency, extracted, action, draft_reply, "
        "rationale, confidence, cited_kb_doc_ids, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            conversation_id,
            decision["urgency"],
            json.dumps(decision["extracted"]),
            decision["action"],
            decision.get("draft_reply"),
            decision["rationale"],
            decision["confidence"],
            json.dumps(decision.get("cited_kb_doc_ids", [])),
            _now(),
        ),
    )
    conn.commit()
    return cur.lastrowid


def get_decisions(conn: sqlite3.Connection, conversation_id: str) -> list[dict]:
    rows = conn.execute("SELECT * FROM decisions WHERE conversation_id = ? ORDER BY id", (conversation_id,)).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        d["extracted"] = json.loads(d["extracted"])
        d["cited_kb_doc_ids"] = json.loads(d["cited_kb_doc_ids"])
        out.append(d)
    return out


# --- tool_calls ---

def insert_tool_call(conn: sqlite3.Connection, tool_call_id: str, conversation_id: str, tool_name: str,
                      args: dict, status: str, idempotency_key: Optional[str] = None) -> None:
    conn.execute(
        "INSERT INTO tool_calls (tool_call_id, conversation_id, tool_name, args, status, "
        "idempotency_key, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (tool_call_id, conversation_id, tool_name, json.dumps(args), status, idempotency_key, _now()),
    )
    conn.commit()


def get_tool_call(conn: sqlite3.Connection, tool_call_id: str) -> Optional[dict]:
    row = conn.execute("SELECT * FROM tool_calls WHERE tool_call_id = ?", (tool_call_id,)).fetchone()
    if row is None:
        return None
    d = dict(row)
    d["args"] = json.loads(d["args"])
    d["result"] = json.loads(d["result"]) if d["result"] else None
    return d


def get_tool_calls(conn: sqlite3.Connection, conversation_id: str) -> list[dict]:
    rows = conn.execute(
        "SELECT * FROM tool_calls WHERE conversation_id = ? ORDER BY created_at", (conversation_id,)
    ).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        d["args"] = json.loads(d["args"])
        d["result"] = json.loads(d["result"]) if d["result"] else None
        out.append(d)
    return out


def update_tool_call(conn: sqlite3.Connection, tool_call_id: str, status: str,
                      reason: Optional[str] = None, result: Optional[dict] = None) -> None:
    conn.execute(
        "UPDATE tool_calls SET status = ?, reason = ?, result = ?, resolved_at = ? WHERE tool_call_id = ?",
        (status, reason, json.dumps(result) if result is not None else None, _now(), tool_call_id),
    )
    conn.commit()


# --- trace_events ---

def insert_trace_event(conn: sqlite3.Connection, run_id: str, conversation_id: str, step_index: int,
                        node: str, event_type: str, payload: Optional[dict] = None,
                        model: Optional[str] = None, latency_ms: Optional[int] = None,
                        tokens: Optional[dict] = None) -> None:
    conn.execute(
        "INSERT INTO trace_events (run_id, conversation_id, step_index, node, event_type, payload, "
        "model, latency_ms, tokens, timestamp) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            run_id, conversation_id, step_index, node, event_type,
            json.dumps(payload) if payload is not None else None,
            model, latency_ms,
            json.dumps(tokens) if tokens is not None else None,
            _now(),
        ),
    )
    conn.commit()


def get_trace_events(conn: sqlite3.Connection, conversation_id: str) -> list[dict]:
    rows = conn.execute(
        "SELECT * FROM trace_events WHERE conversation_id = ? ORDER BY id", (conversation_id,)
    ).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        d["payload"] = json.loads(d["payload"]) if d["payload"] else None
        d["tokens"] = json.loads(d["tokens"]) if d["tokens"] else None
        out.append(d)
    return out


# --- runs ---

def insert_run(conn: sqlite3.Connection, run: dict) -> None:
    conn.execute(
        "INSERT INTO runs (run_id, conversation_id, started_at, finished_at, duration_ms, steps, "
        "specialists_called, tokens, cost_usd, models_used, final_urgency, final_action, "
        "pending_approvals, completed, stopped_reason, error) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            run["run_id"], run["conversation_id"], run["started_at"], run.get("finished_at"),
            run.get("duration_ms"), run.get("steps"),
            json.dumps(run.get("specialists_called", [])),
            json.dumps(run.get("tokens", {})),
            run.get("cost_usd"),
            json.dumps(run.get("models_used", {})),
            run.get("final_urgency"), run.get("final_action"),
            json.dumps(run.get("pending_approvals", [])),
            int(run.get("completed", False)),
            run.get("stopped_reason"), run.get("error"),
        ),
    )
    conn.commit()


def get_runs(conn: sqlite3.Connection, conversation_id: str) -> list[dict]:
    rows = conn.execute("SELECT * FROM runs WHERE conversation_id = ? ORDER BY started_at", (conversation_id,)).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        d["specialists_called"] = json.loads(d["specialists_called"]) if d["specialists_called"] else []
        d["tokens"] = json.loads(d["tokens"]) if d["tokens"] else {}
        d["models_used"] = json.loads(d["models_used"]) if d["models_used"] else {}
        d["pending_approvals"] = json.loads(d["pending_approvals"]) if d["pending_approvals"] else []
        d["completed"] = bool(d["completed"])
        out.append(d)
    return out


def get_usage_summary(conn: sqlite3.Connection, conversation_id: str) -> dict:
    row = conn.execute(
        "SELECT COUNT(*) as total_runs, "
        "COALESCE(SUM(json_extract(tokens, '$.total')), 0) as total_tokens, "
        "COALESCE(SUM(cost_usd), 0) as cost_usd, "
        "COALESCE(AVG(duration_ms), 0) as avg_latency_ms "
        "FROM runs WHERE conversation_id = ?",
        (conversation_id,),
    ).fetchone()
    return dict(row)
