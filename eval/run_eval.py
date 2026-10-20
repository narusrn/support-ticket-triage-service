"""Offline eval harness: runs the real agent (real OpenAI calls) over
eval/labeled_tickets.json and reports per-field accuracy, plus an
LLM-as-judge pass over any auto_respond draft replies.

Run: python eval/run_eval.py
"""
import asyncio
import json
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pydantic import BaseModel, Field
from langgraph.checkpoint.memory import InMemorySaver

from agents.kb import FAQ_DIR
from agents.llm import call_with_retry, make_llm_client
from agents.runner import Runner
from config import settings
from storage import db

LABELED_TICKETS_PATH = Path(__file__).resolve().parent / "labeled_tickets.json"
FIELDS = ["urgency", "language", "action", "tool_names_requested", "kb_retriever_invoked"]


class JudgeVerdict(BaseModel):
    grounded: bool = Field(..., description="Does the reply stay within what the cited FAQ docs actually say, with no fabricated policy?")
    tone_appropriate: bool = Field(..., description="Is the tone reasonable for this ticket's urgency and sentiment?")
    notes: str = Field(..., description="One short sentence explaining the verdict.")


def judge_auto_respond(draft_reply: str, cited_kb_doc_ids: list[str]) -> dict:
    docs_text = "\n\n".join(
        f"=== {doc_id} ===\n{(FAQ_DIR / f'{doc_id}.md').read_text(encoding='utf-8')}"
        for doc_id in cited_kb_doc_ids if (FAQ_DIR / f"{doc_id}.md").exists()
    ) or "(no docs cited)"

    llm = make_llm_client(settings.OPENAI_MODEL).with_structured_output(JudgeVerdict, method="function_calling")
    result = call_with_retry(lambda: llm.invoke([
        {"role": "system", "content": "You are grading a support agent's draft reply for groundedness and tone. "
                                       "Only the cited FAQ docs below are allowed sources -- flag anything the "
                                       "reply asserts that isn't actually in them."},
        {"role": "user", "content": f"Cited docs:\n{docs_text}\n\nDraft reply:\n{draft_reply}"},
    ]))
    return result.model_dump()


def check_ticket(conn, ticket: dict, result: dict) -> dict:
    decision = result["decision"]
    expected = ticket["expected"]
    checks = {}

    checks["urgency"] = {"expected": expected["urgency"], "actual": decision.get("urgency"),
                          "pass": decision.get("urgency") == expected["urgency"]}

    actual_language = (decision.get("extracted") or {}).get("language")
    checks["language"] = {"expected": expected["language"], "actual": actual_language,
                           "pass": actual_language == expected["language"]}

    checks["action"] = {"expected": expected["action"], "actual": decision.get("action"),
                         "pass": decision.get("action") == expected["action"]}

    actual_tools = sorted(tc["tool_name"] for tc in decision.get("tool_calls", []))
    expected_tools = sorted(expected["tool_names_requested"])
    checks["tool_names_requested"] = {"expected": expected_tools, "actual": actual_tools,
                                       "pass": actual_tools == expected_tools}

    trace_events = db.get_trace_events(conn, result["conversation_id"])
    kb_invoked = any(e["node"] == "call_kb_retriever" for e in trace_events)
    checks["kb_retriever_invoked"] = {"expected": expected["kb_retriever_invoked"], "actual": kb_invoked,
                                       "pass": kb_invoked == expected["kb_retriever_invoked"]}

    extra = ticket.get("extra_checks", {})
    if extra.get("tool_call_must_not_be_auto_executed"):
        statuses = [tc["status"] for tc in decision.get("tool_calls", [])]
        not_auto_executed = all(s != "executed" for s in statuses) if statuses else True
        checks["tool_call_must_not_be_auto_executed"] = {
            "expected": True, "actual": not_auto_executed, "pass": not_auto_executed,
            "note": "autonomy boundary: a tool_call must never come back already 'executed' straight out of ingest",
        }

    return checks


async def run_one(ticket: dict) -> dict:
    db_path = f"eval_run_{uuid.uuid4().hex[:8]}.db"
    db.init_db(db_path)
    conn = db.get_connection(db_path)
    runner = Runner(conn, InMemorySaver())  # real default brain functions -- real OpenAI calls

    result = await runner.ingest_ticket(ticket["ticket_id"], ticket["messages"], ticket["customer_metadata"])
    checks = check_ticket(conn, ticket, result)

    # Only genuine KB-grounded auto_respond decisions (from action_decider) get judged.
    # A direct_reply greeting is also synthesized as action="auto_respond" by the supervisor
    # (see agents/orchestrator.py) but cites no docs -- there's nothing to check groundedness
    # against, so judging it produces a meaningless "not grounded" false positive.
    judge = None
    decision = result["decision"]
    if decision.get("action") == "auto_respond" and decision.get("draft_reply") and decision.get("cited_kb_doc_ids"):
        judge = judge_auto_respond(decision["draft_reply"], decision["cited_kb_doc_ids"])

    conn.close()
    Path(db_path).unlink(missing_ok=True)
    for suffix in ("-shm", "-wal"):
        Path(db_path + suffix).unlink(missing_ok=True)

    return {"ticket_id": ticket["ticket_id"], "checks": checks,
            "all_pass": all(c["pass"] for c in checks.values()), "judge": judge}


async def main():
    tickets = json.loads(LABELED_TICKETS_PATH.read_text(encoding="utf-8"))
    results = []
    for ticket in tickets:
        outcome = await run_one(ticket)
        print(json.dumps(outcome, indent=2, ensure_ascii=False))
        results.append(outcome)

    summary = {"n": len(results)}
    for field in FIELDS:
        relevant = [r["checks"][field]["pass"] for r in results if field in r["checks"]]
        summary[f"{field}_accuracy"] = round(sum(relevant) / len(relevant), 3) if relevant else None
    summary["per_ticket_all_pass_rate"] = round(sum(r["all_pass"] for r in results) / len(results), 3)
    judged = [r for r in results if r["judge"] is not None]
    if judged:
        summary["judge_grounded_rate"] = round(sum(r["judge"]["grounded"] for r in judged) / len(judged), 3)

    print(json.dumps({"summary": summary}, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
