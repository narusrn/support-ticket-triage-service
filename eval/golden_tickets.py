"""Shared loader for eval/golden_tickets.xlsx -- one row per ticket, flattened
into plain columns (not JSON-in-a-cell) so the file is easy to hand-edit in
Excel. Every message in the current set is customer-only, so the "thread"
column is just those messages newline-joined, one per line; load_golden_tickets
splits it back into one {"role": "customer", ...} dict per non-empty line. This
means a single message can't itself contain a literal newline -- none here do,
but a future multi-line customer message would need a different delimiter.
"""
from pathlib import Path
from typing import Any

import openpyxl

GOLDEN_TICKETS_PATH = Path(__file__).resolve().parent / "golden_tickets.xlsx"

_METADATA_COLUMNS = ("plan", "tenure_months", "prior_tickets", "region", "seats")
_INT_METADATA_COLUMNS = ("tenure_months", "prior_tickets", "seats")


def load_golden_tickets(path: Path = GOLDEN_TICKETS_PATH) -> list[dict[str, Any]]:
    wb = openpyxl.load_workbook(path, read_only=True)
    ws = wb.active
    rows = ws.iter_rows(values_only=True)
    header = next(rows)

    tickets = []
    for row in rows:
        r = dict(zip(header, row))
        if r.get("ticket_id") is None:
            continue

        customer_metadata = {}
        for col in _METADATA_COLUMNS:
            value = r.get(col)
            if value is None or value == "":
                continue
            customer_metadata[col] = int(value) if col in _INT_METADATA_COLUMNS else value

        expected = {
            "urgency": r["expected_urgency"],
            "language": r["expected_language"],
            "action": r["expected_action"],
            "tool_names_requested": [t.strip() for t in (r["expected_tool_names"] or "").split(",") if t.strip()],
            "kb_retriever_invoked": bool(r["expected_kb_retriever_invoked"]),
        }

        ticket = {
            "ticket_id": r["ticket_id"],
            "messages": [{"role": "customer", "content": line} for line in r["thread"].split("\n") if line],
            "customer_metadata": customer_metadata,
            "expected": expected,
            "notes": r.get("notes") or "",
        }
        if r.get("extra_check_not_auto_executed"):
            ticket["extra_checks"] = {"tool_call_must_not_be_auto_executed": True}
        tickets.append(ticket)

    wb.close()
    return tickets


if __name__ == "__main__":
    tickets = load_golden_tickets()
    print(f"loaded {len(tickets)} tickets")
    for t in tickets:
        print(" ", t["ticket_id"], "-", len(t["messages"]), "message(s)")
