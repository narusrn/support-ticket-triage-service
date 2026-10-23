"""Feeds eval/labeled_tickets.json through the REAL, running HTTP API (not the
in-process Runner eval/run_eval.py uses) so the service's SQLite DB ends up with
real conversations/runs/tool_calls -- i.e. something to actually see in
GET /api/v1/dashboard and /metrics, not an empty service.

For every pending_approval tool call this produces, approves most and rejects
one (whichever comes last) so the dashboard's tool-call-status breakdown shows
a realistic mix instead of every single one landing the same way.

Usage:
  python scripts/seed_demo_data.py
  python scripts/seed_demo_data.py --base-url http://127.0.0.1:8000/api/v1 --api-key ...
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import requests

from scripts.chat import _die_on_error

LABELED_TICKETS = Path(__file__).resolve().parent.parent / "eval" / "labeled_tickets.json"


def main() -> None:
    for stream in (sys.stdin, sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except AttributeError:
            pass

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/api/v1")
    parser.add_argument("--api-key", default=None, help="X-API-Key header; omit if the server has none configured")
    args = parser.parse_args()
    headers = {"X-API-Key": args.api_key} if args.api_key else {}

    tickets = json.loads(LABELED_TICKETS.read_text(encoding="utf-8"))
    pending: list[tuple[str, str]] = []  # (conversation_id, tool_call_id)

    for ticket in tickets:
        body = {"messages": ticket["messages"], "customer_metadata": ticket["customer_metadata"]}
        data = _die_on_error(requests.post(f"{args.base_url}/tickets", json=body, headers=headers))
        conversation_id = data["conversation_id"]
        tool_calls = data["decision"].get("tool_calls", [])
        print(f"-- {ticket['ticket_id']}: status={data['status']} action={data['decision']['action']} "
              f"urgency={data['decision']['urgency']}")
        for tc in tool_calls:
            if tc["status"] == "pending_approval":
                pending.append((conversation_id, tc["tool_call_id"]))

    # resolve every pending tool call -- reject the last one, approve the rest, so the
    # dashboard's tool_call_status_breakdown isn't a monoculture of "executed"
    for i, (conversation_id, tool_call_id) in enumerate(pending):
        verb = "reject" if i == len(pending) - 1 else "approve"
        reason = "amount looks wrong" if verb == "reject" else "verified, seeded demo data"
        url = f"{args.base_url}/conversations/{conversation_id}/tool-calls/{tool_call_id}/{verb}"
        _die_on_error(requests.post(url, json={"reason": reason}, headers=headers))
        print(f"   {verb}d {tool_call_id} ({conversation_id})")

    print(f"\nSeeded {len(tickets)} conversations, resolved {len(pending)} tool call(s). "
          f"Open {args.base_url}/dashboard to see them.")


if __name__ == "__main__":
    main()
