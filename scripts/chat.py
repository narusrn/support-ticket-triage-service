"""Terminal client for the triage API -- a thin REPL that turns operator input
into the exact same HTTP calls an operator would otherwise type as curl. Adds no
logic the API doesn't already have; api/routes/ remains the single source of
truth for behavior.

Usage:
  python scripts/chat.py --ticket eval/labeled_tickets.json --ticket-id ticket_1_billing
  python scripts/chat.py --ticket my_ticket.json          # {"messages": [...], "customer_metadata": {...}}
  python scripts/chat.py --paste                          # paste a raw ticket thread interactively
  python scripts/chat.py --conversation-id c_abc12345      # resume an existing conversation

In the REPL:
  plain text              -> POST /conversations/{id}/messages
  /approve <tc_id> [why]  -> POST /conversations/{id}/tool-calls/{tc_id}/approve
  /reject <tc_id> [why]   -> POST /conversations/{id}/tool-calls/{tc_id}/reject
  /history                -> GET /conversations/{id}
  /quit                   -> exit
"""
import argparse
import json
import sys

import requests


def parse_command(line: str) -> tuple[str, dict]:
    """Pure parsing, kept separate from any HTTP call so it's testable without
    a running server or network mocking."""
    line = line.strip()
    if line in ("/quit", "/exit"):
        return "quit", {}
    if line == "/history":
        return "history", {}
    if line.startswith("/approve ") or line.startswith("/reject "):
        verb, rest = line[1:].split(" ", 1)
        parts = rest.strip().split(" ", 1)
        tool_call_id = parts[0]
        reason = parts[1].strip() if len(parts) > 1 and parts[1].strip() else None
        return verb, {"tool_call_id": tool_call_id, "reason": reason}
    return "message", {"text": line}


def read_pasted_thread(line_source) -> str:
    """Joins lines from line_source (any iterable of str -- an input() loop when run
    interactively, or a plain list in tests) until a line that's just '.', the classic
    one-dot-ends-the-message convention (same idea as SMTP DATA). Chosen over relying
    on Ctrl+D/Ctrl+Z EOF handling, which behaves differently across terminals."""
    lines = []
    for line in line_source:
        if line == ".":
            break
        lines.append(line)
    return "\n".join(lines)


def _input_lines():
    while True:
        try:
            yield input()
        except EOFError:
            return


def _ingest_pasted_thread(base_url: str, headers: dict) -> str:
    print("Paste the ticket thread below -- whatever was copied, any format. "
          "Type a single '.' on its own line when done:")
    content = read_pasted_thread(_input_lines())
    if not content.strip():
        sys.exit("(empty paste, aborting)")

    plan = ""
    while plan not in ("free", "pro", "enterprise"):
        plan = input("customer plan [free/pro/enterprise]: ").strip().lower()
    tenure_raw = input("tenure_months [0]: ").strip()
    prior_raw = input("prior_tickets [0]: ").strip()

    body = {
        "messages": [{"role": "customer", "content": content}],
        "customer_metadata": {
            "plan": plan,
            "tenure_months": int(tenure_raw) if tenure_raw else 0,
            "prior_tickets": int(prior_raw) if prior_raw else 0,
        },
    }
    data = _die_on_error(requests.post(f"{base_url}/tickets", json=body, headers=headers))
    _print_decision(data)
    return data["conversation_id"]


def _load_ticket_body(path: str, ticket_id: str | None) -> dict:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        if ticket_id is None:
            ids = [t["ticket_id"] for t in data]
            raise SystemExit(f"{path} contains multiple tickets -- pass --ticket-id one of: {ids}")
        matches = [t for t in data if t["ticket_id"] == ticket_id]
        if not matches:
            raise SystemExit(f"no ticket_id={ticket_id!r} in {path}")
        ticket = matches[0]
        return {"messages": ticket["messages"], "customer_metadata": ticket["customer_metadata"]}
    return data  # already {"messages": [...], "customer_metadata": {...}}


def _die_on_error(resp: requests.Response) -> dict:
    if resp.status_code >= 400:
        print(f"error {resp.status_code}: {resp.text}", file=sys.stderr)
        sys.exit(1)
    return resp.json()


def _print_decision(data: dict) -> None:
    d = data["decision"]
    print(f"-- status={data['status']} urgency={d['urgency']} action={d['action']} confidence={d['confidence']}")
    if d.get("draft_reply"):
        print(f"   draft_reply: {d['draft_reply']}")
    for tc in d.get("tool_calls", []):
        print(f"   tool_call {tc['tool_call_id']}: {tc['tool_name']} args={tc['args']} [{tc['status']}]")
    if data.get("reply"):
        print(f"agent: {data['reply']}")


def _print_history(base_url: str, headers: dict, conversation_id: str) -> None:
    data = _die_on_error(requests.get(f"{base_url}/conversations/{conversation_id}", headers=headers))
    print(f"-- conversation {conversation_id} ({data['status']}) --")
    for m in data["messages"]:
        print(f"[{m['role']}] {m['content']}")
    print(f"usage: {data['usage']}")


def main() -> None:
    # Windows consoles default to a legacy codepage that mangles non-ASCII (em dashes, Thai
    # text like the labeled Thai outage ticket) even though the API returns valid UTF-8.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except AttributeError:
            pass

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/api/v1")
    parser.add_argument("--api-key", default=None, help="X-API-Key header; omit if the server has none configured")
    parser.add_argument("--ticket", help="Path to a ticket JSON file (or a labeled_tickets.json-style list)")
    parser.add_argument("--ticket-id", help="Which entry to use, when --ticket points at a list of tickets")
    parser.add_argument("--paste", action="store_true", help="Paste a raw ticket thread interactively instead of loading a JSON file")
    parser.add_argument("--conversation-id", help="Resume an existing conversation instead of ingesting a new ticket")
    args = parser.parse_args()

    headers = {"X-API-Key": args.api_key} if args.api_key else {}

    if args.conversation_id:
        conversation_id = args.conversation_id
        _print_history(args.base_url, headers, conversation_id)
    elif args.paste:
        conversation_id = _ingest_pasted_thread(args.base_url, headers)
    elif args.ticket:
        body = _load_ticket_body(args.ticket, args.ticket_id)
        data = _die_on_error(requests.post(f"{args.base_url}/tickets", json=body, headers=headers))
        conversation_id = data["conversation_id"]
        _print_decision(data)
    else:
        parser.error("need --ticket <file.json>, --paste, or --conversation-id <id>")
        return

    print(f"\n(conversation_id={conversation_id} -- type a message, /approve <tc_id> [reason], "
          f"/reject <tc_id> [reason], /history, or /quit)")
    while True:
        try:
            line = input("> ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line.strip():
            continue

        verb, payload = parse_command(line)
        if verb == "quit":
            break
        elif verb == "history":
            _print_history(args.base_url, headers, conversation_id)
        elif verb in ("approve", "reject"):
            url = f"{args.base_url}/conversations/{conversation_id}/tool-calls/{payload['tool_call_id']}/{verb}"
            body = {"reason": payload["reason"]} if payload["reason"] else {}
            data = _die_on_error(requests.post(url, json=body, headers=headers))
            _print_decision(data)
        else:  # message
            url = f"{args.base_url}/conversations/{conversation_id}/messages"
            data = _die_on_error(requests.post(url, json={"message": payload["text"]}, headers=headers))
            _print_decision(data)


if __name__ == "__main__":
    main()
