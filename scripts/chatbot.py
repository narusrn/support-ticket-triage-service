"""A nicer terminal client for the triage API, built on `rich` -- same 5 endpoints
and same command vocabulary as scripts/chat.py (reuses its parsing/loading helpers
directly, no duplicated logic), just styled like an actual chatbot: colored urgency/
status badges, panels for the agent's replies, tables for tool calls and history,
and a spinner while waiting on the API.

Usage:
  python scripts/chatbot.py --ticket eval/labeled_tickets.json --ticket-id ticket_1_billing
  python scripts/chatbot.py --paste
  python scripts/chatbot.py --conversation-id c_abc12345

Same REPL commands as scripts/chat.py: plain text, /approve <id> [reason],
/reject <id> [reason], /history, /quit.
"""
import argparse
import sys
from pathlib import Path

# running this file directly (`python scripts/chatbot.py`) puts scripts/ on sys.path,
# not the project root, so `from scripts.chat import ...` below would otherwise fail --
# same fix as eval/run_eval.py uses for the same reason.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import requests
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import IntPrompt, Prompt
from rich.table import Table
from rich.text import Text

from scripts.chat import _load_ticket_body, parse_command, read_pasted_thread

URGENCY_STYLES = {"critical": "bold red", "high": "bold dark_orange", "medium": "yellow", "low": "green"}
TOOL_STATUS_STYLES = {"pending_approval": "yellow", "executed": "green", "rejected": "red"}
ROLE_STYLES = {"customer": "cyan", "agent": "green", "operator": "magenta"}


def _call(console: Console, headers: dict, method: str, url: str, **kwargs) -> dict:
    with console.status("[bold green]Waiting for the agent...[/]", spinner="dots"):
        resp = requests.request(method, url, headers=headers, **kwargs)
    if resp.status_code >= 400:
        console.print(f"[bold red]error {resp.status_code}:[/] {resp.text}")
        sys.exit(1)
    return resp.json()


def render_decision(console: Console, data: dict) -> None:
    d = data["decision"]
    urgency_style = URGENCY_STYLES.get(d["urgency"], "white")
    status_style = "green" if data["status"] == "completed" else "yellow"

    header = Text()
    header.append("status ", style="dim")
    header.append(f"{data['status']}   ", style=status_style)
    header.append("urgency ", style="dim")
    header.append(f"{d['urgency'].upper()}   ", style=urgency_style)
    header.append("action ", style="dim")
    header.append(f"{d['action']}   ", style="bold")
    header.append("confidence ", style="dim")
    header.append(f"{d['confidence']:.2f}", style="cyan")
    console.print(Panel(header, title="Decision", border_style=urgency_style, expand=False))

    if d.get("draft_reply"):
        console.print(Panel(Markdown(d["draft_reply"]), title="Agent (draft reply)", border_style="blue"))

    tool_calls = d.get("tool_calls", [])
    if tool_calls:
        table = Table(title="Tool calls", border_style="dim")
        table.add_column("tool_call_id", style="dim")
        table.add_column("tool_name", overflow="fold")
        table.add_column("status")
        table.add_column("args", overflow="fold")
        for tc in tool_calls:
            style = TOOL_STATUS_STYLES.get(tc["status"], "white")
            table.add_row(tc["tool_call_id"], tc["tool_name"], f"[{style}]{tc['status']}[/{style}]", str(tc["args"]))
        console.print(table)

    if data.get("reply"):
        console.print(Panel(Markdown(data["reply"]), title="Agent", border_style="green"))


def render_history(console: Console, base_url: str, headers: dict, conversation_id: str) -> None:
    data = _call(console, headers, "GET", f"{base_url}/conversations/{conversation_id}")
    table = Table(title=f"conversation {conversation_id} ({data['status']})", border_style="dim")
    table.add_column("role")
    table.add_column("content", overflow="fold")
    for m in data["messages"]:
        style = ROLE_STYLES.get(m["role"], "white")
        table.add_row(f"[{style}]{m['role']}[/{style}]", m["content"])
    console.print(table)
    console.print(f"[dim]usage: {data['usage']}[/]")


def ingest_pasted(console: Console, base_url: str, headers: dict) -> str:
    console.print(Panel(
        "Type or paste whatever you want to ask or report below -- a real ticket thread, "
        "or just a question.\nA single '.' on its own line, then Enter, ends it.",
        title="New conversation", border_style="cyan",
    ))

    def lines():
        while True:
            try:
                yield console.input("[dim]paste>[/] ")
            except EOFError:
                return

    content = read_pasted_thread(lines())
    if not content.strip():
        console.print("[red](empty paste, aborting)[/]")
        sys.exit(1)

    plan = Prompt.ask("customer plan", choices=["free", "pro", "enterprise"], default="free")
    tenure_months = IntPrompt.ask("tenure_months", default=0)
    prior_tickets = IntPrompt.ask("prior_tickets", default=0)

    body = {
        "messages": [{"role": "customer", "content": content}],
        "customer_metadata": {"plan": plan, "tenure_months": tenure_months, "prior_tickets": prior_tickets},
    }
    data = _call(console, headers, "POST", f"{base_url}/tickets", json=body)
    render_decision(console, data)
    return data["conversation_id"]


def main() -> None:
    # same rationale as scripts/chat.py: Windows consoles default to a legacy codepage
    # that mangles non-ASCII input/output otherwise (seen live: a Thai reply rendered
    # as mojibake because only stdin, not stdout/stderr, was reconfigured here).
    for stream in (sys.stdin, sys.stdout, sys.stderr):
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

    console = Console()
    headers = {"X-API-Key": args.api_key} if args.api_key else {}

    if args.conversation_id:
        conversation_id = args.conversation_id
        render_history(console, args.base_url, headers, conversation_id)
    elif args.paste:
        conversation_id = ingest_pasted(console, args.base_url, headers)
    elif args.ticket:
        body = _load_ticket_body(args.ticket, args.ticket_id)
        data = _call(console, headers, "POST", f"{args.base_url}/tickets", json=body)
        conversation_id = data["conversation_id"]
        render_decision(console, data)
    else:
        parser.error("need --ticket <file.json>, --paste, or --conversation-id <id>")
        return

    console.print(Panel(
        f"conversation_id=[bold]{conversation_id}[/]\n"
        r"Type a message, [bold]/approve <tc_id> \[reason][/], [bold]/reject <tc_id> \[reason][/], "
        "[bold]/history[/], or [bold]/quit[/]",
        title="Ready", border_style="cyan",
    ))

    while True:
        try:
            line = Prompt.ask("[bold cyan]you[/]")
        except (EOFError, KeyboardInterrupt):
            console.print()
            break
        if not line.strip():
            continue

        verb, payload = parse_command(line)
        if verb == "quit":
            break
        elif verb == "history":
            render_history(console, args.base_url, headers, conversation_id)
        elif verb in ("approve", "reject"):
            url = f"{args.base_url}/conversations/{conversation_id}/tool-calls/{payload['tool_call_id']}/{verb}"
            body = {"reason": payload["reason"]} if payload["reason"] else {}
            data = _call(console, headers, "POST", url, json=body)
            render_decision(console, data)
        else:  # message
            url = f"{args.base_url}/conversations/{conversation_id}/messages"
            data = _call(console, headers, "POST", url, json={"message": payload["text"]})
            render_decision(console, data)


if __name__ == "__main__":
    main()
