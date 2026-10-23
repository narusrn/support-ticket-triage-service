"""Interactive dev console for the triage graph (agents/orchestrator.py) -- styled
with `rich` like scripts/chatbot.py, but talks to the graph directly in-process (no
running server, no HTTP) and streams each node's turn live as it happens, so you can
watch Supervisor/Triage Classifier/KB Retriever/Action Decider/System talk to each
other in real time.

Usage:
  python scripts/react_playground.py --ticket tickets.json --ticket-id ticket_1_billing  # a JSON list of tickets
  python scripts/react_playground.py --paste
"""
import argparse
import sys
import uuid
from pathlib import Path

# running this file directly puts scripts/ on sys.path, not the project root --
# same fix as eval/run_eval.py and scripts/chatbot.py use for the same reason.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, Prompt

from agents.llm import get_supervisor_prompt
from agents.orchestrator import build_graph
from scripts.chat import _load_ticket_body
from storage import db

SPEAKER_STYLES = {
    "Supervisor": "bold cyan",
    "Triage Classifier": "bold yellow",
    "KB Retriever": "bold magenta",
    "Action Decider": "bold green",
    "System": "dim",
}


def _print_agent_message(console: Console, content: str) -> None:
    speaker, _, text = content.partition(": ")
    if not text:  # no "Speaker: " prefix -- e.g. a direct_reply answer to the customer
        speaker, text = "Agent", content
    style = SPEAKER_STYLES.get(speaker, "white")
    console.print(f"  [{style}]{speaker}:[/{style}] {text}")


def _stream_until_pause(console: Console, graph, payload, config: dict, seen: int = 0) -> tuple[dict, int]:
    """Streams the graph turn by turn, printing every NEW agent message as soon as a
    node produces it, instead of waiting for the whole run to finish like .invoke()
    would -- this is the "watch them talk live" part. Returns (last_state, new_seen) --
    `seen` must carry over across calls (ingest, then each resume after an approval),
    since stream_mode="values" re-yields the FULL messages list from the start of the
    thread every time, not just what's new since the last call."""
    last_state: dict = {}
    for state in graph.stream(payload, config, stream_mode="values"):
        last_state = state
        for m in state.get("messages", [])[seen:]:
            if m["role"] == "agent":
                _print_agent_message(console, m["content"])
        seen = len(state.get("messages", []))
    return last_state, seen


def _reset_fields(messages: list[dict]) -> dict:
    """The payload every turn is built from -- fresh urgency/extracted/kb_results/decision/
    pending_tool/step every time, with only `messages` carrying real history forward.
    Needed so a SECOND ticket pasted mid-session gets re-classified from scratch instead
    of inheriting the first ticket's already-final decision (see supervisor()'s
    decision-is-not-None fast path -- without this reset it would just short-circuit to
    END immediately, as if the new message didn't matter)."""
    return {"messages": messages, "urgency": None, "extracted": None, "kb_results": None,
            "decision": None, "pending_tool": None, "tool_calls": [], "step": 0, "trace": []}


def _run_and_resolve(console: Console, graph, config: dict, payload, seen: int) -> tuple[dict, int]:
    """Runs payload through the graph, then keeps resolving approval interrupts (asking
    the operator each time) until the graph reaches a real final state. Shared by the
    first ingest and every later turn, so both get the exact same approval handling."""
    state, seen = _stream_until_pause(console, graph, payload, config, seen)
    while "__interrupt__" in state:
        pending = state["pending_tool"]
        console.print(Panel(f"{pending['tool_name']}({pending['args']})",
                             title="Awaiting human approval", border_style="yellow"))
        approve = Confirm.ask("Approve this action?", default=True)
        reason = Prompt.ask("reason (optional)", default="") or None
        state, seen = _stream_until_pause(
            console, graph,
            Command(resume={"decision": "approve" if approve else "reject", "reason": reason}),
            config, seen,
        )
    return state, seen


def _ingest(console: Console, graph, conversation_id: str, ticket_body: dict) -> tuple[dict, int]:
    metadata = ticket_body["customer_metadata"]
    if metadata:
        metadata_line = ", ".join(f"{k}={v}" for k, v in metadata.items())
    else:
        # --paste doesn't ask for plan/tenure/prior_tickets anymore -- triage_classifier.md
        # already says to weigh "plan tier, seat count, tenure, prior tickets" when deciding
        # urgency, so just tell it to pull those signals straight out of the thread text
        # itself when present ("I'm on the Enterprise plan...") instead of leaving it unweighed.
        metadata_line = ("not provided -- infer plan/tenure/prior-ticket signals from the "
                          "thread text itself if the customer mentions them, otherwise treat as unknown")
    messages = [
        {"role": "system", "content": get_supervisor_prompt()},
        {"role": "system", "content": f"Customer metadata: {metadata_line}"},
    ] + [{"role": m["role"], "content": m["content"]} for m in ticket_body["messages"]]
    initial_state = {"conversation_id": conversation_id, **_reset_fields(messages)}

    console.print(Panel("\n".join(m["content"] for m in ticket_body["messages"]),
                         title="customer", border_style="cyan"))
    config = {"configurable": {"thread_id": conversation_id}}
    return _run_and_resolve(console, graph, config, initial_state, 0)


def _print_final_decision(console: Console, state: dict) -> None:
    d = state.get("decision") or {}
    # A greeting/non-ticket message is represented as a trivial auto_respond decision
    # (see the supervisor's direct_reply branch) purely so every response has the same
    # shape for the API contract -- but the reply was already shown as a chat line above
    # (_print_agent_message), so repeating it in a "Final decision" box right under it
    # is just noise here. Only show the panel when it's an actual triage outcome.
    if d.get("rationale") == "direct conversational reply -- not a ticket requiring triage":
        return
    console.print(Panel(
        f"action: [bold]{d.get('action')}[/]   urgency: {d.get('urgency')}   confidence: {d.get('confidence')}\n"
        f"rationale: {d.get('rationale')}" + (f"\n\ndraft_reply: {d['draft_reply']}" if d.get("draft_reply") else ""),
        title="Final decision", border_style="green",
    ))


def _keep_going(console: Console, graph, conversation_id: str, state: dict, seen: int) -> None:
    """Found live: an earlier version answered every follow-up through a separate
    lightweight "advisory" LLM call that never touched the graph at all -- so pasting a
    SECOND real ticket mid-session just got chatted about instead of actually being
    triaged (no Supervisor/Triage Classifier/etc. lines at all). Every turn here re-enters
    the real graph instead: supervisor() decides fresh each time whether this new message
    is a real ticket (full pipeline, same as the first message) or just conversation
    (direct_reply) -- there's no separate code path needed for "just chatting," because
    that's already exactly what step 1 of _REACT_SUPERVISOR_RULES is for."""
    config = {"configurable": {"thread_id": conversation_id}}
    console.print("[dim](keep typing -- a real ticket gets the full pipeline again, "
                  "anything else just gets answered. blank line or /quit to stop)[/]")
    while True:
        try:
            line = Prompt.ask("[bold cyan]you[/]")
        except (EOFError, KeyboardInterrupt):
            console.print()
            return
        if not line.strip() or line.strip() in ("/quit", "/exit"):
            return
        payload = {"conversation_id": conversation_id,
                   **_reset_fields(state["messages"] + [{"role": "customer", "content": line}])}
        state, seen = _run_and_resolve(console, graph, config, payload, seen)
        _print_final_decision(console, state)


def main() -> None:
    for stream in (sys.stdin, sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except AttributeError:
            pass

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ticket", help="Path to a ticket JSON file (or a JSON list of tickets)")
    parser.add_argument("--ticket-id", help="Which entry to use, when --ticket points at a list of tickets")
    parser.add_argument("--paste", action="store_true", help="Start by typing a message instead of loading a ticket file")
    parser.add_argument("--db", default="data/react_playground.db", help="Scratch SQLite file for tool-call bookkeeping")
    args = parser.parse_args()

    console = Console()

    if args.paste:
        # Found live: the old "type lines, '.' ends it" ritual (borrowed from scripts/chat.py,
        # where it matters for pasting a real multi-message email thread) was confusing here
        # for the common case -- typing one message and hitting Enter just looked stuck,
        # waiting for a "." the user didn't know to type. This is a chat playground, so the
        # first message is treated exactly like every later one: one line, sent immediately.
        console.print(Panel("Type a message to start -- it becomes the first customer message.",
                             title="New conversation", border_style="cyan"))
        content = Prompt.ask("[bold cyan]you[/]")
        if not content.strip():
            console.print("[red](empty message, aborting)[/]")
            sys.exit(1)
        ticket_body = {"messages": [{"role": "customer", "content": content}], "customer_metadata": {}}
    elif args.ticket:
        ticket_body = _load_ticket_body(args.ticket, args.ticket_id)
    else:
        parser.error("need --ticket <file.json> or --paste")
        return

    db.init_db(args.db)
    conn = db.get_connection(args.db)
    conversation_id = f"c_play_{uuid.uuid4().hex[:8]}"
    db.insert_conversation(conn, conversation_id, "completed", ticket_body["customer_metadata"])

    graph = build_graph(conn, InMemorySaver())  # real classify_fn/decide_fn/supervisor_llm -- real OpenAI calls
    state, seen = _ingest(console, graph, conversation_id, ticket_body)
    _print_final_decision(console, state)
    _keep_going(console, graph, conversation_id, state, seen)


if __name__ == "__main__":
    main()
