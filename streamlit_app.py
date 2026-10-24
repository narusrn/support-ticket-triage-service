"""Streamlit chat UI for the triage agent -- deployable to Streamlit Cloud. Calls
agents.runner.Runner directly, in-process (no running FastAPI server needed), the
same way scripts/react_playground.py talks to the graph directly instead of over
HTTP -- Streamlit Cloud only runs this one process, so there's nothing to call out
to even if we wanted an HTTP hop.

Shows the internal handoff trail (Supervisor -> Triage Classifier -> KB Retriever ->
Action Decider), not just the final reply -- the same "Speaker: sentence" lines
GET /conversations/{id} returns, color-coded by speaker, same distinction
agents.runner._is_narration() already draws between internal narration and a real
customer-facing reply.

Local run:
  streamlit run streamlit_app.py

Streamlit Cloud: point the app at this file. Set OPENAI_API_KEY under the app's
Settings -> Secrets (TOML: OPENAI_API_KEY = "sk-...") -- there's no .env file on
Cloud, so config.py's own .env loading never fires there; this file copies
st.secrets into os.environ before importing anything from the project, since
config.Settings() reads the environment at import time.
"""
import asyncio
import os
import sys
import uuid
from pathlib import Path

import streamlit as st

try:
    for _key in ("OPENAI_API_KEY",):
        if _key in st.secrets:
            os.environ[_key] = st.secrets[_key]
except Exception:
    pass  # no secrets.toml locally -- config.py's own .env loading covers that case

sys.path.insert(0, str(Path(__file__).resolve().parent))

from langgraph.checkpoint.memory import InMemorySaver

from agents.runner import Runner, _is_narration
from storage import db

st.set_page_config(page_title="Support Ticket Triage", page_icon="🎫")

URGENCY_COLORS = {"critical": "red", "high": "orange", "medium": "orange", "low": "green"}
SPEAKER_COLORS = {
    "Supervisor": "blue",
    "Triage Classifier": "orange",
    "KB Retriever": "violet",
    "Action Decider": "green",
    "System": "gray",
}


@st.cache_resource
def get_runner() -> Runner:
    # InMemorySaver, not AsyncSqliteSaver: simplest checkpointer that survives across
    # Streamlit reruns within one running process (cached via st.cache_resource) without
    # an async context-manager lifecycle to manage across reruns -- a real server restart
    # loses in-flight graph state either way on Streamlit Cloud's ephemeral filesystem, so
    # there's no durability to buy here. The ticket/run/tool_call history itself still
    # persists to a real SQLite file for the session's lifetime (conversations, decisions,
    # usage -- everything GET /conversations/{id} would show).
    db_path = os.getenv("STREAMLIT_DB_PATH", "./data/streamlit_app.db")
    db.init_db(db_path)
    conn = db.get_connection(db_path)
    return Runner(conn, InMemorySaver())


def run_async(coro):
    return asyncio.run(coro)


runner = get_runner()

if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = None

st.title("🎫 Support Ticket Triage")
st.caption("LangGraph agent — classifies, searches a knowledge base (agentic RAG), "
           "and decides the next action. Internal handoffs are shown as they happen; "
           "side-effecting actions need your approval below.")

with st.sidebar:
    st.subheader("New ticket")
    st.caption("Used for the first message of a new conversation only.")
    plan = st.selectbox("Plan", ["free", "pro", "enterprise"])
    tenure_months = st.number_input("Tenure (months)", min_value=0, value=0)
    prior_tickets = st.number_input("Prior tickets", min_value=0, value=0)
    if st.session_state.conversation_id:
        st.caption(f"conversation_id: `{st.session_state.conversation_id}`")
    if st.button("Start new conversation"):
        st.session_state.conversation_id = None
        st.rerun()


def render_message(m: dict) -> None:
    role = m["role"]
    if role in ("customer", "operator"):
        with st.chat_message("user"):
            if role == "operator":
                st.caption("you, as operator")
            st.markdown(m["content"])
    elif _is_narration(m):
        # An internal handoff line ("Supervisor: ...", "KB Retriever: ...") -- the
        # supervisor/specialists talking to each other, not a customer-facing reply.
        # Kept out of a chat bubble and color-coded by speaker so the handoff trail
        # reads as a trace, not as more things the agent "said" to the customer.
        speaker, _, text = m["content"].partition(": ")
        color = SPEAKER_COLORS.get(speaker, "gray")
        st.markdown(f":{color}[**{speaker}**] {text}")
    else:
        # A real reply: a direct_reply to a non-ticket message, an auto_respond
        # draft_reply, or an advisory follow-up answer -- none of these carry a
        # "Speaker: " prefix, which is exactly how _is_narration() tells them apart.
        with st.chat_message("assistant"):
            st.markdown(m["content"])


def get_latest_decision(conversation_id: str) -> dict:
    decisions = db.get_decisions(runner.conn, conversation_id)
    return decisions[-1] if decisions else {}


def get_pending_tool_call(conversation_id: str) -> dict | None:
    for tc in db.get_tool_calls(runner.conn, conversation_id):
        if tc["status"] == "pending_approval":
            return tc
    return None


conversation_id = st.session_state.conversation_id

if conversation_id:
    for m in db.get_messages(runner.conn, conversation_id):
        render_message(m)

    d = get_latest_decision(conversation_id)
    if d.get("urgency"):
        color = URGENCY_COLORS.get(d["urgency"], "gray")
        st.caption(f":{color}[{d['urgency'].upper()}]  ·  action: **{d.get('action')}**  "
                   f"·  confidence {d.get('confidence', 0):.2f}")
    if d.get("draft_reply"):
        # auto_respond's draft_reply is never written into the messages table (only
        # an advisory send_message() reply is) -- it lives solely on the decision
        # record, so it has to be rendered separately from the history loop above.
        with st.chat_message("assistant"):
            st.markdown(d["draft_reply"])
    elif d.get("rationale") and d.get("action") in ("route_to_specialist", "escalate_to_human"):
        # Neither of these actions has a customer-facing reply -- show why, so the
        # turn doesn't end in silence after a page of internal handoff lines.
        with st.chat_message("assistant"):
            st.markdown(f"_{d['rationale']}_")

pending_tool = get_pending_tool_call(conversation_id) if conversation_id else None

if pending_tool:
    st.warning(f"Approval needed: **{pending_tool['tool_name']}**({pending_tool['args']})")
    reason = st.text_input("Reason (optional)", key="approval_reason")
    col1, col2 = st.columns(2)
    if col1.button("✅ Approve", use_container_width=True):
        with st.spinner("Executing..."):
            run_async(runner.approve(conversation_id, pending_tool["tool_call_id"], reason or None))
        st.rerun()
    if col2.button("❌ Reject", use_container_width=True):
        with st.spinner("Rejecting..."):
            run_async(runner.reject(conversation_id, pending_tool["tool_call_id"], reason or None))
        st.rerun()
else:
    prompt = st.chat_input("Describe the issue, or continue the conversation...")
    if prompt:
        with st.spinner("Thinking..."):
            if conversation_id is None:
                conversation_id = f"c_{uuid.uuid4().hex[:8]}"
                metadata = {"plan": plan, "tenure_months": int(tenure_months), "prior_tickets": int(prior_tickets)}
                run_async(runner.ingest_ticket(
                    conversation_id, [{"role": "customer", "content": prompt}], metadata))
                st.session_state.conversation_id = conversation_id
            else:
                run_async(runner.send_message(conversation_id, prompt))
        st.rerun()
