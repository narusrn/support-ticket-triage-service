"""Streamlit chat UI for the triage agent -- deployable to Streamlit Cloud. Calls
agents.runner.Runner directly, in-process (no running FastAPI server needed), the
same way scripts/react_playground.py talks to the graph directly instead of over
HTTP -- Streamlit Cloud only runs this one process, so there's nothing to call out
to even if we wanted an HTTP hop.

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

from agents.runner import Runner
from storage import db

st.set_page_config(page_title="Support Ticket Triage", page_icon="🎫")

URGENCY_COLORS = {"critical": "red", "high": "orange", "medium": "orange", "low": "green"}


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
if "messages" not in st.session_state:
    st.session_state.messages = []  # [{"role": "customer"|"agent", "content": str}]
if "pending_tool" not in st.session_state:
    st.session_state.pending_tool = None

st.title("🎫 Support Ticket Triage")
st.caption("LangGraph agent — classifies, searches a knowledge base (agentic RAG), "
           "and decides the next action. Side-effecting actions need your approval below.")

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
        st.session_state.messages = []
        st.session_state.pending_tool = None
        st.rerun()

for m in st.session_state.messages:
    with st.chat_message("user" if m["role"] == "customer" else "assistant"):
        st.markdown(m["content"])


def render_decision(data: dict) -> None:
    d = data["decision"]
    urgency = d.get("urgency", "?")
    color = URGENCY_COLORS.get(urgency, "gray")
    st.markdown(f":{color}[**{urgency.upper()}**]  ·  action: **{d.get('action')}**  "
                f"·  confidence {d.get('confidence', 0):.2f}")

    reply_text = data.get("reply") or d.get("draft_reply")
    if reply_text:
        st.markdown(reply_text)
        st.session_state.messages.append({"role": "agent", "content": reply_text})
    elif d.get("rationale"):
        # route_to_specialist / escalate_to_human usually has no customer-facing reply --
        # show the rationale instead of leaving the chat turn silent.
        note = f"_{d['rationale']}_"
        st.markdown(note)
        st.session_state.messages.append({"role": "agent", "content": note})

    for tc in d.get("tool_calls", []):
        if tc["status"] == "pending_approval":
            st.session_state.pending_tool = tc


if st.session_state.pending_tool:
    tc = st.session_state.pending_tool
    st.warning(f"Approval needed: **{tc['tool_name']}**({tc['args']})")
    reason = st.text_input("Reason (optional)", key="approval_reason")
    col1, col2 = st.columns(2)
    if col1.button("✅ Approve", use_container_width=True):
        data = run_async(runner.approve(st.session_state.conversation_id, tc["tool_call_id"], reason or None))
        st.session_state.pending_tool = None
        with st.chat_message("assistant"):
            render_decision(data)
        st.rerun()
    if col2.button("❌ Reject", use_container_width=True):
        data = run_async(runner.reject(st.session_state.conversation_id, tc["tool_call_id"], reason or None))
        st.session_state.pending_tool = None
        with st.chat_message("assistant"):
            render_decision(data)
        st.rerun()
else:
    prompt = st.chat_input("Describe the issue, or continue the conversation...")
    if prompt:
        st.session_state.messages.append({"role": "customer", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.spinner("Thinking..."):
            if st.session_state.conversation_id is None:
                conversation_id = f"c_{uuid.uuid4().hex[:8]}"
                metadata = {"plan": plan, "tenure_months": int(tenure_months), "prior_tickets": int(prior_tickets)}
                data = run_async(runner.ingest_ticket(
                    conversation_id, [{"role": "customer", "content": prompt}], metadata))
                st.session_state.conversation_id = conversation_id
            else:
                data = run_async(runner.send_message(st.session_state.conversation_id, prompt))
        with st.chat_message("assistant"):
            render_decision(data)
