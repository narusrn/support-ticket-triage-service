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

Lines from a turn that just finished reveal one at a time with a short pause (see
REVEAL_DELAY_SECONDS) instead of all appearing at once -- not real streaming. Runner
calls graph.ainvoke() (the whole production/API path), which only returns once the
full run is done; there is no node-by-node event to stream here the way
scripts/react_playground.py's graph.stream() gets one directly from the graph.
Re-rendering history on a later rerun (e.g. after approve/reject) always draws
instantly -- only a turn's own brand-new lines get the pause.

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
import time
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

REVEAL_DELAY_SECONDS = 0.45
URGENCY_COLORS = {"critical": "red", "high": "orange", "medium": "orange", "low": "green"}

# Real golden-set tickets (eval/golden_tickets.xlsx), not invented text -- picked to
# each demo a different path: direct_reply, auto_respond+RAG, escalate+approval gate,
# Thai language detection. Multi-message threads are newline-joined into one block,
# same convention scripts/chat.py --paste and the raw text/plain ingest endpoint use,
# since a new conversation's first message here is a single chat_input submission.
EXAMPLE_TICKETS = [
    {
        "label": "👋 Just a greeting",
        "text": "Hi, who are you? What can you help with?",
    },
    {
        "label": "📄 FAQ-answerable (data export)",
        "text": "Hi, how do I export my data as a CSV? I need it for an audit next week, no rush.",
    },
    {
        "label": "💸 Billing dispute (needs approval)",
        "text": "\n".join([
            "My payment failed when I tried to upgrade to Pro. Can you check what's wrong?",
            "I tried again with a different card. Now I see TWO pending charges but my account still shows Free plan??",
            "Okay this is getting ridiculous. Just checked my bank app — I have THREE charges of $29.99 now. None refunded. And I STILL don't have Pro access.",
            "HELLO?? Is anyone there??? I need this fixed NOW. I have a presentation in 2 hours and need the Pro export features. If these charges aren't reversed by end of day I'm disputing all of them with my bank.",
        ]),
    },
    {
        "label": "🔥 Critical outage (Thai)",
        "text": "\n".join([
            "ระบบเข้าไม่ได้ครับ ขึ้น error 500",
            "ลองหลายเครื่องแล้ว ทั้ง Chrome, Safari, Firefox ผลเหมือนกันหมด เพื่อนร่วมงานก็เข้าไม่ได้เหมือนกัน",
            "ตอนนี้ลูกค้าโวยเข้ามาเยอะมาก เรามี demo กับลูกค้ารายใหญ่บ่ายนี้ ถ้าระบบไม่กลับมา deal นี้อาจจะหลุด",
            "เช็ค status.company.com แล้ว บอกว่า all systems operational แต่เราใช้งานไม่ได้จริงๆ region Asia มีปัญหาหรือเปล่า?",
        ]),
    },
]
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
if "seen_count" not in st.session_state:
    st.session_state.seen_count = 0  # how many messages this conversation has already shown
if "revealing" not in st.session_state:
    st.session_state.revealing = False  # True for exactly the rerun right after an action
if "pending_example" not in st.session_state:
    st.session_state.pending_example = None  # set by an example button, consumed like a chat_input submission

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
        st.session_state.seen_count = 0
        st.session_state.revealing = False
        st.session_state.pending_example = None
        st.rerun()


def render_message(m: dict, delay: bool = False) -> None:
    if delay:
        time.sleep(REVEAL_DELAY_SECONDS)
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
    all_messages = db.get_messages(runner.conn, conversation_id)
    seen = st.session_state.seen_count
    revealing = st.session_state.revealing

    for m in all_messages[:seen]:
        render_message(m)  # already shown on an earlier rerun -- draw instantly
    for m in all_messages[seen:]:
        render_message(m, delay=revealing)  # this turn's new lines -- reveal one at a time
    st.session_state.seen_count = len(all_messages)

    d = get_latest_decision(conversation_id)
    # A greeting/non-ticket message gets a synthetic auto_respond decision purely so
    # every response has the same API shape (see agents/orchestrator.py's supervisor)
    # -- its reply is already one of all_messages (unprefixed, not narration) and its
    # urgency/action/confidence aren't a real triage outcome, so neither belongs here.
    is_direct_reply = d.get("rationale") == "direct conversational reply -- not a ticket requiring triage"

    if d.get("urgency") and not is_direct_reply:
        color = URGENCY_COLORS.get(d["urgency"], "gray")
        st.caption(f":{color}[{d['urgency'].upper()}]  ·  action: **{d.get('action')}**  "
                   f"·  confidence {d.get('confidence', 0):.2f}")

    already_shown = {m["content"] for m in all_messages}
    if d.get("draft_reply") and d["draft_reply"] not in already_shown:
        # auto_respond's draft_reply is never written into the messages table (only
        # an advisory send_message() reply, or a direct_reply's own answer, is) -- it
        # lives solely on the decision record, so it has to be rendered separately
        # from the history loop above. The already_shown check is what keeps a
        # direct_reply from appearing twice -- its answer IS in all_messages, so this
        # skips it instead of rendering the same text as a second bubble.
        if revealing:
            time.sleep(REVEAL_DELAY_SECONDS)
        with st.chat_message("assistant"):
            st.markdown(d["draft_reply"])
    elif d.get("rationale") and d.get("action") in ("route_to_specialist", "escalate_to_human"):
        # Neither of these actions has a customer-facing reply -- show why, so the
        # turn doesn't end in silence after a page of internal handoff lines.
        if revealing:
            time.sleep(REVEAL_DELAY_SECONDS)
        with st.chat_message("assistant"):
            st.markdown(f"_{d['rationale']}_")

    st.session_state.revealing = False  # consumed -- the next rerun just redraws history instantly

pending_tool = get_pending_tool_call(conversation_id) if conversation_id else None

if pending_tool:
    st.warning(f"Approval needed: **{pending_tool['tool_name']}**({pending_tool['args']})")
    reason = st.text_input("Reason (optional)", key="approval_reason")
    col1, col2 = st.columns(2)
    if col1.button("✅ Approve", use_container_width=True):
        with st.spinner("Executing..."):
            run_async(runner.approve(conversation_id, pending_tool["tool_call_id"], reason or None))
        st.session_state.revealing = True
        st.rerun()
    if col2.button("❌ Reject", use_container_width=True):
        with st.spinner("Rejecting..."):
            run_async(runner.reject(conversation_id, pending_tool["tool_call_id"], reason or None))
        st.session_state.revealing = True
        st.rerun()
else:
    if conversation_id is None:
        # Examples only make sense for starting a new conversation -- st.chat_input
        # can't be pre-filled programmatically (no `value=` the way st.text_input
        # has), so a click sends the example immediately via pending_example instead
        # of just dropping the text into the box for the user to review first.
        st.caption("Try an example:")
        cols = st.columns(len(EXAMPLE_TICKETS))
        for col, example in zip(cols, EXAMPLE_TICKETS):
            if col.button(example["label"], use_container_width=True):
                st.session_state.pending_example = example["text"]
                st.rerun()

    prompt = st.chat_input("Describe the issue, or continue the conversation...")
    if not prompt and st.session_state.pending_example:
        prompt = st.session_state.pending_example
        st.session_state.pending_example = None
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
        st.session_state.revealing = True
        st.rerun()
