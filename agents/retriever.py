"""Agentic RAG retriever: embeddings + a vector store + a tool-calling retrieve
-> grade-relevance -> rewrite-query-and-retry loop, replacing the earlier TF-IDF
search (agents/kb.py, now retired) with real vector search -- on request, to
demonstrate the pattern (the spec itself never required a vector DB: "a handful
of mock FAQ docs is plenty").

Modeled on the user's own reference implementation (narusrn/BBLAgent's
agents/retriever.py), adapted in three ways:

1. Sync, not async. The reference's graph is async because its whole
   orchestrator is async-first. agents/orchestrator.py's nodes are all sync
   (call_triage_classifier, call_action_decider, etc.) even though the outer
   graph is invoked via ainvoke() elsewhere -- LangChain's embeddings/
   vectorstore/chat-model .invoke() calls work either way, so there's no real
   need to introduce an async node, and tests/test_orchestrator.py calls
   graph.invoke() synchronously throughout, which a lone async node would be
   an easy way to quietly break later.
2. Per-topic doc_id attribution kept. The reference chunks its whole
   knowledge_base.txt as one blob -- it never needs to cite which section an
   answer came from. This project's Decision.cited_kb_doc_ids is part of the
   public API contract (and the eval harness's groundedness judging), so
   knowledge_base.txt is parsed into one Document per "# doc_id" section
   FIRST, each tagged with that doc_id, THEN split with
   RecursiveCharacterTextSplitter per section -- real chunking, just scoped
   so a split chunk still carries its parent's doc_id.
3. RetrieverState.query/kb_results are named to MATCH agents.orchestrator.
   GraphState's own query/kb_results keys on purpose: build_retriever_graph()'s
   compiled output is embedded directly as a node in the parent graph (see
   agents/orchestrator.py's "kb_retriever_search" node), the same way the
   reference embeds its own retriever_graph into its orchestrator -- and
   LangGraph only auto-transfers state into/out of an embedded subgraph for
   key names the parent and child schemas share. retriever_messages is the
   one key deliberately named to NOT match (agents.orchestrator.GraphState's
   own messages key) -- this loop's internal tool-calling scratchpad must
   never leak into the supervisor's conversation history.
"""
import re
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Literal, Optional, TypedDict

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

from agents.llm import call_with_retry, make_llm_client
from config import settings

KNOWLEDGE_BASE_PATH = Path(__file__).resolve().parent.parent / "data" / "knowledge_base.txt"

_SECTION_RE = re.compile(r"^#\s*(\S+)\s*$", re.MULTILINE)

_QUERY_SYSTEM_PROMPT = (
    "You help answer support questions by searching a knowledge base. Call "
    "retrieve_knowledge_base with a short, specific query capturing what the "
    "customer is actually asking -- not their whole message verbatim. Call it "
    "at most once per turn."
)

_GRADE_PROMPT = """\
You are grading whether a retrieved passage is relevant to a user question.

Retrieved passage:
{context}

User question: {question}

If the passage contains information that actually helps answer the question, grade it 'yes', else 'no'."""

_REWRITE_PROMPT = """\
This question didn't retrieve a relevant passage from the knowledge base:
-------
{question}
-------
Reason about the underlying intent and rewrite it as a clearer, more specific
question likely to retrieve a better match. Return only the rewritten question."""


def _parse_sections(text: str) -> dict[str, str]:
    """knowledge_base.txt is "# doc_id" headers followed by that topic's body,
    blank-line separated -- same structure the old data/faq/*.md files had,
    just all in one file. Keeping one Document per section (before any
    character-level splitting) is what lets a retrieved chunk still answer
    "which doc_id is this from"."""
    parts = _SECTION_RE.split(text)[1:]  # [doc_id, body, doc_id, body, ...]
    return {doc_id.strip(): body.strip() for doc_id, body in zip(parts[0::2], parts[1::2])}


def get_doc_text(doc_id: str, path: Path = KNOWLEDGE_BASE_PATH) -> Optional[str]:
    """For eval/run_eval.py's LLM-as-judge: the full text of one cited doc_id's
    section, so the judge can check a reply against what that section actually
    says. The old agents.kb.FAQ_DIR (one file per doc_id) no longer exists --
    knowledge_base.txt replaces it, so this is the equivalent lookup."""
    return _parse_sections(path.read_text(encoding="utf-8")).get(doc_id)


def _build_documents(path: Path = KNOWLEDGE_BASE_PATH) -> list[Document]:
    sections = _parse_sections(path.read_text(encoding="utf-8"))
    # chunk_size chosen the same way the reference's own comment reasons about
    # it: big enough that every CURRENT topic stays one whole chunk (the
    # longest here is well under 800 chars), small enough that a future
    # longer topic still gets safely sub-split instead of growing unbounded.
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    documents = []
    for doc_id, body in sections.items():
        for chunk in splitter.split_text(body):
            documents.append(Document(page_content=chunk, metadata={"doc_id": doc_id}))
    return documents


class RetrieverState(TypedDict):
    query: str  # shared name with GraphState.query -- auto-transferred in when embedded as a node
    retriever_messages: Annotated[list, add_messages]  # this loop's own tool-calling scratchpad --
                                                          # deliberately NOT agents.orchestrator.GraphState's
                                                          # messages key, so none of this leaks into the
                                                          # supervisor's conversation history.
    retries: int
    kb_results: list[dict]  # [{"doc_id": ..., "snippet": ...}, ...] -- shared name with
                              # GraphState.kb_results -- auto-transferred back out when embedded


class GradeDocuments(BaseModel):
    binary_score: Literal["yes", "no"] = Field(description="Are the retrieved docs relevant to the question?")


def build_retriever_graph(embeddings=None, llm=None, documents: Optional[list[Document]] = None):
    """Injectable the same way agents/orchestrator.py's build_graph() is --
    tests pass fake embeddings/llm/documents, nothing here needs real network
    access or an API key when they do."""
    embeddings = embeddings or OpenAIEmbeddings(model=settings.EMBEDDING_MODEL, api_key=settings.OPENAI_API_KEY)
    llm = llm or make_llm_client(settings.OPENAI_MODEL)
    documents = documents if documents is not None else _build_documents()

    vectorstore = InMemoryVectorStore.from_documents(documents=documents, embedding=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": settings.RETRIEVER_TOP_K})

    def _do_retrieve(query: str) -> list[Document]:
        return retriever.invoke(query)

    @tool
    def retrieve_knowledge_base(query: str) -> str:
        """Search the knowledge base for passages relevant to the query."""
        docs = _do_retrieve(query)
        return "\n\n".join(d.page_content for d in docs) or "No relevant information found in the knowledge base."

    llm_with_tool = llm.bind_tools([retrieve_knowledge_base])

    def generate_query_or_respond(state: RetrieverState) -> dict:
        history = state.get("retriever_messages") or []
        seed = []
        if not history:
            seed = [SystemMessage(content=_QUERY_SYSTEM_PROMPT), HumanMessage(content=state["query"])]
            history = seed
        response = call_with_retry(lambda: llm_with_tool.invoke(history))
        return {"retriever_messages": seed + [response]}

    def route_on_tool_calls(state: RetrieverState) -> str:
        last = state["retriever_messages"][-1]
        return "retrieve" if getattr(last, "tool_calls", None) else END

    def retrieve_node(state: RetrieverState) -> dict:
        last = state["retriever_messages"][-1]
        tool_messages = []
        docs: list[dict] = []
        for tc in last.tool_calls:
            query = tc["args"].get("query", state["query"])
            found = _do_retrieve(query)
            docs.extend({"doc_id": d.metadata.get("doc_id", "unknown"), "snippet": d.page_content} for d in found)
            content = "\n\n".join(d.page_content for d in found) or "No relevant information found in the knowledge base."
            tool_messages.append(ToolMessage(content=content, tool_call_id=tc["id"]))
        return {"retriever_messages": tool_messages, "kb_results": docs}

    def grade_documents(state: RetrieverState) -> str:
        docs = state.get("kb_results") or []
        if not docs:
            return "rewrite_question" if state.get("retries", 0) < settings.RETRIEVER_MAX_REWRITES else END

        context = "\n\n".join(d["snippet"] for d in docs)
        grader = llm.with_structured_output(GradeDocuments)
        result = call_with_retry(
            lambda: grader.invoke(_GRADE_PROMPT.format(question=state["query"], context=context)))

        if result.binary_score == "yes":
            return END
        if state.get("retries", 0) >= settings.RETRIEVER_MAX_REWRITES:
            return END
        return "rewrite_question"

    def rewrite_question(state: RetrieverState) -> dict:
        response = call_with_retry(lambda: llm.invoke(_REWRITE_PROMPT.format(question=state["query"])))
        return {
            "retriever_messages": [HumanMessage(content=response.content)],
            "retries": state.get("retries", 0) + 1,
        }

    graph = StateGraph(RetrieverState)
    graph.add_node("generate_query_or_respond", generate_query_or_respond)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("rewrite_question", rewrite_question)

    graph.set_entry_point("generate_query_or_respond")
    graph.add_conditional_edges("generate_query_or_respond", route_on_tool_calls,
                                 {"retrieve": "retrieve", END: END})
    graph.add_conditional_edges("retrieve", grade_documents, {END: END, "rewrite_question": "rewrite_question"})
    graph.add_edge("rewrite_question", "generate_query_or_respond")

    return graph.compile()


@lru_cache(maxsize=1)
def _default_graph():
    # Built once per process -- this is what actually embeds knowledge_base.txt,
    # so the (small, one-time) embedding cost only happens on the first real
    # kb_retriever call, not every single one.
    return build_retriever_graph()


def retrieve(query: str, graph=None) -> list[dict]:
    """Plain sync entry point -- [{"doc_id": ..., "snippet": ...}, ...], the same
    shape agents/orchestrator.py's kb_results already expects. Useful standalone
    (tests/test_retriever.py's own tests, a future CLI/script caller) even though
    agents/orchestrator.py itself now embeds the compiled graph directly as a node
    instead of calling this."""
    g = graph or _default_graph()
    result = g.invoke({"query": query, "retriever_messages": [], "retries": 0, "kb_results": []})
    return result.get("kb_results") or []


def build_stub_graph(fn):
    """Wraps a plain query -> list[dict] callable as a trivial one-node compiled
    graph with RetrieverState's own schema. For tests that need to inject fake KB
    results into agents.orchestrator.build_graph(retriever_graph=...) without a
    real embeddings/LLM pair -- agents/orchestrator.py embeds whatever compiled
    graph it's given directly as a node (for get_graph(xray=True) to expand), so
    a fake stand-in has to be a compiled graph too, not just a function."""
    def _stub(state: RetrieverState) -> dict:
        return {"kb_results": fn(state["query"])}

    graph = StateGraph(RetrieverState)
    graph.add_node("stub", _stub)
    graph.set_entry_point("stub")
    graph.add_edge("stub", END)
    return graph.compile()
