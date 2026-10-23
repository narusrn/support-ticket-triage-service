"""Tests for agents/retriever.py (the agentic RAG retriever: embeddings + vector
search + grade/rewrite loop). No network calls, no API key needed -- embeddings
and every LLM call (query generation, grading, rewriting) are injected fakes,
same house rule as agents/orchestrator.py's own tests.
"""
import re

from langchain_core.documents import Document
from langchain_core.messages import AIMessage

from agents.retriever import GradeDocuments, build_retriever_graph, get_doc_text, retrieve

FAKE_DOCS = [
    Document(page_content="Dark mode lets you switch to a darker color theme from Settings.",
             metadata={"doc_id": "faq_dark_mode"}),
    Document(page_content="Refunds for duplicate charges are issued within 5 to 10 business days.",
             metadata={"doc_id": "faq_refund_policy"}),
]


class _FakeEmbeddings:
    """Deterministic, keyword-overlap-based "embedding" -- feature-hashed bag of
    words into a fixed-size vector. No real semantics, but similarity still
    tracks literal word overlap, which is all these tests need: whether the
    right doc_id comes back for a query that obviously shares its vocabulary."""
    DIM = 64

    def _vec(self, text: str) -> list[float]:
        vec = [0.0] * self.DIM
        for word in re.findall(r"[a-z0-9]+", text.lower()):
            vec[hash(word) % self.DIM] += 1.0
        return vec

    def embed_documents(self, texts):
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        return self._vec(text)


class _FakeMessage:
    def __init__(self, content=""):
        self.content = content


def _tool_call_message(query: str, call_id: str = "call_1") -> AIMessage:
    """A real AIMessage, not a bare fake object -- it has to survive LangGraph's
    add_messages reducer, which coerces every message through langchain_core's
    BaseMessage machinery."""
    return AIMessage(content="", tool_calls=[{"name": "retrieve_knowledge_base", "args": {"query": query}, "id": call_id}])


class _FakeLLM:
    """Scripts all three ways agents/retriever.py calls an LLM: bind_tools()
    (query generation), with_structured_output() (grading), and a plain
    .invoke() (query rewriting) -- each its own queue, popped in call order."""
    def __init__(self, tool_responses=None, grade_responses=None, rewrite_responses=None):
        self._tool_responses = list(tool_responses or [])
        self._grade_responses = list(grade_responses or [])
        self._rewrite_responses = list(rewrite_responses or [])

    def bind_tools(self, tools):
        queue = self._tool_responses

        class _Bound:
            def invoke(self, messages):
                return queue.pop(0)
        return _Bound()

    def with_structured_output(self, schema):
        queue = self._grade_responses

        class _Structured:
            def invoke(self, prompt):
                return queue.pop(0)
        return _Structured()

    def invoke(self, prompt):
        return self._rewrite_responses.pop(0)


def test_relevant_on_first_try_returns_the_right_doc_id():
    llm = _FakeLLM(
        tool_responses=[_tool_call_message("dark mode settings")],
        grade_responses=[GradeDocuments(binary_score="yes")],
    )
    graph = build_retriever_graph(embeddings=_FakeEmbeddings(), llm=llm, documents=FAKE_DOCS)

    result = retrieve("how do I turn on dark mode?", graph=graph)

    assert any(r["doc_id"] == "faq_dark_mode" for r in result)


def test_one_rewrite_recovers_from_a_bad_first_query():
    """Found live-adjacent lesson (same one orchestrator.py's own kb_retriever
    bug taught): a query with no lexical overlap with anything should be
    recoverable by rewriting, not just given up on immediately."""
    llm = _FakeLLM(
        tool_responses=[
            _tool_call_message("xyzzy unrelated gibberish", "call_1"),
            _tool_call_message("refund duplicate charge", "call_2"),
        ],
        grade_responses=[GradeDocuments(binary_score="no"), GradeDocuments(binary_score="yes")],
        rewrite_responses=[_FakeMessage(content="refund duplicate charge")],
    )
    graph = build_retriever_graph(embeddings=_FakeEmbeddings(), llm=llm, documents=FAKE_DOCS)

    result = retrieve("I was charged twice, what happens now", graph=graph)

    assert any(r["doc_id"] == "faq_refund_policy" for r in result)


def test_gives_up_gracefully_once_rewrite_budget_is_exhausted(monkeypatch):
    """RETRIEVER_MAX_REWRITES caps how many times this loops -- grading keeps
    saying "no" forever here, and the graph must still terminate (return
    best-effort results) instead of looping, the same loop-guard spirit as
    agents/orchestrator.py's MAX_STEPS."""
    from config import settings
    monkeypatch.setattr(settings, "RETRIEVER_MAX_REWRITES", 1)

    llm = _FakeLLM(
        tool_responses=[
            _tool_call_message("first try", "call_1"),
            _tool_call_message("second try", "call_2"),
        ],
        grade_responses=[GradeDocuments(binary_score="no"), GradeDocuments(binary_score="no")],
        rewrite_responses=[_FakeMessage(content="second try")],
    )
    graph = build_retriever_graph(embeddings=_FakeEmbeddings(), llm=llm, documents=FAKE_DOCS)

    result = retrieve("some question", graph=graph)  # must not hang or raise

    assert isinstance(result, list)


def test_get_doc_text_returns_a_sections_body(tmp_path):
    path = tmp_path / "kb.txt"
    path.write_text("# faq_example\nSome body text.\n\n# faq_other\nOther text.\n", encoding="utf-8")

    assert get_doc_text("faq_example", path=path) == "Some body text."
    assert get_doc_text("does_not_exist", path=path) is None
