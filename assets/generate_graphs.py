"""Regenerates assets/graph-overview.png and assets/graph-retriever.png from the
REAL compiled LangGraph objects (agents/orchestrator.py, agents/retriever.py) --
not hand-drawn, so they never drift from the actual node/edge wiring, including
the Command(goto=...) dynamic routing edges a static diagram would miss.

graph-overview.png is rendered with xray=True: agents/orchestrator.py embeds
agents/retriever.py's compiled graph directly as the "kb_retriever_search" node
(not called from inside a function), specifically so this expands it into its
own nested box instead of one opaque node -- same technique the reference
project (narusrn/BBLAgent) uses for its own call_data_retriever node.

Run: python assets/generate_graphs.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_core.documents import Document
from langgraph.checkpoint.memory import InMemorySaver

import agents.retriever as retriever_module
from agents.orchestrator import build_graph
from storage import db

ASSETS_DIR = Path(__file__).resolve().parent


class _FakeEmbeddings:
    def embed_documents(self, texts):
        return [[1.0, 0.0] for _ in texts]

    def embed_query(self, text):
        return [1.0, 0.0]


class _FakeLLM:
    def bind_tools(self, tools):
        return self

    def with_structured_output(self, schema):
        return self


def main() -> None:
    db.init_db(":memory:")
    conn = db.get_connection(":memory:")
    retriever_graph = retriever_module.build_retriever_graph(
        embeddings=_FakeEmbeddings(), llm=_FakeLLM(),
        documents=[Document(page_content="x", metadata={"doc_id": "d"})])

    overview = build_graph(conn, InMemorySaver(), classify_fn=lambda s, m: {},
                            decide_fn=lambda s, m: {}, retriever_graph=retriever_graph)
    (ASSETS_DIR / "graph-overview.png").write_bytes(overview.get_graph(xray=True).draw_mermaid_png())
    (ASSETS_DIR / "graph-retriever.png").write_bytes(retriever_graph.get_graph().draw_mermaid_png())

    print("wrote graph-overview.png and graph-retriever.png")


if __name__ == "__main__":
    main()
