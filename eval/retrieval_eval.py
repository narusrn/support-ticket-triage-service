"""Retrieval-layer benchmark: Hit Rate@k and MRR for the vector search itself,
separate from eval/run_eval.py's end-to-end agent eval.

Why this is a different thing from run_eval.py's kb_retriever_invoked /
LLM-as-judge checks: those two only ever see the FINAL decision, after the
agent's own grade/rewrite loop has already run -- a judge call can still say
"grounded: true" even if the wrong doc got cited, as long as the reply stays
faithful to whatever WAS cited. This script measures the vector search's own
ranking quality directly: for a real customer query, does the correct FAQ
doc_id actually come back, and how high does it rank -- independent of
whether the agent's grade/rewrite loop later decided to use it, retry, or
give up.

Hit Rate@k: fraction of queries where the expected doc_id appears anywhere in
the top k results.
MRR (Mean Reciprocal Rank): average of 1/rank of the expected doc_id's first
appearance (0 if it never appears in the full ranked list).

Uses the real knowledge base and real OpenAI embeddings -- costs a handful of
embedding calls, not free, not part of pytest.

Run: python eval/retrieval_eval.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.retriever import _build_documents
from config import settings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings

# (query, expected_doc_id) -- queries are the real customer text from golden
# tickets that are KB-answerable (eval/golden_tickets.xlsx,
# expected_kb_retriever_invoked=true), joined the same way
# agents/orchestrator.py's call_kb_retriever builds its query (customer
# messages, space-joined) -- not hand-written, so this measures retrieval
# against the same queries the real system actually issues.
GOLDEN_RETRIEVAL_SET = [
    (
        "Hey, just wondering if you support dark mode? No rush "
        "Thanks! Found Settings > Appearance. But I'm on Pro and only see 'Light' and 'System Default' -- no dark mode toggle? "
        "Switched to 'System Default', my Mac is in dark mode, but your app still shows light theme. Bug, or am I missing something? "
        "Random one while I have you -- any way to schedule dark mode, auto-switch at 6pm? Would be cool",
        "faq_dark_mode",
    ),
    ("Hi, how do I export my data as a CSV? I need it for an audit next week, no rush.", "faq_data_export"),
    (
        "I tried resetting my password yesterday but the link had already expired by the time I clicked it. "
        "Could you tell me how to get a working reset link?",
        "faq_account_access",
    ),
    (
        "I just changed my account's timezone. Will this week's Monday digest email go out at the new time, or the old one?",
        "faq_notification_settings",
    ),
    (
        "We're adding 5 new teammates to our Pro-team account partway through the billing cycle. "
        "Will we get charged for those seats right away, and is it the full amount or prorated?",
        "faq_team_seat_management",
    ),
    (
        "I tried logging into the mobile app yesterday but the one-time code never showed up as a push notification. "
        "Could you tell me another way to see the code?",
        "faq_mobile_app_login",
    ),
    (
        "พนักงานใหม่ที่เราเชิญเข้าทีมยังไม่ได้กด accept คำเชิญเลยค่ะ เชิญไปได้ประมาณ 10 วันแล้ว "
        "จะยังใช้ invite link เดิมได้อยู่ไหม หรือต้องเชิญใหม่คะ?",
        "faq_team_seat_management",
    ),
]


def main() -> None:
    documents = _build_documents()
    embeddings = OpenAIEmbeddings(model=settings.EMBEDDING_MODEL, api_key=settings.OPENAI_API_KEY)
    vectorstore = InMemoryVectorStore.from_documents(documents=documents, embedding=embeddings)
    # Rank every doc, not just RETRIEVER_TOP_K=2 -- MRR needs the real rank,
    # and Hit Rate@k is reported at a few k values below.
    full_k = len(documents)

    ranks = []  # rank (1-based) of the expected doc_id's first appearance, or None
    for query, expected_doc_id in GOLDEN_RETRIEVAL_SET:
        results = vectorstore.similarity_search(query, k=full_k)
        seen_doc_ids = [d.metadata["doc_id"] for d in results]
        rank = next((i + 1 for i, doc_id in enumerate(seen_doc_ids) if doc_id == expected_doc_id), None)
        ranks.append(rank)
        print(f"{expected_doc_id:28} rank={rank}   top-3: {seen_doc_ids[:3]}")

    n = len(ranks)
    mrr = sum((1 / r) if r else 0 for r in ranks) / n
    print(f"\nMRR = {mrr:.3f}  (n={n})")
    for k in (1, 2, 3):
        hit_rate = sum(1 for r in ranks if r is not None and r <= k) / n
        print(f"Hit Rate@{k} = {hit_rate:.3f}" + ("  <- production's RETRIEVER_TOP_K" if k == settings.RETRIEVER_TOP_K else ""))


if __name__ == "__main__":
    main()
