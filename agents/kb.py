"""Knowledge base search over data/faq/*.md.

Plain TF-IDF + cosine similarity, stdlib only -- no vector DB, no embedding API
call. Matches the spec's "a handful of mock FAQ docs is plenty" and keeps
search_knowledge_base callable with zero network dependency.
"""
import math
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path

FAQ_DIR = Path(__file__).resolve().parent.parent / "data" / "faq"

_TOKEN_RE = re.compile(r"[a-zA-Z0-9]+")


def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text)]


def _load_docs(faq_dir: Path) -> dict[str, str]:
    return {path.stem: path.read_text(encoding="utf-8") for path in sorted(faq_dir.glob("*.md"))}


def _build_index(docs: dict[str, str]) -> tuple[dict[str, dict[str, float]], dict[str, float]]:
    tokenized = {doc_id: _tokenize(text) for doc_id, text in docs.items()}

    doc_freq: Counter = Counter()
    for tokens in tokenized.values():
        doc_freq.update(set(tokens))

    n_docs = len(tokenized) or 1
    idf = {term: math.log((n_docs + 1) / (freq + 1)) + 1 for term, freq in doc_freq.items()}

    doc_vectors = {}
    for doc_id, tokens in tokenized.items():
        tf = Counter(tokens)
        doc_vectors[doc_id] = {term: count * idf[term] for term, count in tf.items()}

    return doc_vectors, idf


@lru_cache(maxsize=1)
def _get_index(faq_dir_str: str) -> tuple[dict[str, dict[str, float]], dict[str, float]]:
    return _build_index(_load_docs(Path(faq_dir_str)))


def _cosine(a: dict[str, float], b: dict[str, float]) -> float:
    common = set(a) & set(b)
    if not common:
        return 0.0
    dot = sum(a[t] * b[t] for t in common)
    norm_a = math.sqrt(sum(v * v for v in a.values())) or 1.0
    norm_b = math.sqrt(sum(v * v for v in b.values())) or 1.0
    return dot / (norm_a * norm_b)


def search_knowledge_base(query: str, top_k: int = 3, min_score: float = 0.05,
                           faq_dir: Path = FAQ_DIR) -> list[dict]:
    doc_vectors, idf = _get_index(str(faq_dir))

    q_tf = Counter(_tokenize(query))
    q_vector = {term: count * idf[term] for term, count in q_tf.items() if term in idf}

    scored = [(doc_id, _cosine(q_vector, vec)) for doc_id, vec in doc_vectors.items()]
    scored.sort(key=lambda pair: -pair[1])

    return [{"doc_id": doc_id, "score": round(score, 4)} for doc_id, score in scored[:top_k] if score >= min_score]
