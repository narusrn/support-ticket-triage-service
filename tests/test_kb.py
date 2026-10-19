from agents.kb import search_knowledge_base


def test_dark_mode_query_ranks_dark_mode_doc_first():
    results = search_knowledge_base("does the app support dark mode?")
    assert results, "expected at least one match"
    assert results[0]["doc_id"] == "faq_dark_mode"


def test_refund_query_ranks_refund_doc_first():
    results = search_knowledge_base("I was charged twice, can I get a refund?")
    assert results[0]["doc_id"] == "faq_refund_policy"


def test_outage_query_ranks_outage_doc_first():
    results = search_knowledge_base("getting a 500 error, is there an outage?")
    assert results[0]["doc_id"] == "faq_outage_status"


def test_unrelated_query_returns_no_high_confidence_match():
    results = search_knowledge_base("xyzzy quantum teleportation unrelated gibberish")
    assert results == []


def test_results_are_capped_at_top_k():
    results = search_knowledge_base("account plan billing export password", top_k=2)
    assert len(results) <= 2
