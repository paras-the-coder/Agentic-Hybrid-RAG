from langchain_core.documents import Document

from src.retrieval import (
    DEFAULT_RRF_K,
    bm25_scores,
    dedupe_with_scores,
    hybrid_rerank,
    rank_candidates,
    reciprocal_rank_fusion,
    tokenize,
)


# --------------------------------------------------------------------------- #
# tokenize
# --------------------------------------------------------------------------- #
def test_tokenize_lowercases_and_drops_stopwords():
    assert tokenize("What IS the Total Revenue of Tesla") == ["total", "revenue", "tesla"]


def test_tokenize_is_not_domain_specific():
    # Domain words like "employee"/"leave" must survive (no overfit stop-words).
    toks = tokenize("employee leave without pay regularization scenarios")
    for word in ["employee", "leave", "pay", "regularization", "scenarios"]:
        assert word in toks


# --------------------------------------------------------------------------- #
# reciprocal_rank_fusion
# --------------------------------------------------------------------------- #
def test_rrf_rewards_items_ranked_high_in_multiple_lists():
    # Item 2 is first in both lists -> must have the highest fused score.
    scores = reciprocal_rank_fusion([[2, 0, 1], [2, 1, 0]])
    assert max(scores, key=scores.get) == 2


def test_rrf_uses_one_based_rank_formula():
    # Single list: top item contributes exactly 1/(k+1).
    scores = reciprocal_rank_fusion([[5]], k=DEFAULT_RRF_K)
    assert abs(scores[5] - 1.0 / (DEFAULT_RRF_K + 1)) < 1e-12


def test_rrf_accumulates_across_lists():
    scores = reciprocal_rank_fusion([[0], [0]], k=60)
    assert abs(scores[0] - 2.0 / 61) < 1e-12


# --------------------------------------------------------------------------- #
# bm25_scores
# --------------------------------------------------------------------------- #
def test_bm25_scores_length_matches_corpus():
    corpus = [["tesla", "revenue"], ["apple", "iphone"], ["tesla", "energy", "storage"]]
    scores = bm25_scores("tesla energy", corpus)
    assert len(scores) == len(corpus)


def test_bm25_handles_empty_documents_without_crashing():
    scores = bm25_scores("anything", [[], ["word"]])
    assert len(scores) == 2


def test_bm25_empty_corpus_returns_empty():
    assert bm25_scores("q", []) == []


# --------------------------------------------------------------------------- #
# dedupe_with_scores
# --------------------------------------------------------------------------- #
def test_dedupe_drops_content_duplicates_and_clamps_scores():
    a = Document(page_content="same text", metadata={"source": "/abs/path/x.pdf"})
    b = Document(page_content="same   text", metadata={"source": "x.pdf"})  # whitespace dup
    c = Document(page_content="different", metadata={"source": "y.pdf"})
    out = dedupe_with_scores([(a, 1.5), (b, 0.9), (c, -0.2)])
    assert len(out) == 2                       # b is a duplicate of a
    assert out[0][0].metadata["source"] == "x.pdf"   # basename-normalized
    assert out[0][1] == 1.0                    # clamped to [0,1]
    assert out[1][1] == 0.0                    # clamped to [0,1]


# --------------------------------------------------------------------------- #
# rank_candidates / hybrid_rerank
# --------------------------------------------------------------------------- #
def _doc(text, cosine, source="d.pdf", page=0):
    return Document(page_content=text, metadata={"source": source, "page": page})


def test_vector_mode_orders_by_cosine():
    cands = [(_doc("a", 0.4), 0.4), (_doc("b", 0.9), 0.9), (_doc("c", 0.6), 0.6)]
    order, aux = rank_candidates(cands, "query", mode="vector")
    assert order == [1, 2, 0]
    assert "bm25" not in aux


def test_hybrid_mode_promotes_lexically_matching_chunk():
    # Doc 0 has the lower cosine but is the only lexical match for the query;
    # hybrid fusion should rank it at or above the higher-cosine non-match.
    cands = [
        (_doc("tesla automotive revenue grew sharply", 0.50), 0.50),
        (_doc("unrelated content about weather", 0.55), 0.55),
    ]
    order, aux = rank_candidates(cands, "tesla automotive revenue", mode="hybrid")
    assert "bm25" in aux and "rrf" in aux
    assert order[0] == 0


def test_hybrid_rerank_preserves_cosine_and_returns_top_n():
    cands = [(_doc(f"chunk {i}", i / 10), i / 10) for i in range(10)]
    top = hybrid_rerank(cands, "chunk", top_n=3, mode="hybrid")
    assert len(top) == 3
    # Raw cosine must remain on metadata["score"] for the agent's routers.
    for d in top:
        assert "score" in d.metadata
        assert "rrf_score" in d.metadata


def test_rerank_empty_input_returns_empty():
    assert hybrid_rerank([], "q") == []
    assert rank_candidates([], "q") == ([], {})
