import os
import re
from typing import Any, Dict, List, Tuple

from langchain_core.documents import Document

# Default RRF constant (the "60" in 1/(60+rank)).
DEFAULT_RRF_K = 60

# A small, generic English stop-word list. Kept deliberately domain-agnostic so
# lexical scoring is not tuned to any particular question or document.
_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "have",
    "how", "in", "into", "is", "it", "its", "of", "on", "or", "that", "the", "to",
    "was", "were", "what", "when", "where", "which", "who", "why", "will", "with",
    "this", "these", "those", "their", "they", "you", "your", "i", "we", "our",
}


def tokenize(text: str) -> List[str]:
    """Lower-cases, extracts word tokens, and drops generic English stop-words."""
    return [w for w in re.findall(r"\b\w+\b", text.lower()) if w not in _STOPWORDS]


def reciprocal_rank_fusion(rankings: List[List[int]], k: int = DEFAULT_RRF_K) -> Dict[int, float]:
    """
    Merge several ranked lists of item ids using Reciprocal Rank Fusion.

    Each ``rankings`` element is a list of item indices ordered best-first. The
    returned dict maps each item index to its summed RRF score, where an item at
    1-based rank ``r`` in a list contributes ``1 / (k + r)``.
    """
    scores: Dict[int, float] = {}
    for ranking in rankings:
        for rank, idx in enumerate(ranking, start=1):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank)
    return scores


def bm25_scores(query: str, corpus_tokens: List[List[str]]) -> List[float]:
    """BM25Okapi score of ``query`` against each pre-tokenized document."""
    if not corpus_tokens:
        return []
    from rank_bm25 import BM25Okapi

    # BM25Okapi cannot handle empty documents; substitute a sentinel token.
    safe_corpus = [toks if toks else ["__empty__"] for toks in corpus_tokens]
    bm25 = BM25Okapi(safe_corpus)
    query_tokens = tokenize(query) or ["__empty__"]
    return [float(s) for s in bm25.get_scores(query_tokens)]


def dedupe_with_scores(
    docs_with_scores: List[Tuple[Document, float]],
) -> List[Tuple[Document, float]]:
    """
    Normalize source paths to basenames, clamp cosine to [0, 1], stamp it onto
    ``metadata["score"]``, and drop content duplicates (keeping the first/best).
    """
    seen: set = set()
    unique: List[Tuple[Document, float]] = []
    for doc, score in docs_with_scores:
        if "source" in doc.metadata:
            doc.metadata["source"] = os.path.basename(doc.metadata["source"])
        cleaned = " ".join(doc.page_content.split())
        if cleaned in seen:
            continue
        seen.add(cleaned)
        similarity = max(0.0, min(1.0, float(score)))
        doc.metadata["score"] = similarity
        unique.append((doc, similarity))
    return unique


def rank_candidates(
    candidates: List[Tuple[Document, float]],
    query: str,
    mode: str = "hybrid",
    rrf_k: int = DEFAULT_RRF_K,
) -> Tuple[List[int], Dict[str, Any]]:
    """
    Return the full reranked ordering (best-first list of candidate indices) for
    a retrieval ``mode`` plus the auxiliary per-index scores used.

    mode="vector"  -> order by dense cosine only.
    mode="hybrid"  -> order by RRF(cosine-rank, bm25-rank).
    """
    n = len(candidates)
    if n == 0:
        return [], {}

    cosines = [c for _, c in candidates]
    vector_ranking = sorted(range(n), key=lambda i: cosines[i], reverse=True)

    if mode == "vector":
        return vector_ranking, {"cosine": cosines}

    docs = [d for d, _ in candidates]
    corpus_tokens = [tokenize(d.page_content) for d in docs]
    bm25 = bm25_scores(query, corpus_tokens)
    bm25_ranking = sorted(range(n), key=lambda i: bm25[i], reverse=True)

    rrf = reciprocal_rank_fusion([vector_ranking, bm25_ranking], k=rrf_k)
    order = sorted(range(n), key=lambda i: rrf.get(i, 0.0), reverse=True)
    return order, {"cosine": cosines, "bm25": bm25, "rrf": rrf}


def hybrid_rerank(
    candidates: List[Tuple[Document, float]],
    query: str,
    top_n: int = 6,
    mode: str = "hybrid",
    rrf_k: int = DEFAULT_RRF_K,
) -> List[Document]:
    """
    Rerank deduped (doc, cosine) candidates and return the top ``top_n`` docs.

    Each returned doc carries ``metadata["score"]`` (raw cosine, preserved for
    the agent's routing), and in hybrid mode also ``metadata["bm25_score"]`` and
    ``metadata["rrf_score"]`` for transparency/debugging.
    """
    if not candidates:
        return []

    order, aux = rank_candidates(candidates, query, mode=mode, rrf_k=rrf_k)
    docs = [d for d, _ in candidates]
    cosines = [c for _, c in candidates]

    for i, doc in enumerate(docs):
        doc.metadata["score"] = cosines[i]
        if "bm25" in aux:
            doc.metadata["bm25_score"] = float(aux["bm25"][i])
        if "rrf" in aux:
            doc.metadata["rrf_score"] = float(aux["rrf"].get(i, 0.0))

    return [docs[i] for i in order[:top_n]]
