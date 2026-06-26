import pytest
from langchain_core.documents import Document

# Importing src.graph requires GROQ/TAVILY keys in the environment (loaded from
# .env by conftest). If they are absent, skip this module rather than error.
graph = pytest.importorskip(
    "src.graph",
    reason="src.graph requires GROQ_API_KEY/TAVILY in the environment to import.",
)

from src.graph import (  # noqa: E402
    compute_confidence,
    decide_post_condense,
    decide_post_critique,
    decide_post_generate,
    decide_post_retrieve,
    decide_to_generate,
)


def _doc(score, source="d.pdf"):
    return Document(page_content="x", metadata={"source": source, "score": score})


# --------------------------------------------------------------------------- #
# decide_post_condense  (intent classifier routing)
# --------------------------------------------------------------------------- #
def test_chitchat_routes_straight_to_generate():
    assert decide_post_condense({"intent": "chitchat"}) == "generate"


def test_factual_routes_to_retrieve():
    assert decide_post_condense({"intent": "factual"}) == "retrieve"
    assert decide_post_condense({}) == "retrieve"  # default when unset


# --------------------------------------------------------------------------- #
# decide_post_retrieve  (adaptive fast-path at cosine >= 0.82)
# --------------------------------------------------------------------------- #
def test_fast_path_when_similarity_at_or_above_threshold():
    assert decide_post_retrieve({"documents": [_doc(0.82), _doc(0.5)]}) == "generate"
    assert decide_post_retrieve({"documents": [_doc(0.95)]}) == "generate"


def test_standard_path_when_similarity_below_threshold():
    assert decide_post_retrieve({"documents": [_doc(0.81), _doc(0.4)]}) == "grade_documents"
    assert decide_post_retrieve({"documents": []}) == "grade_documents"


# --------------------------------------------------------------------------- #
# decide_to_generate  (web-search fallback)
# --------------------------------------------------------------------------- #
def test_fallback_routes_to_web_search():
    assert decide_to_generate({"search_fallback": "yes"}) == "transform_query_and_search"


def test_no_fallback_routes_to_generate():
    assert decide_to_generate({"search_fallback": "no"}) == "generate"
    assert decide_to_generate({}) == "generate"


# --------------------------------------------------------------------------- #
# decide_post_generate
# --------------------------------------------------------------------------- #
def test_chitchat_bypasses_critique():
    assert decide_post_generate({"intent": "chitchat", "documents": [_doc(0.9)]}) == "save_history"


def test_fast_path_bypasses_critique():
    state = {"documents": [_doc(0.9)], "retry_count": 0}
    assert decide_post_generate(state) == "save_history"


def test_web_fallback_still_goes_through_critique():
    state = {"documents": [Document(page_content="w", metadata={"source": "web_fallback", "score": 0.85})],
             "retry_count": 0}
    assert decide_post_generate(state) == "critique_generation"


def test_low_similarity_goes_through_critique():
    assert decide_post_generate({"documents": [_doc(0.5)], "retry_count": 0}) == "critique_generation"


# --------------------------------------------------------------------------- #
# decide_post_critique  (retry loop)
# --------------------------------------------------------------------------- #
def test_failed_critique_routes_back_to_generate():
    assert decide_post_critique({"critique_feedback": "missing detail"}) == "generate"


def test_passed_critique_routes_to_save_history():
    assert decide_post_critique({"critique_feedback": ""}) == "save_history"
    assert decide_post_critique({}) == "save_history"


# --------------------------------------------------------------------------- #
# compute_confidence  (heuristic High/Medium/Low)
# --------------------------------------------------------------------------- #
def test_confidence_low_when_no_documents():
    assert compute_confidence({"documents": []}) == "Low"


def test_confidence_low_when_critique_failed():
    assert compute_confidence({"documents": [_doc(0.9)], "critique_feedback": "bad"}) == "Low"


def test_confidence_medium_on_web_fallback():
    assert compute_confidence({"documents": [_doc(0.9)], "search_fallback": "yes"}) == "Medium"


def test_confidence_high_on_strong_local_match():
    state = {"documents": [_doc(0.55)], "retry_count": 0, "search_fallback": "no"}
    assert compute_confidence(state) == "High"


def test_confidence_medium_after_one_retry():
    state = {"documents": [_doc(0.55)], "retry_count": 1, "search_fallback": "no"}
    assert compute_confidence(state) == "Medium"


def test_confidence_low_on_weak_similarity():
    assert compute_confidence({"documents": [_doc(0.30)], "retry_count": 0}) == "Low"
