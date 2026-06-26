import argparse
import json
import os
import sys
import time
import warnings
from typing import Any, Dict, List, Tuple

import numpy as np

# UTF-8 console on Windows.
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

warnings.filterwarnings("ignore")

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from dotenv import load_dotenv
load_dotenv()

from src.retrieval import dedupe_with_scores, rank_candidates

GOLD_PATH = os.path.join(_PROJECT_ROOT, "data", "eval", "qa_gold.jsonl")
EVAL_DIR = os.path.join(_PROJECT_ROOT, "evaluation")

# Candidate pool size and final cutoff (mirror the production agent's top-6).
CANDIDATE_K = 60
DEFAULT_TOP_K = 6
# Web-search escalation threshold, mirrored from grade_documents() in src/graph.py.
FALLBACK_THRESHOLD = 0.40


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_gold(path: str = GOLD_PATH) -> Tuple[List[Dict], List[Dict]]:
    """Returns (in_domain, out_of_domain) gold rows."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Gold dataset not found at {path}")
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    in_domain = [r for r in rows if r.get("type") == "in_domain"]
    out_of_domain = [r for r in rows if r.get("type") == "out_of_domain"]
    return in_domain, out_of_domain


# ---------------------------------------------------------------------------
# Relevance + per-query metrics
# ---------------------------------------------------------------------------
def _doc_page(doc) -> int:
    """Stored page metadata is a (possibly float) 0-based index; gold pages use
    the same convention, so compare the integer page directly."""
    page = doc.metadata.get("page", -1)
    try:
        return int(round(float(page)))
    except (TypeError, ValueError):
        return -1


def _is_relevant(doc, relevant_sources: set, relevant_pages: set) -> bool:
    source = os.path.basename(doc.metadata.get("source", "")).lower()
    if relevant_sources and source not in relevant_sources:
        return False
    return _doc_page(doc) in relevant_pages


def score_query(ranked_docs: List[Any], gold: Dict, top_k: int) -> Dict[str, float]:
    """Compute Precision@K, Recall@K, MRR, Hit-Rate, Source-Hit for one query."""
    relevant_sources = {s.lower() for s in gold.get("relevant_sources", [])}
    relevant_pages = {int(p) for p in gold.get("relevant_pages", [])}

    top = ranked_docs[:top_k]
    flags = [_is_relevant(d, relevant_sources, relevant_pages) for d in top]
    num_relevant_retrieved = sum(flags)

    precision = num_relevant_retrieved / top_k if top_k else 0.0

    # Recall over distinct gold pages actually covered by the top-K chunks.
    covered_pages = {
        _doc_page(d)
        for d in top
        if _is_relevant(d, relevant_sources, relevant_pages)
    }
    recall = (len(covered_pages) / len(relevant_pages)) if relevant_pages else 0.0

    mrr = 0.0
    for rank, is_rel in enumerate(flags, start=1):
        if is_rel:
            mrr = 1.0 / rank
            break

    hit_rate = 1.0 if num_relevant_retrieved > 0 else 0.0
    source_hit = 1.0 if any(
        os.path.basename(d.metadata.get("source", "")).lower() in relevant_sources
        for d in top
    ) else 0.0

    return {
        "precision_at_k": precision,
        "recall_at_k": recall,
        "mrr": mrr,
        "hit_rate": hit_rate,
        "source_hit": source_hit,
    }


# ---------------------------------------------------------------------------
# Retrieval (shared candidate pool, reused across both configs)
# ---------------------------------------------------------------------------
def get_candidate_pool(vectorstore, question: str) -> List[Tuple[Any, float]]:
    """One Pinecone call -> deduped (doc, cosine) candidates, reused per config."""
    raw = vectorstore.similarity_search_with_score(question, k=CANDIDATE_K)
    return dedupe_with_scores(raw)


def evaluate_config(
    mode: str,
    in_domain: List[Dict],
    out_of_domain: List[Dict],
    pools_in: List[List[Tuple[Any, float]]],
    pools_ood: List[List[Tuple[Any, float]]],
    top_k: int,
) -> Dict[str, Any]:
    """Score one reranking ``mode`` ("vector" or "hybrid") over cached pools."""
    per_query: List[Dict[str, float]] = []
    for gold, pool in zip(in_domain, pools_in):
        order, _ = rank_candidates(pool, gold["question"], mode=mode)
        ranked_docs = [pool[i][0] for i in order]
        per_query.append(score_query(ranked_docs, gold, top_k))

    def _mean(key: str) -> float:
        return float(np.mean([m[key] for m in per_query])) if per_query else 0.0

    aggregates = {
        "precision_at_k": _mean("precision_at_k"),
        "recall_at_k": _mean("recall_at_k"),
        "mrr": _mean("mrr"),
        "hit_rate": _mean("hit_rate"),
        "source_hit": _mean("source_hit"),
    }

    # Out-of-domain: fallback trigger = best cosine below the escalation threshold.
    fallback_flags: List[float] = []
    for pool in pools_ood:
        max_cosine = max((c for _, c in pool), default=0.0)
        fallback_flags.append(1.0 if max_cosine < FALLBACK_THRESHOLD else 0.0)
    fallback_rate = float(np.mean(fallback_flags)) if fallback_flags else 0.0

    return {
        "in_domain": {
            "n_queries": len(in_domain),
            "top_k": top_k,
            "aggregates": aggregates,
            "per_query_mrr": [m["mrr"] for m in per_query],
        },
        "out_of_domain": {
            "n_queries": len(out_of_domain),
            "fallback_threshold": FALLBACK_THRESHOLD,
            "fallback_trigger_rate": fallback_rate,
        },
    }


# ---------------------------------------------------------------------------
# Significance test
# ---------------------------------------------------------------------------
def wilcoxon_mrr(vector_mrr: List[float], hybrid_mrr: List[float]) -> Dict[str, Any]:
    """Wilcoxon signed-rank test on paired per-query MRR (hybrid vs. vector)."""
    from scipy.stats import wilcoxon

    diffs = [h - v for h, v in zip(hybrid_mrr, vector_mrr)]
    n_nonzero = sum(1 for d in diffs if d != 0)

    if n_nonzero == 0:
        return {
            "statistic": 0.0,
            "p_value": 1.0,
            "n_pairs": len(diffs),
            "n_nonzero_diffs": 0,
            "significant_at_0.05": False,
            "note": "All paired MRR differences are zero; reranking did not change the first-relevant rank on any query.",
        }
    try:
        stat, p = wilcoxon(hybrid_mrr, vector_mrr, zero_method="wilcox")
    except ValueError as exc:
        return {
            "statistic": None,
            "p_value": None,
            "n_pairs": len(diffs),
            "n_nonzero_diffs": n_nonzero,
            "significant_at_0.05": False,
            "note": f"Wilcoxon could not be computed: {exc}",
        }
    return {
        "statistic": float(stat),
        "p_value": float(p),
        "n_pairs": len(diffs),
        "n_nonzero_diffs": n_nonzero,
        "significant_at_0.05": bool(p < 0.05),
        "mean_mrr_delta": float(np.mean(diffs)),
    }


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
def run(top_k: int = DEFAULT_TOP_K) -> Dict[str, Any]:
    print("=" * 64)
    print("🔬 DETERMINISTIC RETRIEVAL ABLATION  (vector-only vs. BM25-hybrid)")
    print("   Zero LLM calls — embeddings + Pinecone + BM25/RRF only.")
    print("=" * 64)

    in_domain, out_of_domain = load_gold()
    print(f"📄 Gold: {len(in_domain)} in-domain + {len(out_of_domain)} out-of-domain queries. Top-K={top_k}.")

    from src.database import get_vectorstore
    vectorstore = get_vectorstore()

    # Build candidate pools once (shared by both configs).
    print("⚙️  Building dense candidate pools (one Pinecone query per question)...")
    t0 = time.time()
    pools_in = [get_candidate_pool(vectorstore, r["question"]) for r in in_domain]
    pools_ood = [get_candidate_pool(vectorstore, r["question"]) for r in out_of_domain]
    elapsed = time.time() - t0
    print(f"   Retrieved pools for {len(pools_in) + len(pools_ood)} queries in {elapsed:.1f}s.")

    vector_res = evaluate_config("vector", in_domain, out_of_domain, pools_in, pools_ood, top_k)
    hybrid_res = evaluate_config("hybrid", in_domain, out_of_domain, pools_in, pools_ood, top_k)

    significance = wilcoxon_mrr(
        vector_res["in_domain"]["per_query_mrr"],
        hybrid_res["in_domain"]["per_query_mrr"],
    )

    results = {
        "config": {"candidate_k": CANDIDATE_K, "top_k": top_k, "rrf_k": 60},
        "vector_only": vector_res,
        "bm25_hybrid": hybrid_res,
        "significance_test": {
            "method": "wilcoxon_signed_rank",
            "paired_on": "per_query_mrr",
            **significance,
        },
    }

    _print_report(results)

    os.makedirs(EVAL_DIR, exist_ok=True)
    out_path = os.path.join(EVAL_DIR, "ablation_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n💾 Saved: {out_path}")

    return results


def _fmt_pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def _print_report(results: Dict[str, Any]) -> None:
    v = results["vector_only"]["in_domain"]["aggregates"]
    h = results["bm25_hybrid"]["in_domain"]["aggregates"]
    sig = results["significance_test"]

    print("\n" + "=" * 64)
    print("📊 RETRIEVAL ABLATION RESULTS")
    print("=" * 64)
    rows = [
        ("Precision@K", "precision_at_k"),
        ("Recall@K", "recall_at_k"),
        ("MRR", "mrr"),
        ("Hit-Rate", "hit_rate"),
        ("Source-Hit", "source_hit"),
    ]
    print(f"{'Metric':<14}{'Vector-only':>14}{'BM25-hybrid':>14}{'Δ':>10}")
    print("-" * 52)
    for label, key in rows:
        delta = h[key] - v[key]
        print(f"{label:<14}{_fmt_pct(v[key]):>14}{_fmt_pct(h[key]):>14}{delta * 100:>+9.1f}%")

    vf = results["vector_only"]["out_of_domain"]["fallback_trigger_rate"]
    hf = results["bm25_hybrid"]["out_of_domain"]["fallback_trigger_rate"]
    print(f"\nFallback-Trigger Rate (OOD): vector={_fmt_pct(vf)}  hybrid={_fmt_pct(hf)}")

    print("\nSignificance (Wilcoxon signed-rank on paired MRR):")
    if sig.get("p_value") is None:
        print(f"  {sig.get('note')}")
    else:
        verdict = "SIGNIFICANT (p<0.05)" if sig.get("significant_at_0.05") else "not significant"
        print(f"  statistic={sig['statistic']}, p={sig['p_value']:.4f} -> {verdict}")
        if sig.get("note"):
            print(f"  note: {sig['note']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Deterministic retrieval ablation harness.")
    parser.add_argument("--k", type=int, default=DEFAULT_TOP_K, help="Top-K cutoff (default 6).")
    args = parser.parse_args()
    run(top_k=args.k)


if __name__ == "__main__":
    main()
