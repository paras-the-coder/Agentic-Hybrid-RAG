import os
import sys
import csv
import time
import json
import warnings
from typing import List, Dict, Any, Tuple

import numpy as np

# Ensure stdout uses UTF-8 on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

warnings.filterwarnings("ignore")

# Add project root to path
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from dotenv import load_dotenv
load_dotenv()

dataset_path = os.path.join(_PROJECT_ROOT, "data", "eval", "rag_eval_dataset.csv")

# ---------------------------------------------------------------------------
# Global ChatGroq Rate Limit Protection Monkeypatch
# ---------------------------------------------------------------------------
from langchain_groq import ChatGroq
_original_chatgroq_invoke = ChatGroq.invoke

def _wrapped_chatgroq_invoke(self, *args, **kwargs):
    for attempt in range(6):
        try:
            return _original_chatgroq_invoke(self, *args, **kwargs)
        except Exception as e:
            err_str = str(e)
            if "429" in err_str or "rate_limit" in err_str.lower() or "limit reached" in err_str.lower():
                sleep_time = (2 ** attempt) + 3.0
                print(f"⚠️ Groq rate limit hit for model {self.model_name}. Sleeping for {sleep_time:.1f}s before retry {attempt+1}/6...")
                time.sleep(sleep_time)
            else:
                raise e
    raise Exception("Max attempts exceeded for ChatGroq invocation due to rate limit/api errors.")

ChatGroq.invoke = _wrapped_chatgroq_invoke

def invoke_llm_with_backoff(chain, inputs: Dict[str, Any], max_attempts: int = 5) -> Any:
    """Invokes an LLM chain (protected globally by ChatGroq.invoke monkeypatch)."""
    return chain.invoke(inputs)

# ---------------------------------------------------------------------------
# Sentence Transformers & Embeddings Cache
# ---------------------------------------------------------------------------
_encoder = None
def get_encoder():
    """Lazy-loads the local HuggingFace SentenceTransformer model."""
    global _encoder
    if _encoder is None:
        print("⚙️ Initializing local HuggingFace embedding model (BAAI/bge-small-en-v1.5)...")
        from sentence_transformers import SentenceTransformer
        _encoder = SentenceTransformer("BAAI/bge-small-en-v1.5")
    return _encoder

# ---------------------------------------------------------------------------
# RAG Runners
# ---------------------------------------------------------------------------
def run_basic_rag(question: str) -> Dict[str, Any]:
    """Runs a simulated Basic RAG: Retrieve -> Generate (no grading/critiques)."""
    import src.graph
    from langchain_groq import ChatGroq
    # Override LLM model to avoid TPD limits on llama-3.3-70b-versatile
    src.graph.llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
    
    from src.graph import retrieve, generate
    
    # Initialize basic graph state
    initial_state = {
        "question": question,
        "documents": [],
        "retry_count": 0,
        "critique_feedback": "",
        "confidence": "Low",
        "source": "all"
    }
    
    # Run Retrieve node
    ret_state = retrieve(initial_state)
    
    # Run Generate node
    gen_state = generate(ret_state)
    
    return {
        "answer": gen_state.get("generation", ""),
        "documents": gen_state.get("documents", [])
    }

def run_agentic_rag(question: str) -> Dict[str, Any]:
    """Runs the full Agentic Hybrid RAG graph workflow (grading, rewrite, search, critique, retry)."""
    import src.graph
    from langchain_groq import ChatGroq
    # Override LLM model to avoid TPD limits on llama-3.3-70b-versatile
    src.graph.llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
    
    from src.graph import app
    
    # Expose the app.invoke method
    res = app.invoke({"question": question, "source": "all"})
    
    return {
        "answer": res.get("generation", ""),
        "documents": res.get("documents", [])
    }

# ---------------------------------------------------------------------------
# Metrics Computation
# ---------------------------------------------------------------------------
def calculate_keyword_score(retrieved_text: str, ground_truth: str) -> float:
    """Calculates word-level keyword overlap ratio between ground truth and context."""
    gt_words = set(ground_truth.lower().split())
    ret_words = set(retrieved_text.lower().split())
    if not gt_words:
        return 0.0
    overlap = len(gt_words & ret_words)
    return overlap / len(gt_words)

def calculate_embedding_similarity(text1: str, text2: str) -> float:
    """Calculates cosine similarity using BAAI/bge-small-en-v1.5 embeddings."""
    if not text1.strip() or not text2.strip():
        return 0.0
    
    model = get_encoder()
    emb1 = model.encode(text1)
    emb2 = model.encode(text2)
    
    from sklearn.metrics.pairwise import cosine_similarity
    score = cosine_similarity([emb1], [emb2])[0][0]
    return float(score)

def check_hallucination(retrieved_chunks: str, answer: str) -> str:
    """Asks the LLM to classify if the answer is SUPPORTED, PARTIALLY_SUPPORTED, or UNSUPPORTED."""
    from langchain_groq import ChatGroq
    from langchain_core.prompts import ChatPromptTemplate
    
    prompt = ChatPromptTemplate.from_template(
        "Context:\n{retrieved_chunks}\n\n"
        "Answer:\n{answer}\n\n"
        "Is every statement in the answer supported by the context?\n\n"
        "Return ONLY one of these three exact words:\n"
        "SUPPORTED\n"
        "PARTIALLY_SUPPORTED\n"
        "UNSUPPORTED"
    )
    
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
    chain = prompt | llm
    
    try:
        # Wrap in our robust rate limit helper
        res = invoke_llm_with_backoff(chain, {"retrieved_chunks": retrieved_chunks, "answer": answer})
        val = res.content.strip().upper()
        
        # Exact word extraction
        for label in ["SUPPORTED", "PARTIALLY_SUPPORTED", "UNSUPPORTED"]:
            if label in val:
                return label
        return "UNSUPPORTED"
    except Exception as e:
        print(f"Failed hallucination check LLM call: {e}. Defaulting to UNSUPPORTED.")
        return "UNSUPPORTED"

def check_document_hit(documents: List[Any], expected_document: str) -> float:
    """Checks if the retriever successfully retrieved chunks from the expected source PDF."""
    if not expected_document:
        return 0.0
    
    exp_base = os.path.basename(expected_document).lower().strip()
    for doc in documents:
        doc_source = os.path.basename(doc.metadata.get("source", "")).lower().strip()
        if exp_base in doc_source or doc_source in exp_base:
            return 1.0
    return 0.0

# ---------------------------------------------------------------------------
# Evaluation Runner
# ---------------------------------------------------------------------------
def run_evaluation():
    print("=" * 60)
    print("🧪 CUSTOM PROGRAMMATIC RAG EVALUATION HARNESS")
    print("=" * 60)
    
    # 1. Load dataset
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Evaluation dataset not found at {dataset_path}")
        
    all_pairs = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            all_pairs.append(row)
            
    # Select exactly 10 questions per source document to form the 30-question benchmark
    tsla_pairs = [r for r in all_pairs if r["expected_document"] == "tsla-20251231-gen.pdf"][:10]
    mb_pairs = [r for r in all_pairs if r["expected_document"] == "MembershipHandbook.pdf"][:10]
    aiesl_pairs = [r for r in all_pairs if r["expected_document"] == "Aiesl Employees service regulation.pdf"][:10]
    
    qa_pairs = tsla_pairs + mb_pairs + aiesl_pairs
    print(f"📄 Loaded {len(qa_pairs)} evaluation Q&A pairs (10 per PDF source).")
    
    # Ensure encoder is pre-loaded to prevent timing lag in latency metrics
    get_encoder()
    
    basic_results = []
    agentic_results = []
    
    # --- BASIC RAG RUN ---
    print("\n🚀 Running Basic RAG evaluation...")
    basic_halluc_counts = {}
    for idx, row in enumerate(qa_pairs):
        q = row["question"]
        gt = row["ground_truth"]
        exp_doc = row["expected_document"]
        
        print(f"[{idx+1}/{len(qa_pairs)}] Question: {q[:60]}...")
        
        start = time.time()
        res = run_basic_rag(q)
        latency = time.time() - start
        
        ans = res["answer"]
        docs = res["documents"]
        ret_text = "\n\n".join([doc.page_content for doc in docs])
        
        # Metrics
        kw_score = calculate_keyword_score(ret_text, gt)
        emb_ret_score = calculate_embedding_similarity(ret_text, gt)
        retrieval_score = 0.5 * kw_score + 0.5 * emb_ret_score
        
        ans_sim = calculate_embedding_similarity(ans, gt)
        doc_hit = check_document_hit(docs, exp_doc)
        
        # Sample hallucination checks (first 3 for each document source)
        if basic_halluc_counts.get(exp_doc, 0) < 3:
            print(f"-> Evaluating Hallucination (Sample check for {exp_doc})...")
            halluc = check_hallucination(ret_text, ans)
            basic_halluc_counts[exp_doc] = basic_halluc_counts.get(exp_doc, 0) + 1
        else:
            halluc = "SKIPPED"
        
        basic_results.append({
            "question": q,
            "ground_truth": gt,
            "answer": ans,
            "retrieval_score": retrieval_score,
            "ans_similarity": ans_sim,
            "doc_hit": doc_hit,
            "hallucination": halluc,
            "latency": latency
        })
        
        # Gentle rate limit throttle
        time.sleep(1.0)
        
    # --- AGENTIC RAG RUN ---
    print("\n🚀 Running Agentic Hybrid RAG evaluation...")
    agentic_halluc_counts = {}
    for idx, row in enumerate(qa_pairs):
        q = row["question"]
        gt = row["ground_truth"]
        exp_doc = row["expected_document"]
        
        print(f"[{idx+1}/{len(qa_pairs)}] Question: {q[:60]}...")
        
        start = time.time()
        res = run_agentic_rag(q)
        latency = time.time() - start
        
        ans = res["answer"]
        docs = res["documents"]
        ret_text = "\n\n".join([doc.page_content for doc in docs])
        
        # Metrics
        kw_score = calculate_keyword_score(ret_text, gt)
        emb_ret_score = calculate_embedding_similarity(ret_text, gt)
        retrieval_score = 0.5 * kw_score + 0.5 * emb_ret_score
        
        ans_sim = calculate_embedding_similarity(ans, gt)
        doc_hit = check_document_hit(docs, exp_doc)
        
        # Sample hallucination checks (first 3 for each document source)
        if agentic_halluc_counts.get(exp_doc, 0) < 3:
            print(f"-> Evaluating Hallucination (Sample check for {exp_doc})...")
            halluc = check_hallucination(ret_text, ans)
            agentic_halluc_counts[exp_doc] = agentic_halluc_counts.get(exp_doc, 0) + 1
        else:
            halluc = "SKIPPED"
        
        agentic_results.append({
            "question": q,
            "ground_truth": gt,
            "answer": ans,
            "retrieval_score": retrieval_score,
            "ans_similarity": ans_sim,
            "doc_hit": doc_hit,
            "hallucination": halluc,
            "latency": latency
        })
        
        # Gentle rate limit throttle
        time.sleep(1.5)

    # ---------------------------------------------------------------------------
    # Compile & Calculate Aggregates
    # ---------------------------------------------------------------------------
    def compute_aggregates(results: List[Dict]) -> Dict[str, Any]:
        ret_scores = [r["retrieval_score"] for r in results]
        doc_hits = [r["doc_hit"] for r in results]
        ans_sims = [r["ans_similarity"] for r in results]
        latencies = [r["latency"] for r in results]
        
        # Filter out skipped hallucination checks for strict/lenient calculations
        halluc_results = [r for r in results if r["hallucination"] != "SKIPPED"]
        total_halluc = len(halluc_results)
        
        unsupported = sum(1 for r in halluc_results if r["hallucination"] == "UNSUPPORTED")
        partially = sum(1 for r in halluc_results if r["hallucination"] == "PARTIALLY_SUPPORTED")
        supported = sum(1 for r in halluc_results if r["hallucination"] == "SUPPORTED")
        
        strict_halluc = unsupported / total_halluc if total_halluc else 0.0
        lenient_halluc = (unsupported + partially) / total_halluc if total_halluc else 0.0
        
        return {
            "avg_retrieval": float(np.mean(ret_scores)),
            "doc_hit_rate": float(np.mean(doc_hits)),
            "avg_similarity": float(np.mean(ans_sims)),
            "strict_halluc": strict_halluc,
            "lenient_halluc": lenient_halluc,
            "avg_latency": float(np.mean(latencies)),
            "breakdown": {
                "SUPPORTED": supported,
                "PARTIALLY_SUPPORTED": partially,
                "UNSUPPORTED": unsupported,
                "SKIPPED": len(results) - total_halluc
            }
        }
        
    basic_agg = compute_aggregates(basic_results)
    agentic_agg = compute_aggregates(agentic_results)

    # ---------------------------------------------------------------------------
    # Print Comparison Report
    # ---------------------------------------------------------------------------
    report_md = f"""# 📊 Agentic RAG Performance Ablation Report

This report compares **Basic RAG** against **Agentic Hybrid RAG** on our curated dataset of {len(qa_pairs)} gold questions.

## Summary Comparison Table

| Metric | Basic RAG | Agentic Hybrid RAG |
| :--- | :---: | :---: |
| **Retrieval Score (Hybrid)** | {basic_agg['avg_retrieval']*100:.1f}% | {agentic_agg['avg_retrieval']*100:.1f}% |
| **Document Hit Rate** | {basic_agg['doc_hit_rate']*100:.1f}% | {agentic_agg['doc_hit_rate']*100:.1f}% |
| **Answer Similarity (Cosine)** | {basic_agg['avg_similarity']:.3f} | {agentic_agg['avg_similarity']:.3f} |
| **Strict Hallucination Rate** | {basic_agg['strict_halluc']*100:.1f}% | {agentic_agg['strict_halluc']*100:.1f}% |
| **Lenient Hallucination Rate** | {basic_agg['lenient_halluc']*100:.1f}% | {agentic_agg['lenient_halluc']*100:.1f}% |
| **Average Latency** | {basic_agg['avg_latency']:.2f}s | {agentic_agg['avg_latency']:.2f}s |

---

## 🔍 Metric Interpretations & Key Insights

1. **Retrieval Score:** Combines word-level keyword overlap and embedding similarity against ground truth.
2. **Document Hit Rate:** Proves if the retriever targeted the exact expected source file.
3. **Strict Hallucination Rate:** Percentage of answers that contain completely unsupported claims (`UNSUPPORTED` label). Calculated based on a representative sample of 9 queries (3 per document source).
4. **Lenient Hallucination Rate:** Percentage of answers containing any partially unsupported context (`PARTIALLY_SUPPORTED` + `UNSUPPORTED`). Calculated based on a representative sample of 9 queries (3 per document source).
5. **Latency vs. Accuracy Trade-off:** Shows how the grading, rewrite, Tavily web search fallback, and self-critique/retry loop impact latency while reducing hallucination rates and increasing answer accuracy.
"""

    print("\n" + "="*60)
    print("📊 FINAL ABLATION STUDY RESULTS")
    print("="*60)
    print(report_md)
    print("="*60)

    # Save outputs
    eval_dir = os.path.join(_PROJECT_ROOT, "evaluation")
    os.makedirs(eval_dir, exist_ok=True)
    
    with open(os.path.join(eval_dir, "comparison_report.md"), "w", encoding="utf-8") as f:
        f.write(report_md)
        
    with open(os.path.join(eval_dir, "comparison_results.json"), "w", encoding="utf-8") as f:
        json.dump({
            "basic_rag": {
                "aggregates": basic_agg,
                "per_query": basic_results
            },
            "agentic_rag": {
                "aggregates": agentic_agg,
                "per_query": agentic_results
            }
        }, f, indent=2)
        
    print(f"💾 Report saved to: {os.path.join(eval_dir, 'comparison_report.md')}")
    print(f"💾 Detailed JSON saved to: {os.path.join(eval_dir, 'comparison_results.json')}")

if __name__ == "__main__":
    run_evaluation()
