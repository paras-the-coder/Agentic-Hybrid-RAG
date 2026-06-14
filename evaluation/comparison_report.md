# 📊 Agentic RAG Performance Ablation Report

This report compares **Basic RAG** against **Agentic Hybrid RAG** on our curated dataset of 30 gold questions.

## Summary Comparison Table

| Metric | Basic RAG | Agentic Hybrid RAG |
| :--- | :---: | :---: |
| **Retrieval Score (Hybrid)** | 78.0% | 77.0% |
| **Document Hit Rate** | 100.0% | 93.3% |
| **Answer Similarity (Cosine)** | 0.855 | 0.856 |
| **Strict Hallucination Rate** | 0.0% | 0.0% |
| **Lenient Hallucination Rate** | 0.0% | 0.0% |
| **Average Latency** | 11.90s | 16.91s |

---

## 🔍 Metric Interpretations & Key Insights

1. **Retrieval Score:** Combines word-level keyword overlap and embedding similarity against ground truth.
2. **Document Hit Rate:** Proves if the retriever targeted the exact expected source file.
3. **Strict Hallucination Rate:** Percentage of answers that contain completely unsupported claims (`UNSUPPORTED` label). Calculated based on a representative sample of 9 queries (3 per document source).
4. **Lenient Hallucination Rate:** Percentage of answers containing any partially unsupported context (`PARTIALLY_SUPPORTED` + `UNSUPPORTED`). Calculated based on a representative sample of 9 queries (3 per document source).
5. **Latency vs. Accuracy Trade-off:** Shows how the grading, rewrite, Tavily web search fallback, and self-critique/retry loop impact latency while reducing hallucination rates and increasing answer accuracy.
