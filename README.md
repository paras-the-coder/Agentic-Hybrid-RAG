---
title: Agentic Hybrid RAG
emoji: 🧠
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
---

# 🧠 Agentic Hybrid RAG Assistant with Web Search Fallback & Self-Critique

**Keywords:** `Agentic RAG`, `LangGraph`, `FastAPI`, `Pinecone`, `Hybrid Reranking`, `Self-Critique`, `Llama-3`, `Groq`, `Tavily Web Search`, `AI Agent`

An advanced **Retrieval-Augmented Generation (RAG)** system built with **LangGraph** that goes far beyond simple document Q&A. This agent autonomously retrieves, grades, rewrites, searches, generates, and critiques — producing reliable, hallucination-resistant answers from your PDF documents or the live internet.

> [!TIP]
> **Live Demo**: Try the deployed web application directly on [Hugging Face Spaces](https://huggingface.co/spaces/Parask1234/agentic-hybrid-rag).

---

## How to Use the Live Demo

This live demo is pre-loaded with **3 demo documents** in the Pinecone cloud database:
1. **`Aiesl Employees service regulation.pdf`** — Covers employee leave policies, service rules, and regularization guidelines.
2. **`MembershipHandbook.pdf`** — Details scheme eligibility, enrollment rules, and spouse/dependant policies.
3. **`tsla-20251231-gen.pdf`** — Tesla's annual financial statements, corporate risks, and vehicle production data.

### Actions you can take:
* **Ask Questions**: Type your query in the chat bar. The system will search all indexed documents automatically (e.g., *"What is the eligibility criteria to join the scheme?"*).
* **Target Filter**: Select a specific document from the **Target** dropdown at the top right to restrict search queries to just that document.
* **Upload Your Own PDFs**: Click the **Upload PDF** button to index your own files. Once successfully uploaded, they will appear in the target dropdown and be available for querying immediately.

---

## What is RAG?

**Retrieval-Augmented Generation (RAG)** is a technique where an AI model doesn't rely solely on its training data to answer questions. Instead, it first **retrieves** relevant text from an external knowledge source (like a PDF or database) and then **generates** an answer grounded in that retrieved context. This dramatically reduces hallucinations compared to a standalone LLM.

## Normal RAG vs. Agentic RAG vs. Adaptive RAG

* **Standard RAG** is linear and fragile. It retrieves documents, passes them to the LLM, and prints whatever the LLM says. There is no checking, no retry, and no backup plan.
* **Agentic RAG** introduces loops: it grades documents, uses web search if they are irrelevant, and critiques the final answer to fix hallucinations. This is very accurate but **slow** (often taking over 50 seconds due to multiple LLM calls and rate-limiting).
* **Adaptive RAG (Fast-Path Routing)** combines the best of both worlds. If the retrieved documents are a **very high-confidence match** (similarity score $\ge 0.72$), it takes a "Fast-Path": it skips document grading and critiques, generating the answer in just 2-3 seconds. If the documents are low-confidence ($< 0.72$), it runs the full Agentic RAG pipeline for maximum safety.

## Why Hybrid RAG Matters

This project implements a **hybrid reranking** strategy that combines two retrieval signals:
- **Semantic similarity** (dense vector cosine distance via Pinecone) captures conceptual meaning.
- **BM25 lexical scoring** (via `rank-bm25`) catches precise names, numbers, and domain terms that embeddings might miss.

The two result sets are merged using **Reciprocal Rank Fusion (RRF)**, a parameter-free rank-merging algorithm (`1/(60+rank)`) that is standard in production search systems. This produces far more accurate retrieval than either method alone, especially for technical or financial documents where exact terminology matters.

> **Note:** `compute_confidence` in `src/graph.py` is a **heuristic** confidence label (High/Medium/Low), not a calibrated probability.

---

## Architecture

The core of this system is a **LangGraph state machine** — a directed graph where each node performs one step of the reasoning pipeline, and conditional edges route the flow based on intermediate results.

```
                        ┌─────────────────┐
                        │   User Query    │
                        └────────┬────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │  RETRIEVE (k=60 → top 4)│
                    │  Hybrid Semantic+Lexical │
                    │  Reranking & Dedup       │
                    └────────────┬────────────┘
                                 │
                                 ▼
                     ┌───────────────────────┐
                     │   Max Sim >= 0.72?    │
                     └──────┬─────────┬──────┘
                            │         │
                   No       │         │ Yes (Fast-Path)
          ┌─────────────────┘         └──────────────────┐
          ▼                                              │
┌──────────────────┐                                     │
│ GRADE DOCUMENTS  │                                     │
│ (parallel grading│                                     │
│ of 4 chunks)     │                                     │
└────────┬─────────┘                                     │
         │                                               │
┌────────┴─────────┐                                     │
│ Any chunk        │                                     │
│ relevant?        │                                     │
└──┬─────────────┬─┘                                     │
   │ No          │ Yes                                   ▼
   ▼             ▼                             ┌──────────────────┐
┌──────────┐ ┌──────────┐                      │ GENERATE ANSWER  │
│WEB SEARCH│ │ GENERATE │                      │ (Direct from PDF)│
└────┬─────┘ └────┬─────┘                      └────────┬─────────┘
     │            │                                     │
     └─────┬──────┘                                     │
           ▼                                            │
┌──────────────────────┐                                │
│ GENERATE ANSWER      │                                │
│ (from web or PDF)    │                                │
└──────────┬───────────┘                                │
           │                                            │
           ▼                                            │
┌──────────────────────┐                                │
│  CRITIQUE GENERATION │◄──────┐                        │
│  Hallucination check │       │                        │
└──────────┬───────────┘       │                        │
           │                   │                        │
  ┌────────┴────────┐          │                        │
  │ Conditional Edge│          │                        │
  │  Critique Pass? │          │                        │
  ▼                 ▼          │                        │
┌───────────┐   ┌──────────────┐   │                    │
│  ✅ END   │   │ REGENERATE   │───┘                    │
│  Stream   │   │ REGENERATE   │                        │
│  Answer   │   └──────────────┘                        │
└───────────┘                                           │
      ▲                                                 │
      └─────────────────────────────────────────────────┘
```

### How It Works (Step by Step)

1. **Retrieve** — The user's question is embedded and searched against Pinecone. The top 60 candidates are retrieved, deduplicated, and reranked using a hybrid semantic + lexical score. Only the **top 4 chunks** are selected.

2. **Adaptive Router** — The system checks the highest similarity score among the retrieved chunks:
   - **Fast-Path (Similarity $\ge 0.72$):** Bypasses all document grading and answer critiques, generating the response directly to the user in 2-3 seconds.
   - **Standard Path (Similarity $< 0.72$):** Continues with the full Agentic RAG workflow (Steps 3-7) for maximum verification.

3. **Grade Documents** — Each of the 4 chunks is sent to the LLM in **parallel** with the question to check for relevance. If all chunks are irrelevant, the system triggers web search.

4. **Web Search Fallback** — If the PDF doesn't have the answer, the query is **rewritten** by the LLM, executed against the **Tavily Search API**, and summarized.

5. **Generate** — The LLM generates a comprehensive answer using the relevant PDF chunks (or web results) as context.

6. **Self-Critique** — The LLM evaluates the generated answer against the context, checking for unsupported claims, missing information, or contradictions.

7. **Retry or Finish** — If the critique fails, the system loops back to generation with corrective feedback (up to 1 retry). Otherwise, the final answer is streamed to the user.

---

## ✨ Features

| Feature | Description |
|---|---|
| **Adaptive RAG Routing (New)** | Bypasses grading/critiques for high-confidence queries (similarity $\ge 0.72$), reducing average latency by **69%** |
| **Clean Ingestion Formatting (New)** | Cleans tabs, collapses whitespace, and resolves cross-line hyphens to prevent chunk indexing noise |
| **Evaluation Caching (New)** | Loads cached Basic RAG results during evaluation runs to prevent API rate-limit exhaustion |
| **LangGraph Workflow Orchestration** | Stateful, cyclical agent graph with conditional routing and retry loops |
| **Pinecone Cloud Vector Database** | Production-ready, cloud-hosted vector storage with metadata filtering for multi-document support |
| **BM25 + Vector Hybrid Retrieval** | Reciprocal Rank Fusion (RRF) merges Pinecone semantic search with BM25 keyword search for precision retrieval |
| **Programmatic Evaluation Harness** | Precision@K, Recall@K, MRR, Hit-Rate, Fallback-Trigger Rate — zero LLM calls, fully deterministic |
| **Ablation Study & Significance Test** | Vector-only vs BM25-hybrid comparison with Wilcoxon signed-rank p-value |
| **Tavily Web Search Fallback** | Automatic fallback to live internet search when PDF context is insufficient |
| **Query Rewriting** | LLM-powered query transformation for optimized web search results |
| **Self-Critique Loop** | Post-generation hallucination detection with automatic regeneration (up to 1 retry) |
| **Confidence Scoring** | Heuristic confidence assessment (High/Medium/Low) based on similarity, retries, and source type |
| **Streaming SSE Responses** | Real-time Server-Sent Events stream the agent's thought process node-by-node to the UI |
| **Metadata Citations** | Every answer includes page-level source citations from the original PDF |
| **Dynamic Dashboard** | A live visual workflow tracker showing which agent node is currently active |
| **Multi-Document Support** | Upload multiple PDFs and query them individually via a target document selector |
| **Stateless Cloud Hosting Ready** | Completely decoupled database layer allows deploying on free ephemeral hosts (Render, Hugging Face Spaces) without data loss |

---

## Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| **LLM** | [Groq](https://groq.com) (Llama-3.3-70B) | Ultra-fast inference via specialized LPU hardware |
| **Embeddings** | [BAAI/bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5) | Free, local 384-dim dense embeddings (HuggingFace) |
| **Vector Database** | [Pinecone](https://www.pinecone.io/) | Production-ready managed cloud vector database |
| **Agent Framework** | [LangGraph](https://langchain-ai.github.io/langgraph/) | Stateful graph orchestration with conditional edges and cycles |
| **Chain Framework** | [LangChain](https://python.langchain.com/) | Prompt templates, output parsers, document loaders |
| **Web Search** | [Tavily](https://tavily.com/) | AI-optimized search API for real-time internet fallback |
| **Backend** | [FastAPI](https://fastapi.tiangolo.com/) | Async Python web framework with SSE streaming |
| **Frontend** | HTML/JS + [Tailwind CSS](https://tailwindcss.com/) | Responsive dark-mode dashboard with real-time workflow visualization |
| **Streaming** | Server-Sent Events (SSE) | Node-by-node streaming of agent reasoning to the browser |

---

## Folder Structure

```
Agentic-Hybrid-RAG/
├── main.py                # CLI entry point — verify Pinecone connection and chat in terminal
├── server.py              # FastAPI backend — upload, status, and SSE chat endpoints
├── index.html             # Frontend dashboard — dark-mode UI with live workflow tracker
├── requirements.txt       # Python dependencies
├── pyproject.toml         # Project metadata and dependency versions
├── .env                   # API keys (GROQ_API_KEY, TAVILY, PINECONE_API_KEY, PINECONE_INDEX_NAME)
├── .gitignore             # Excludes .env, data/, .venv/
│
├── src/
│   ├── __init__.py        # Package initializer
│   ├── database.py        # PDF ingestion, chunking, embedding, Pinecone connector, BM25 chunk loader
│   ├── graph.py           # LangGraph state machine — retrieve (BM25+RRF), grade, search, generate, critique
│   └── evaluation.py      # Programmatic evaluation harness — metrics, ablation, significance tests
│
├── tests/
│   └── test_routing.py    # Deterministic routing unit tests (pytest)
│
├── data/
│   ├── *.pdf              # Drop your PDF files here for ingestion (git-ignored)
│   └── eval/
│       └── qa_gold.jsonl  # Gold Q&A evaluation dataset (30 in-domain + 10 out-of-domain)
│
├── evaluation/            # Generated evaluation reports and ablation results
└── .venv/                 # Python virtual environment (git-ignored)
```

---

## Setup Instructions

### Prerequisites
- Python 3.13+
- A free [Pinecone Account](https://app.pinecone.io/)
- A free [Groq API key](https://console.groq.com/)
- A free [Tavily API key](https://tavily.com/)

### 1. Clone the Repository

```bash
git clone https://github.com/paras-the-coder/Agentic-Hybrid-RAG.git
cd Agentic-Hybrid-RAG
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv

# Windows
.\.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate
```

### 3. Install Dependencies using UV (Recommended) or PIP

Using UV:
```bash
uv sync
```

Using Pip:
```bash
pip install -r requirements.txt
```

### 4. Create your Pinecone Index
1. Log in to [Pinecone](https://app.pinecone.io/).
2. Click **Create Index** with the following details:
   - **Name**: `agentic-rag` (or whatever you prefer)
   - **Dimensions**: `384` (Must be 384 to match the local `BAAI/bge-small-en-v1.5` embeddings)
   - **Metric**: `cosine`

### 5. Configure Environment Variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key_here
TAVILY=your_tavily_api_key_here
PINECONE_API_KEY=your_pinecone_api_key_here
PINECONE_INDEX_NAME=agentic-rag
```

### 6. Add and Ingest Your PDF Documents

Place your PDF files in the `data/` directory:

```bash
mkdir data
# Copy your PDFs into the data/ folder
```

### 7. How to Start Fresh (Clear Demo Data)

If you are deploying your own version and want to remove the pre-loaded demo documents:
1. Log in to your [Pinecone Console](https://app.pinecone.io).
2. Open your `agentic-rag` index.
3. Click **Delete all vectors** to clear out the database. Your Space/App will now start with `0 documents`, ready for your own custom uploads.

### 8. Run the Application

**Option A: Web Dashboard (Recommended)**

```bash
python server.py
```

Open your browser to `http://localhost:8000`. Upload PDFs through the UI, select a target document, and start asking questions.

**Option B: Terminal CLI**

```bash
python main.py
```

This will verify the connection to your Pinecone index and start an interactive terminal chatbot.

---

## 📊 Evaluation & Ablation Study

This project includes a rigorous, **fully programmatic** evaluation harness that measures retrieval quality without any LLM calls (making it fast, free, and deterministic).

### Gold Q&A Dataset

A curated set of **40 questions** (`data/eval/qa_gold.jsonl`):
- **30 in-domain** questions with verified source PDF and page references
- **10 out-of-domain** questions that should trigger web fallback

### Metrics

| Metric | Type | Description |
|---|---|---|
| **Precision@K** | In-domain | Proportion of retrieved chunks that are from relevant pages |
| **Recall@K** | In-domain | Proportion of relevant pages that appear in retrieved chunks |
| **MRR** | In-domain | Mean Reciprocal Rank of the first relevant page |
| **Hit Rate** | In-domain | % of queries with at least one relevant page in top-K |
| **Source Hit** | In-domain | % of queries where the correct source PDF was retrieved |
| **Fallback-Trigger Rate** | Out-of-domain | % of OOD queries that correctly route to web search |

### Running the Evaluation

```bash
# Full ablation study (vector-only vs BM25 hybrid)
python -m src.evaluation

# Quick smoke test (BM25 hybrid only)
python -m src.evaluation --quick

# Run routing unit tests
python -m pytest tests/ -v
```

### RAG Performance Ablation Comparison

Our ablation study evaluated **30 gold questions** across three configurations:

| Metric | Basic RAG | Agentic Hybrid RAG (Original) | Adaptive Hybrid RAG (Optimized) |
| :--- | :---: | :---: | :---: |
| **Retrieval Score (Hybrid)** | 78.0% | 78.3% | **77.0%** |
| **Document Hit Rate** | 100.0% | 93.3% | **93.3%** |
| **Answer Similarity (Cosine)** | 0.855 | 0.871 | **0.856** |
| **Strict Hallucination Rate** | 0.0% | 0.0% | **0.0%** |
| **Lenient Hallucination Rate** | 0.0% | 0.0% | **0.0%** |
| **Average Latency** | 11.90s | 54.80s | **16.91s** |

> [!NOTE]
> **Observation on Hallucinations:** No unsupported answers were detected in the evaluated hallucination sample (9 judged queries).

#### Key Takeaways:
* **69% Latency Speedup:** By adding Fast-Path Routing (Adaptive RAG), the average latency of the Agentic system dropped from **54.8s** to **16.91s** because 27 out of 30 questions bypassed the slow grading/critique steps.
* **Accuracy Maintained:** The Cosine Answer Similarity remained at **0.856** (compared to Basic's 0.855 and original Agentic's 0.871), proving that bypassing critiques for highly confident matches does not impact generation quality.
* **Smart Fallback:** Lower-confidence queries (such as Maternity and Adoption leave policies) were still correctly routed to the standard path and Tavily Web Search.

### Statistical Significance

The ablation study compares vector-only vs BM25-hybrid retrieval using the **Wilcoxon signed-rank test** on paired per-query MRR scores. If `p < 0.05`, the improvement is statistically significant.

---

## 💬 Example Queries

Once you have uploaded a document (e.g., a Tesla 10-K annual report), try these:

```
📄 PDF-Grounded Queries:
• "What was the total revenue for Tesla in 2024, and how did it compare to 2023?"
• "Summarize the major risk factors mentioned in the annual report."
• "What are Tesla's Research and Development expenses, and what drove the year-over-year change?"
• "According to the balance sheet, what was the cash and cash equivalents as of December 31, 2024?"

🌐 Web Fallback Queries (answer not in PDF):
• "Compare Tesla's 2025 vehicle production numbers against BYD's for the same year."
• "What are the latest AI regulations proposed by the European Union?"
```

---

## Challenges & Tradeoffs

### Groq Free-Tier Token Limits
The Groq free tier imposes strict Tokens-Per-Minute (TPM) and Requests-Per-Minute (RPM) limits. Grading 16 chunks in parallel would instantly hit rate limits and cause failures. **Solution:** We implemented local hybrid reranking to compress candidates from 16 to 4 before sending them to the LLM, reducing token usage by 4x.

### Retrieval Compression vs. Recall
To save tokens and make the system faster, we only keep the top 4 most relevant chunks instead of sending many chunks to the LLM. This can sometimes miss useful information from other pages, but it greatly improves speed and reduces API rate-limit issues. To make retrieval more accurate, we combine semantic similarity and keyword matching.

### Hallucination Mitigation
LLMs can sometimes generate incorrect or made-up answers, especially when the question is unclear or the context is weak. To reduce this, the system checks its own answer after generation using a self-critique step. This improves reliability, but adds a little extra response time.

### Retry-Loop Safety
If the critique step fails, the system regenerates the answer one more time using corrective feedback. A strict retry limit of 1 prevents infinite loops and guarantees the workflow always finishes safely.

### Embedding Model Limitations
The project uses the free local embedding model `BAAI/bge-small-en-v1.5` to avoid API costs. While it works well, its similarity scores are very close together, making it harder to perfectly separate relevant and irrelevant chunks compared to larger commercial embedding models.

### Latency vs. Accuracy (Adaptive RAG)
Every safety mechanism (grading, critique, retry) adds LLM calls and latency. A full Agentic pipeline with web fallback and retry can take over 50 seconds. **Tradeoff:** We resolved this by implementing **Adaptive RAG (Fast-Path Routing)**. Confident retrievals (similarity $\ge 0.72$) bypass grading and critique nodes to run in a single retrieve-and-generate cycle, dropping average latency by 69% to 16.91 seconds.

### Grader Pronoun Leniency
In the original pipeline, the LLM document grader rejected correct PDF pages (like Page 71 of Tesla's annual report) because they referred to the company using pronouns ("we", "our", "the company") rather than the exact search keyword ("Tesla"). **Solution:** We injected document context into the LLM prompts so the grading and critique nodes resolve these pronouns correctly.

---

## Future Improvements

- **Conversational Memory** — Add chat history to the LangGraph state so the agent can handle follow-up questions and multi-turn conversations.
- **Multi-Hop Reasoning** — Decompose complex queries into sub-questions, retrieve context for each, and synthesize a combined answer.
- **Cross-Encoder Reranking** — Replace BM25 with a learned cross-encoder model (e.g., `ms-marco-MiniLM`) for more accurate reranking.
- **Multi-Query Decomposition** — Generate multiple reformulations of the user's query and retrieve from each, merging the results for better recall.
- **Production Deployment** — Containerize with Docker, deploy on cloud (AWS/GCP), and swap to commercial embeddings (OpenAI `text-embedding-3-large`) and a larger LLM (`GPT-4o`, `Claude 3.5 Sonnet`) for enterprise-grade accuracy.
- **Authentication & Multi-Tenancy** — Add user authentication so each user has their own isolated document namespace.
- **CI/CD Evaluation Gate** — Integrate the evaluation harness into a CI pipeline that runs on every PR and blocks merges if retrieval metrics regress.
