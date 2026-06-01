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

## Normal RAG vs. Agentic RAG

A **standard RAG** pipeline is linear and fragile — it retrieves chunks, feeds them to an LLM, and blindly returns whatever the LLM produces. There is no verification, no fallback, and no self-correction. If the retriever pulls the wrong paragraphs, the LLM hallucinates confidently with zero safety net.

**Agentic RAG** changes everything. Instead of a dumb pipeline, the system acts as an **autonomous agent** with decision-making loops:
- It **grades** its own retrieved documents for relevance before using them.
- If the local documents fail, it **falls back** to real-time web search instead of hallucinating.
- After generating an answer, a **self-critique loop** evaluates the response for factual accuracy.
- If the critique fails, the agent **regenerates** the answer with corrective feedback — up to 1 retry.

This creates a self-correcting, multi-path reasoning system that is significantly more reliable than traditional RAG.

## Why Hybrid RAG Matters

This project implements a **hybrid reranking** strategy that combines two retrieval signals:
- **Semantic similarity** (dense vector cosine distance) captures conceptual meaning.
- **Lexical keyword matching** (exact term overlap) catches precise names, numbers, and domain terms that embeddings might miss.

By blending both signals (`60% semantic + 40% lexical`), the system retrieves far more accurate chunks than either method alone, especially for technical or financial documents where exact terminology matters.

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
                    │  RETRIEVE (k=20 → top 4)│
                    │  Hybrid Semantic+Lexical │
                    │  Reranking & Dedup       │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │   GRADE DOCUMENTS       │
                    │   LLM relevance check   │
                    │   (parallel, 4 chunks)  │
                    └────────────┬────────────┘
                                 │
                    ┌────────────┴────────────┐
                    │    Conditional Edge     │
                    │  Any chunk relevant?    │
                    ▼                         ▼
         ┌──────────────────┐     ┌──────────────────────┐
         │     GENERATE     │     │  TRANSFORM QUERY &   │
         │  Answer from PDF │     │  WEB SEARCH (Tavily) │
         └────────┬─────────┘     └──────────┬───────────┘
                  │                          │
                  │         ┌────────────────┘
                  │         │
                  ▼         ▼
         ┌──────────────────────┐
         │    GENERATE ANSWER   │
         │   (from web or PDF)  │
         └──────────┬───────────┘
                    │
                    ▼
         ┌──────────────────────┐
         │  CRITIQUE GENERATION │◄──────┐
         │  Hallucination check │       │
         └──────────┬───────────┘       │
                    │                   │
           ┌────────┴────────┐          │
           │ Conditional Edge│          │
           │  Critique Pass? │          │
           ▼                 ▼          │
     ┌───────────┐   ┌──────────────┐   │
     │  ✅ END   │   │ REGENERATE   │───┘
     │  Stream   │   │ REGENERATE   │
     │  Answer   │   └──────────────┘
     └───────────┘
```

### How It Works (Step by Step)

1. **Retrieve** — The user's question is embedded and searched against Pinecone. The top 20 candidates are retrieved, deduplicated, and reranked using a hybrid semantic + lexical score. Only the **top 4 chunks** survive.

2. **Grade Documents** — Each of the 4 chunks is sent to the LLM in **parallel** with the question: *"Is this chunk actually relevant?"*. The LLM grades each one `yes` or `no`. If all chunks are graded as irrelevant, the system pivots to web search.

3. **Web Search Fallback** — If the PDF doesn't have the answer, the query is first **rewritten** by the LLM into a search-engine-optimized form, then executed against the **Tavily Search API**. Long results are summarized before being passed to generation.

4. **Generate** — The LLM generates a comprehensive answer using the relevant PDF chunks (or web results) as context.

5. **Self-Critique** — A separate LLM call evaluates the generated answer against the source context, checking for unsupported claims, missing information, or contradictions.

6. **Retry or Finish** — If the critique fails, the system loops back to generation with corrective feedback. After passing critique (or exhausting retries), the final answer is streamed to the user.

---

## ✨ Features

| Feature | Description |
|---|---|
| **LangGraph Workflow Orchestration** | Stateful, cyclical agent graph with conditional routing and retry loops |
| **Pinecone Cloud Vector Database** | Production-ready, cloud-hosted vector storage with metadata filtering for multi-document support |
| **Hybrid Semantic + Lexical Reranking** | Combined scoring (`0.6×semantic + 0.4×lexical`) for precision retrieval |
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
│   ├── database.py        # PDF ingestion, chunking, embedding, and Pinecone connector
│   └── graph.py           # LangGraph state machine — retrieve, grade, search, generate, critique
│
├── data/                  # Drop your PDF files here for ingestion (git-ignored)
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

### Latency vs. Accuracy
Every safety mechanism (grading, critique, retry) adds LLM calls and latency. A full pipeline with web fallback and one retry can take 15–20 seconds. **Tradeoff:** We prioritize answer quality and reliability over raw speed, which is acceptable for document Q&A use cases.

---

## Future Improvements

- **Conversational Memory** — Add chat history to the LangGraph state so the agent can handle follow-up questions and multi-turn conversations.
- **Multi-Hop Reasoning** — Decompose complex queries into sub-questions, retrieve context for each, and synthesize a combined answer.
- **Cross-Encoder Reranking** — Replace the lexical heuristic with a learned cross-encoder model (e.g., `ms-marco-MiniLM`) for more accurate reranking.
- **Multi-Query Decomposition** — Generate multiple reformulations of the user's query and retrieve from each, merging the results for better recall.
- **Production Deployment** — Containerize with Docker, deploy on cloud (AWS/GCP), and swap to commercial embeddings (OpenAI `text-embedding-3-large`) and a larger LLM (`GPT-4o`, `Claude 3.5 Sonnet`) for enterprise-grade accuracy.
- **Authentication & Multi-Tenancy** — Add user authentication so each user has their own isolated document namespace.
- **Evaluation Dashboard** — Build an automated evaluation harness that runs test queries on every code change and tracks retrieval precision, hallucination rate, and latency metrics over time.
