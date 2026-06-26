import argparse
import json
import os
import sys
import time
import uuid
import warnings
from typing import Any, Dict, List

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

GOLD_PATH = os.path.join(_PROJECT_ROOT, "data", "eval", "qa_gold.jsonl")
EVAL_DIR = os.path.join(_PROJECT_ROOT, "evaluation")
DEFAULT_MODEL = "llama-3.1-8b-instant"
DEFAULT_SAMPLE = 5


# --------------------------------------------------------------------------- #
# Global Groq rate-limit backoff (mirrors src/evaluation.py)
# --------------------------------------------------------------------------- #
def _install_rate_limit_backoff() -> None:
    from langchain_groq import ChatGroq

    if getattr(ChatGroq, "_ragas_backoff_installed", False):
        return
    original = ChatGroq.invoke

    def wrapped(self, *args, **kwargs):
        for attempt in range(6):
            try:
                return original(self, *args, **kwargs)
            except Exception as exc:  # noqa: BLE001
                msg = str(exc).lower()
                if "429" in msg or "rate_limit" in msg or "limit reached" in msg:
                    sleep_for = (2 ** attempt) + 3.0
                    print(f"⚠️  Groq rate limit hit. Sleeping {sleep_for:.0f}s "
                          f"(retry {attempt + 1}/6)...")
                    time.sleep(sleep_for)
                else:
                    raise
        raise RuntimeError("Groq rate limit: max retries exceeded.")

    ChatGroq.invoke = wrapped
    ChatGroq._ragas_backoff_installed = True


def load_in_domain(n: int) -> List[Dict[str, Any]]:
    rows = []
    with open(GOLD_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                row = json.loads(line)
                if row.get("type") == "in_domain":
                    rows.append(row)
    return rows[:n]


def run_agent(question: str, model: str) -> Dict[str, Any]:
    """Run the production agent and return its answer + the contexts it used."""
    import src.graph
    from langchain_groq import ChatGroq

    # Point the agent at the chosen model (default 8B to spare the TPM budget).
    src.graph.llm = ChatGroq(model=model, temperature=0, timeout=60)
    from src.graph import app

    config = {"configurable": {"thread_id": f"ragas_{uuid.uuid4()}"}}
    state = app.invoke({"question": question, "source": "all"}, config=config)
    docs = state.get("documents", []) or []
    contexts = [d.page_content for d in docs]
    return {"answer": state.get("generation", ""), "contexts": contexts}


def build_dataset(rows: List[Dict[str, Any]], model: str):
    from ragas import EvaluationDataset

    samples = []
    for idx, row in enumerate(rows, start=1):
        q = row["question"]
        print(f"  [{idx}/{len(rows)}] Running agent: {q[:65]}...")
        out = run_agent(q, model)
        if not out["contexts"]:
            # RAGAS context metrics need at least one context; substitute a
            # placeholder so the sample is still scoreable (faithfulness will be
            # low, which is the correct signal for an empty/no-context answer).
            out["contexts"] = ["(no retrieved context)"]
        samples.append({
            "user_input": q,
            "retrieved_contexts": out["contexts"],
            "response": out["answer"] or "(no answer generated)",
            "reference": row.get("ground_truth", "") or "(no reference)",
        })
        time.sleep(1.0)  # gentle throttle between agent runs
    return EvaluationDataset.from_list(samples)


def run(n: int = DEFAULT_SAMPLE, model: str = DEFAULT_MODEL) -> Dict[str, Any]:
    print("=" * 64)
    print("🧪 RAGAS ANSWER-QUALITY EVALUATION  (uses Groq LLM judge)")
    print("=" * 64)

    _install_rate_limit_backoff()

    from ragas import evaluate
    from ragas.metrics import (
        Faithfulness,
        ResponseRelevancy,
        LLMContextPrecisionWithReference,
        LLMContextRecall,
    )
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from langchain_groq import ChatGroq
    from langchain_huggingface import HuggingFaceEmbeddings

    rows = load_in_domain(n)
    print(f"📄 Evaluating {len(rows)} in-domain questions with model '{model}'.\n")

    print("⚙️  Generating answers with the agent (this calls Groq)...")
    dataset = build_dataset(rows, model)

    evaluator_llm = LangchainLLMWrapper(ChatGroq(model=model, temperature=0, timeout=60))
    evaluator_emb = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    )

    metrics = [
        Faithfulness(),
        ResponseRelevancy(),
        LLMContextPrecisionWithReference(),
        LLMContextRecall(),
    ]

    print("\n⚙️  Scoring with RAGAS (this also calls Groq)...")
    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=evaluator_llm,
        embeddings=evaluator_emb,
    )

    scores = {k: (float(v) if v is not None else None) for k, v in result._repr_dict.items()} \
        if hasattr(result, "_repr_dict") else dict(result)

    print("\n" + "=" * 64)
    print("📊 RAGAS RESULTS")
    print("=" * 64)
    for k, v in scores.items():
        print(f"  {k:<32}: {v:.3f}" if isinstance(v, (int, float)) else f"  {k}: {v}")

    os.makedirs(EVAL_DIR, exist_ok=True)
    out_path = os.path.join(EVAL_DIR, "ragas_results.json")
    payload = {"model": model, "n_questions": len(rows), "scores": scores}
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\n💾 Saved: {out_path}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="RAGAS answer-quality evaluation (uses Groq).")
    parser.add_argument("--n", type=int, default=DEFAULT_SAMPLE, help="Number of in-domain questions (default 5).")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Groq model for agent + judge.")
    args = parser.parse_args()
    run(n=args.n, model=args.model)


if __name__ == "__main__":
    main()
