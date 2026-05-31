import os
import asyncio
from typing import List, Dict, Any
from typing_extensions import TypedDict
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq

from langgraph.graph import END, StateGraph, START
from src.database import get_local_retriever

load_dotenv()

# 1. Environment Configuration
# Use .get() to avoid setting env vars to the string "None" if keys are missing
_groq_key = os.getenv("GROQ_API_KEY")
_tavily_key = os.getenv("TAVILY")

if not _groq_key:
    raise EnvironmentError("GROQ_API_KEY is not set in .env")
if not _tavily_key:
    raise EnvironmentError("TAVILY is not set in .env")

os.environ["GROQ_API_KEY"] = _groq_key
os.environ["TAVILY_API_KEY"] = _tavily_key

# Import Tavily AFTER setting the env var so it picks up the key
try:
    from langchain_tavily import TavilySearch
    web_search_tool = TavilySearch(max_results=3)
except ImportError:
    from langchain_community.tools.tavily_search import TavilySearchResults
    web_search_tool = TavilySearchResults(k=3)


# 2. Define Graph State
class GraphState(TypedDict):
    question: str
    generation: str
    search_fallback: str
    documents: List[Document]
    retry_count: int
    critique_feedback: str
    confidence: str
    source: str

# 3. Setup Groq LLM, Retriever, and Tools

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

# --- Setup Custom JSON Parsers ---
class GradeDocuments(BaseModel):
    binary_score: str = Field(description="Documents are relevant to the question, 'yes' or 'no'")

parser = JsonOutputParser(pydantic_object=GradeDocuments)

class CritiqueResult(BaseModel):
    binary_score: str = Field(description="PASS ('yes') or FAIL ('no')")
    reason: str = Field(description="Brief reason if FAIL, else empty")

critique_parser = JsonOutputParser(pydantic_object=CritiqueResult)

critique_system_prompt = """Check answer against context:
1. Unsupported claims
2. Missing info
3. Contradictions

JSON format:
{{"binary_score": "yes" (PASS) or "no" (FAIL), "reason": "reason if FAIL"}}"""

# Heuristic Confidence Scoring Function
def compute_confidence(state: Dict[str, Any]) -> str:
    documents = state.get("documents", [])
    retry_count = state.get("retry_count", 0)
    search_fallback = state.get("search_fallback", "no")
    critique_feedback = state.get("critique_feedback", "")
    
    local_scores = [doc.metadata.get("score", 0.0) for doc in documents]
    max_similarity = max(local_scores) if local_scores else 0.0
    num_sources = len(documents)
    
    if critique_feedback:
        return "Low"
        
    if search_fallback == "yes":
        return "Medium"
        
    # Recalibrated similarity thresholds for sentence-transformers score range
    if max_similarity < 0.38 or num_sources == 0:
        return "Low"
        
    if retry_count == 1:
        return "Medium"
        
    if retry_count == 0 and search_fallback == "no" and max_similarity > 0.40 and num_sources >= 1:
        return "High"
        
    return "Medium"

# 4. Define Nodes (Graph Actions)

def retrieve(state: GraphState) -> Dict[str, Any]:
    print("--- RETRIEVING DOCUMENTS (k=20) ---")
    question = state["question"]
    source_filter = state.get("source")
    
    from src.database import get_vectorstore
        
    vectorstore = get_vectorstore()
    
    # Setup document metadata filtering
    filter_dict = None
    if source_filter and source_filter != "all":
        filter_dict = {"source": os.path.basename(source_filter)}
        print(f"-> Filtering retrieval by document source: {filter_dict['source']}")
    
    # Retrieve top 20 candidates
    # Pinecone returns (Document, score) where score is cosine similarity (0-1, higher = more similar)
    docs_with_scores = vectorstore.similarity_search_with_score(question, k=20, filter=filter_dict)
    
    # Normalize path and deduplicate based on content
    seen_contents = set()
    unique_docs_with_scores = []
    for doc, score in docs_with_scores:
        if "source" in doc.metadata:
            doc.metadata["source"] = os.path.basename(doc.metadata["source"])
            
        cleaned = " ".join(doc.page_content.split())
        if cleaned not in seen_contents:
            seen_contents.add(cleaned)
            # Pinecone cosine similarity is already 0-1 (higher = more similar)
            similarity = max(0.0, min(1.0, score))
            doc.metadata["score"] = similarity
            unique_docs_with_scores.append((doc, similarity))
            
    # Apply lexical keyword boosting (hybrid reranking)
    import re
    stop_words = {"what", "are", "all", "the", "different", "scenarios", "where", "an", "employee", "service", "can", "be", "regularized", "or", "impacted", "by", "leave", "without", "pay", "is", "a", "of", "in", "to", "for", "on", "with", "at", "by", "from", "it", "this", "that"}
    words = re.findall(r'\b\w+\b', question.lower())
    keywords = [w for w in words if len(w) > 2 and w not in stop_words]
    
    reranked_docs = []
    for doc, similarity in unique_docs_with_scores:
        content_lower = doc.page_content.lower()
        matches = sum(1 for kw in keywords if kw in content_lower)
        lexical_score = min(1.0, matches / max(1, len(keywords)))
        
        # Combined score: 60% semantic + 40% lexical
        combined_score = 0.6 * similarity + 0.4 * lexical_score
        doc.metadata["combined_score"] = combined_score
        reranked_docs.append((doc, combined_score))
        
    reranked_docs.sort(key=lambda x: x[1], reverse=True)
    
    # Keep top 4 chunks (drops LLM calls from 16 to 4!)
    top_docs_with_scores = reranked_docs[:4]
    
    # Compress content: clean up whitespace
    final_docs = []
    for doc, _ in top_docs_with_scores:
        content = doc.page_content
        content = re.sub(r'\n{3,}', '\n\n', content)
        content = re.sub(r' {2,}', ' ', content)
        doc.page_content = content.strip()
        final_docs.append(doc)
        
    print(f"-> Retained {len(final_docs)} reranked compressed unique chunks.")
    return {
        "documents": final_docs,
        "question": question,
        "retry_count": 0,
        "critique_feedback": "",
        "confidence": "Low",
        "source": source_filter
    }


def grade_documents(state: GraphState) -> Dict[str, Any]:
    print("--- GRADING RETRIEVED DOCUMENTS ---")
    question = state["question"]
    documents = state["documents"]
    
    # Heuristic Similarity Threshold check
    local_similarities = [doc.metadata.get("score", 0.0) for doc in documents]
    max_similarity = max(local_similarities) if local_similarities else 0.0
    print(f"-> Maximum local chunk similarity: {max_similarity:.4f}")
    
    if max_similarity < 0.40:
        print(f"-> Maximum similarity {max_similarity:.4f} is below threshold 0.40. Triggering Web Fallback!")
        return {"documents": [], "question": question, "search_fallback": "yes"}
    
    system_prompt = """You are a critic grading relevance of a retrieved document to a user question. \n
    Analyze the document content and determine if it has any relevance to answering the user question. \n
    Be somewhat lenient: if the document contains any useful facts, partial context, definitions, tables, notes, or background information that could help answer even part of the question, grade it as relevant ('yes'). \n
    Otherwise, if it is completely off-topic or contains no relevant context at all, grade it as not relevant ('no'). \n
    You must respond strictly in JSON format matching this schema:
    {format_instructions}"""
    
    grade_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
    ]).partial(format_instructions=parser.get_format_instructions())
    
    doc_grader = grade_prompt | llm | parser
    
    import concurrent.futures
    
    def grade_single_doc(item):
        i, doc = item
        for attempt in range(4):
            try:
                print(f"-> CRITIC [Chunk {i}]: Grading started (attempt {attempt+1})...")
                res = doc_grader.invoke({"question": question, "document": doc.page_content})
                return res
            except Exception as e:
                import time
                sleep_duration = (2 ** attempt) + 1.0
                print(f"-> CRITIC [Chunk {i}]: Error during grading (attempt {attempt+1}): {e}. Retrying in {sleep_duration}s...")
                time.sleep(sleep_duration)
        return Exception("Max retries exceeded due to persistent errors/rate limits.")

    print(f"-> CRITIC: Spawning parallel grading for {len(documents)} candidate documents...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(grade_single_doc, enumerate(documents)))
            
    filtered_docs = []
    for doc, res in zip(documents, results):
        if isinstance(res, Exception):
            print(f"-> CRITIC: PARSE ERROR ({res}), SKIPPING CHUNK")
            continue
        
        page_num = doc.metadata.get("page", 0) + 1
        if res.get("binary_score") == "yes":
            print(f"-> CRITIC: DOCUMENT RELEVANT (Page {page_num})")
            filtered_docs.append(doc)
        else:
            print(f"-> CRITIC: DOCUMENT NOT RELEVANT (Page {page_num})")
            
    search_fallback = "no" if filtered_docs else "yes"
    return {"documents": filtered_docs, "question": question, "search_fallback": search_fallback}

def generate(state: GraphState) -> Dict[str, Any]:
    print("--- GENERATING ANSWER WITH GROQ ---")
    question = state["question"]
    documents = state["documents"]
    critique_feedback = state.get("critique_feedback", "")
    retry_count = state.get("retry_count", 0)
    
    if not documents:
        return {"documents": [], "question": question, "generation": "I could not find relevant information."}
    
    context = "\n\n".join([doc.page_content for doc in documents])
    
    if critique_feedback:
        print(f"--- REGENERATING ANSWER (Attempt {retry_count}) ---")
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert assistant. Use the following pieces of retrieved context to answer the question.\n\nContext:\n{context}\n\n"
                       "IMPORTANT: Your previous attempt failed the critique. Please fix this in your new answer:\n{feedback}"),
            ("human", "Question: {question}")
        ])
    else:
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert assistant. Use the following pieces of retrieved context to answer the question. If you don't know the answer, say that you don't know.\n\nContext:\n{context}"),
            ("human", "Question: {question}")
        ])
    
    rag_chain = prompt | llm
    generation = rag_chain.invoke({"context": context, "question": question, "feedback": critique_feedback})
    
    return {"documents": documents, "question": question, "generation": generation.content}

def transform_query_and_search(state: GraphState) -> Dict[str, Any]:
    print("--- TRANSFORMING QUERY & EXECUTING WEB SEARCH ---")
    question = state["question"]
    
    rewrite_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a query rewriter. Convert the input query into an optimized version for web search engines. Respond ONLY with the revised string query, nothing else."),
        ("human", "Initial query: {question}")
    ])
    query_rewriter = rewrite_prompt | llm
    better_question = query_rewriter.invoke({"question": question}).content
    print(f"-> Optimized Query: {better_question}")
    
    search_results = web_search_tool.invoke({"query": better_question})
    
    # Handle search results - support list or dict formats
    results = []
    if isinstance(search_results, list):
        results = search_results
    elif isinstance(search_results, dict) and "results" in search_results:
        results = search_results["results"]
        
    if results and len(results) > 0:
        top_result = results[0]
        content = top_result.get("content", top_result.get("text", str(top_result)))
        
        # Summarize ONLY if length is greater than 1500 characters
        if len(content) > 1500:
            print("--- SUMMARIZING TOP WEB RESULT (>1500 chars) ---")
            summary_prompt = ChatPromptTemplate.from_messages([
                ("system", "You are an expert summarizer. Summarize this web search result concisely, keeping it under 200 words."),
                ("human", "Web content:\n\n{content}")
            ])
            summary_chain = summary_prompt | llm
            try:
                content = summary_chain.invoke({"content": content}).content.strip()
            except Exception as e:
                print(f"Error summarizing web content: {e}. Truncating instead.")
                content = content[:1500] + "\n... [truncated]"
        else:
            print("--- USING RAW TOP WEB RESULT ---")
            
        web_results = content
    else:
        web_results = "No search results found."
        
    new_doc = Document(page_content=web_results, metadata={"source": "web_fallback", "score": 0.85})
    return {"documents": [new_doc], "question": question}

def critique_generation(state: GraphState) -> Dict[str, Any]:
    print("--- CRITIQUING GENERATION ---")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]
    retry_count = state.get("retry_count", 0)
    
    if not documents:
        return {"critique_feedback": "", "retry_count": retry_count, "confidence": "Low"}
        
    context = "\n\n".join([doc.page_content for doc in documents])
    
    critique_prompt = ChatPromptTemplate.from_messages([
        ("system", critique_system_prompt),
        ("human", "Context:\n{context}\n\nAnswer:\n{generation}"),
    ]).partial(format_instructions=critique_parser.get_format_instructions())
    
    critique_grader = critique_prompt | llm | critique_parser
    
    score = "yes"
    reason = ""
    try:
        print("-> CRITIC: Evaluating generation...")
        res = critique_grader.invoke({"context": context, "generation": generation})
        score = res.get("binary_score", "yes").lower().strip()
        reason = res.get("reason", "")
        print(f"-> CRITIC: Score={score}, Reason={reason}")
    except Exception as e:
        print(f"-> CRITIC: Error during critique: {e}. Defaulting to PASS.")
        
    if score == "no" and retry_count < 1:
        print("-> CRITIC: Generation failed critique. Requesting regenerate once...")
        return {"critique_feedback": reason, "retry_count": retry_count + 1}
    else:
        # Compute confidence score
        state_temp = {
            "documents": documents,
            "retry_count": retry_count,
            "search_fallback": state.get("search_fallback", "no"),
            "critique_feedback": "" if score == "yes" else reason
        }
        confidence = compute_confidence(state_temp)
        return {"critique_feedback": "", "retry_count": retry_count, "confidence": confidence}

# 5. Define Conditional Routing Logic
def decide_to_generate(state: GraphState) -> str:
    if state.get("search_fallback") == "yes":
        print("--- DECISION: ROUTE TO WEB SEARCH ---")
        return "transform_query_and_search"
    else:
        print("--- DECISION: ROUTE TO GENERATION ---")
        return "generate"

def decide_post_critique(state: GraphState) -> str:
    if state.get("critique_feedback"):
        print("--- DECISION: CRITIQUE FAILED, ROUTING TO GENERATE ---")
        return "generate"
    else:
        print("--- DECISION: CRITIQUE PASSED, ROUTING TO END ---")
        return END

# 6. Build the LangGraph Workflow
workflow = StateGraph(GraphState)

workflow.add_node("retrieve", retrieve)
workflow.add_node("grade_documents", grade_documents)
workflow.add_node("generate", generate)
workflow.add_node("transform_query_and_search", transform_query_and_search)
workflow.add_node("critique_generation", critique_generation)

workflow.add_edge(START, "retrieve")
workflow.add_edge("retrieve", "grade_documents")
workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "transform_query_and_search": "transform_query_and_search",
        "generate": "generate",
    },
)
workflow.add_edge("transform_query_and_search", "generate")
workflow.add_edge("generate", "critique_generation")
workflow.add_conditional_edges(
    "critique_generation",
    decide_post_critique,
    {
        "generate": "generate",
        END: END,
    },
)

app = workflow.compile()
