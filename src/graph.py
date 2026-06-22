import os
import re
from dotenv import load_dotenv
load_dotenv()

import asyncio
from typing import List, Dict, Any
from typing_extensions import TypedDict

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq

from langgraph.graph import END, StateGraph, START
from src.database import get_local_retriever


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
    web_search_tool = TavilySearch(max_results=5)
except ImportError:
    from langchain_community.tools.tavily_search import TavilySearchResults
    web_search_tool = TavilySearchResults(k=5)


# 2. Define Graph State
class GraphState(TypedDict):
    question: str
    original_question: str
    chat_history: List[Dict[str, Any]]
    generation: str
    search_fallback: str
    documents: List[Document]
    retry_count: int
    critique_feedback: str
    confidence: str
    source: str
    intent: str

# 3. Setup Groq LLM, Retriever, and Tools

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0, timeout=60)

# --- Setup Custom JSON Parsers ---
class GradeDocuments(BaseModel):
    binary_score: str = Field(description="Documents are relevant to the question, 'yes' or 'no'")

parser = JsonOutputParser(pydantic_object=GradeDocuments)

class CritiqueResult(BaseModel):
    binary_score: str = Field(description="PASS ('yes') or FAIL ('no')")
    reason: str = Field(description="Brief reason if FAIL, else empty")

critique_parser = JsonOutputParser(pydantic_object=CritiqueResult)

class CondenseResult(BaseModel):
    intent: str = Field(description="Classify query type: 'chitchat' (greetings, small talk, meta-questions about conversation history, or asking who you are) or 'factual' (asking for facts, policies, statistics, or details from documents)")
    rewritten_question: str = Field(description="The standalone rewritten question if it refers to history context, or the original question word-for-word if it is already standalone")

condense_parser = JsonOutputParser(pydantic_object=CondenseResult)

critique_system_prompt = """Check answer against context:
1. Unsupported claims
2. Missing info
3. Contradictions

Note: Terms in the context like 'we', 'the company', 'our', 'the scheme', or 'the regulations' refer to the target entity of the question (e.g. Tesla, Bupa, or AIESL). Do not fail the answer or context for using pronouns or referring to the entity as 'we' or 'the company' instead of its exact name.

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

def condense_question(state: GraphState) -> Dict[str, Any]:
    print("--- CONDENSING/REWRITING USER QUESTION ---")
    question = state["question"]
    chat_history = state.get("chat_history", [])
    
    # Format the history for the LLM
    history_str = ""
    if chat_history:
        print(f"-> Chat history found ({len(chat_history)} messages). Contextualizing...")
        history_window = chat_history[-10:]
        for msg in history_window:
            role = "User" if msg.get("role") == "user" else "Assistant"
            content = msg.get("content", "")
            sources = msg.get("sources", [])
            sources_str = f" [Sources: {', '.join(sources)}]" if sources else ""
            history_str += f"{role}: {content}{sources_str}\n"
    else:
        history_str = "(No previous conversation)"
        
    condense_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert query analyzer. Analyze the conversation history and the follow-up question.\n"
                   "1. Classify the user's intent as 'chitchat' (if it is a greeting, small talk, meta-question about the chat history, or questions about you) or 'factual' (if it is asking for facts, policies, statistics, or details from documents).\n"
                   "2. If it is 'factual', determine if the question refers to context or pronouns in the history. Rewrite it into a standalone, complete question. If it is already standalone or 'chitchat', keep it as-is.\n\n"
                   "You must respond strictly in JSON matching this schema:\n{format_instructions}"),
        ("human", "Conversation History:\n{history}\n\nFollow-up Question: {question}")
    ]).partial(format_instructions=condense_parser.get_format_instructions())
    
    condenser_chain = condense_prompt | llm | condense_parser
    
    intent = "factual"
    rewritten_question = question
    try:
        res = condenser_chain.invoke({"history": history_str, "question": question})
        intent = res.get("intent", "factual").lower().strip()
        rewritten_question = res.get("rewritten_question", question).strip()
    except Exception as e:
        print(f"-> Condenser/Classifier failed: {e}. Defaulting to factual.")
        
    print(f"-> Original Question: {question}")
    print(f"-> Classified Intent: {intent}")
    print(f"-> Rewritten Standalone Question: {rewritten_question}")
    
    return {
        "question": rewritten_question,
        "original_question": question,
        "intent": intent
    }


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
    
    # Retrieve top 60 candidates
    # Pinecone returns (Document, score) where score is cosine similarity (0-1, higher = more similar)
    docs_with_scores = vectorstore.similarity_search_with_score(question, k=60, filter=filter_dict)
    
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
    
    # Keep top 6 chunks (safety recall margin buffer)
    top_docs_with_scores = reranked_docs[:6]
    
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
    Be lenient: if the document contains any useful facts, partial context, definitions, tables, notes, or background information that could help answer even part of the question, grade it as relevant ('yes'). \n
    Note: The retrieved document comes from the source file '{source_file}'. Terms like 'we', 'the company', 'our', 'the scheme', 'the corporation', or 'the regulations' refer to the entity associated with the source file '{source_file}' (e.g. Tesla for tsla-20251231-gen.pdf, Bupa for MembershipHandbook.pdf, or AIESL for Aiesl Employees service regulation.pdf). Do not penalize documents for using pronouns or referring to the company as 'we' instead of its exact name. \n
    Otherwise, if it is completely off-topic or contains no relevant context at all, grade it as not relevant ('no'). \n
    You must respond strictly in JSON format matching this schema:
    {format_instructions}"""
    
    grade_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "Retrieved document (Source File: {source_file}): \n\n {document} \n\n User question: {question}"),
    ]).partial(format_instructions=parser.get_format_instructions())
    
    doc_grader = grade_prompt | llm | parser
    
    import concurrent.futures
    
    def grade_single_doc(item):
        i, doc = item
        src_file = doc.metadata.get("source", "Unknown PDF Document")
        for attempt in range(4):
            try:
                print(f"-> CRITIC [Chunk {i}]: Grading started (attempt {attempt+1})...")
                res = doc_grader.invoke({
                    "question": question, 
                    "document": doc.page_content,
                    "source_file": src_file
                })
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
    documents = state.get("documents", [])
    critique_feedback = state.get("critique_feedback", "")
    retry_count = state.get("retry_count", 0)
    intent = state.get("intent", "factual")
    
    if intent == "chitchat":
        print("--- GENERATING CHITCHAT/META RESPONSE ---")
        chat_history = state.get("chat_history", [])
        
        # Build history string
        history_str = ""
        for msg in chat_history:
            role = "User" if msg.get("role") == "user" else "Assistant"
            content = msg.get("content", "")
            history_str += f"{role}: {content}\n"
            
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful AI assistant. Answer the user's question directly based on your conversation history. "
                       "If they ask what was discussed, summarize the previous topics and turns. If they greet you, greet them back. "
                       "If they ask who you are, introduce yourself as the RAG Core Agent. "
                       "You do not have external document context for this turn, so speak from the conversation itself. "
                       "Be concise, natural, and helpful. Do not mention any JSON formatting or technical pipelines."),
            ("human", "Conversation History:\n{history}\n\nQuestion: {question}")
        ])
        
        chain = prompt | llm
        generation = chain.invoke({"history": history_str, "question": question})
        return {"documents": [], "question": question, "generation": generation.content}
        
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
        # Combine the top 3 results to get a balanced, accurate context
        combined_contents = []
        for idx, r in enumerate(results[:3], 1):
            c = r.get("content", r.get("text", str(r)))
            url = r.get("url", "unknown source")
            combined_contents.append(f"[Web Result #{idx}] Source: {url}\nContent: {c}")
            
        content = "\n\n---\n\n".join(combined_contents)
        
        # Summarize only if the combined content is exceptionally long (> 4000 characters)
        if len(content) > 4000:
            print("--- SUMMARIZING COMBINED WEB RESULTS (>4000 chars) ---")
            summary_prompt = ChatPromptTemplate.from_messages([
                ("system", "You are an expert summarizer. Combine and summarize these web search results concisely, keeping it under 300 words while retaining all key facts, dates, names, and statistics."),
                ("human", "Web content:\n\n{content}")
            ])
            summary_chain = summary_prompt | llm
            try:
                content = summary_chain.invoke({"content": content}).content.strip()
            except Exception as e:
                print(f"Error summarizing web content: {e}. Truncating instead.")
                content = content[:4000] + "\n... [truncated]"
        else:
            print("--- USING RAW COMBINED WEB RESULTS ---")
            
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


def save_history(state: GraphState) -> Dict[str, Any]:
    print("--- SAVING CONVERSATION HISTORY ---")
    original_question = state.get("original_question", state.get("question"))
    generation = state.get("generation", "")
    documents = state.get("documents", [])
    
    # Extract sources from documents
    sources = []
    for doc in documents:
        src = doc.metadata.get("source")
        if src:
            page = doc.metadata.get("page")
            page_str = f" (Page {page + 1})" if page is not None and isinstance(page, int) else ""
            sources.append(f"{src}{page_str}")
            
    # Deduplicate sources list
    sources = list(sorted(set(sources)))
    
    chat_history = list(state.get("chat_history", []))
    chat_history.append({"role": "user", "content": original_question})
    chat_history.append({"role": "assistant", "content": generation, "sources": sources})
    
    # Slice to store last 20 messages (10 turns) locally
    chat_history = chat_history[-20:]
    
    print(f"-> Chat history updated. Total messages: {len(chat_history)}")
    return {"chat_history": chat_history}

# 5. Define Conditional Routing Logic
def decide_post_condense(state: GraphState) -> str:
    if state.get("intent") == "chitchat":
        print("--- DECISION: CHITCHAT/META QUERY DETECTED. Routing directly to Generate! ---")
        return "generate"
    return "retrieve"

def decide_post_retrieve(state: GraphState) -> str:
    documents = state.get("documents", [])
    local_similarities = [doc.metadata.get("score", 0.0) for doc in documents]
    max_similarity = max(local_similarities) if local_similarities else 0.0
    
    if max_similarity >= 0.82:
        print(f"--- DECISION: FAST-PATH DETECTED (Max similarity {max_similarity:.4f} >= 0.82). Routing directly to Generate! ---")
        return "generate"
    else:
        print(f"--- DECISION: STANDARD PATH DETECTED (Max similarity {max_similarity:.4f} < 0.82). Routing to Grade Documents. ---")
        return "grade_documents"

def decide_to_generate(state: GraphState) -> str:
    if state.get("search_fallback") == "yes":
        print("--- DECISION: ROUTE TO WEB SEARCH ---")
        return "transform_query_and_search"
    else:
        print("--- DECISION: ROUTE TO GENERATION ---")
        return "generate"

def decide_post_generate(state: GraphState) -> str:
    documents = state.get("documents", [])
    local_similarities = [doc.metadata.get("score", 0.0) for doc in documents]
    max_similarity = max(local_similarities) if local_similarities else 0.0
    is_web_fallback = any(doc.metadata.get("source") == "web_fallback" for doc in documents)
    
    if state.get("intent") == "chitchat":
        print("--- DECISION: CHITCHAT BYPASS CRITIQUE. Routing directly to Save History! ---")
        return "save_history"
        
    if max_similarity >= 0.82 and not is_web_fallback and state.get("retry_count", 0) == 0:
        print("--- DECISION: FAST-PATH BYPASS CRITIQUE. Routing directly to Save History! ---")
        return "save_history"
    else:
        print("--- DECISION: STANDARD PATH. Routing to Critique Generation. ---")
        return "critique_generation"

def decide_post_critique(state: GraphState) -> str:
    if state.get("critique_feedback"):
        print("--- DECISION: CRITIQUE FAILED, ROUTING TO GENERATE ---")
        return "generate"
    else:
        print("--- DECISION: CRITIQUE PASSED, ROUTING TO SAVE HISTORY ---")
        return "save_history"

# 6. Build the LangGraph Workflow
workflow = StateGraph(GraphState)

workflow.add_node("condense_question", condense_question)
workflow.add_node("retrieve", retrieve)
workflow.add_node("grade_documents", grade_documents)
workflow.add_node("generate", generate)
workflow.add_node("transform_query_and_search", transform_query_and_search)
workflow.add_node("critique_generation", critique_generation)
workflow.add_node("save_history", save_history)

workflow.add_edge(START, "condense_question")
workflow.add_conditional_edges(
    "condense_question",
    decide_post_condense,
    {
        "retrieve": "retrieve",
        "generate": "generate"
    }
)
workflow.add_conditional_edges(
    "retrieve",
    decide_post_retrieve,
    {
        "generate": "generate",
        "grade_documents": "grade_documents"
    }
)
workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "transform_query_and_search": "transform_query_and_search",
        "generate": "generate",
    },
)
workflow.add_edge("transform_query_and_search", "generate")
workflow.add_conditional_edges(
    "generate",
    decide_post_generate,
    {
        "critique_generation": "critique_generation",
        "save_history": "save_history"
    }
)
workflow.add_conditional_edges(
    "critique_generation",
    decide_post_critique,
    {
        "generate": "generate",
        "save_history": "save_history",
    },
)
workflow.add_edge("save_history", END)

from langgraph.checkpoint.memory import MemorySaver
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)
