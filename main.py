import os
import sys

# Fix Windows console encoding for emoji/unicode characters
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from dotenv import load_dotenv
from src.database import ingest_pdf_directory, DB_DIR

# Load api keys from the local .env file
load_dotenv()

def run_agentic_rag():
    print(" Step 1: Checking/Ingesting Local PDF Documents...")
    # Check if vectors already exist, if not index them
    if not os.path.exists(DB_DIR):
        print(" Local Vector Database empty. Ingesting documents...")
        total_chunks = ingest_pdf_directory()
        if total_chunks == 0:
            print(" Stop: Add a PDF to the 'data/' directory before running!")
            return
    else:
        print("✅ Persistent ChromaDB database found on disk. Skipping ingestion.")

    print("\n Step 2: Starting LangGraph RAG Agent System...")

    from src.graph import app

    print("\n" + "="*50)
    print("Welcome to the Agentic RAG Chatbot!")
    print("Type 'bye', 'good bye', 'exit', or 'quit' to exit.")
    print("="*50 + "\n")

    while True:
        user_input = input("User: ").strip()
        
        # Check for exit conditions
        if user_input.lower() in ["bye", "good bye", "goodbye", "exit", "quit", "q"]:
            print("Chatbot: Goodbye! Have a great day!")
            break
            
        if not user_input:
            continue

        print(f"\n--- Processing Query: '{user_input}' ---")
        inputs = {"question": user_input}
        final_state = None

        # Stream the states visually to console output
        for output in app.stream(inputs):
            for node_name, node_state in output.items():
                final_state = node_state

        # The final state from the last node contains the 'generation' key
        if final_state and "generation" in final_state:
            print(f"\n Final Answer:\n{final_state['generation']}\n")
        else:
            print("\n No generation produced. Check your documents and API keys.\n")

if __name__ == "__main__":
    run_agentic_rag()
