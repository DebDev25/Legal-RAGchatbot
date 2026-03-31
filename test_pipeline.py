import os
from retrieval import query_rag
from dotenv import load_dotenv

load_dotenv()

def test_pipeline():
    print("Testing end-to-end Local RAG Pipeline...")
    # Example complex legal question that needs retrieval
    query = "What did the court rule regarding the admissibility of the evidence?"
    
    print(f"\nUser Query: {query}")
    print("Retrieving chunks and generating answer...\n")
    
    try:
        answer, docs = query_rag(query)
        
        print("\n------- GENERATED ANSWER -------")
        print(answer)
        print("--------------------------------\n")
        
        print(f"Number of chunks retrieved: {len(docs)}")
        for i, doc in enumerate(docs):
            print(f"\n--- Chunk {i+1} ---")
            print(f"Source: {doc.metadata.get('source')} | Page: {doc.metadata.get('page')}")
            # Snippet of retrieved text
            print(f"Text Snippet: {doc.page_content[:200]}...")
            
    except Exception as e:
         print(f"Error during query execution: {e}")

if __name__ == "__main__":
    test_pipeline()
