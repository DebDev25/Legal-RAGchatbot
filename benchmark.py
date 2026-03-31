import time
import pandas as pd
from retrieval import get_rag_chain
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

def run_benchmark():
    print("Loading Local RAG Pipeline & Database...")
    # Load the LLM and Prompt from retrieval.py
    _, prompt, llm = get_rag_chain()
    
    # Manually initialize DB to access underlying distance scores
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2", model_kwargs={"device": "cuda"})
    db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
    
    # 5 Sample Questions tuned for Indian Constitutional / Penal Law
    questions = [
        "What is the punishment for murder under the penal code?",
        "What are the exceptions to culpable homicide amounting to murder?",
        "If a police officer arrests a person without a warrant, what rights do they have?",
        "Can a confession made directly to a police officer be used as definitive evidence in court?",
        "What specifies the penalty for kidnapping a minor for ransom under the penal code?"
    ]
    
    results = []
    
    for i, q in enumerate(questions):
        print(f"\n[{i+1}/5] Processing Query: {q}")
        
        # 1. Retrieval Latency & Relevance Scores Tracking
        start_ret = time.time()
        docs_and_scores = db.similarity_search_with_score(q, k=8)
        ret_latency = time.time() - start_ret
        
        # A lower L2 Distance score indicates mathematically closer relevance.
        avg_distance = sum([score for _, score in docs_and_scores]) / len(docs_and_scores)
        docs = [doc for doc, _ in docs_and_scores]
        
        formatted_context = "\n\n".join(
            f"--- Document: {doc.metadata.get('source', 'Unknown')} | "
            f"Page: {doc.metadata.get('page', 'Unknown')} ---\n{doc.page_content}"
            for doc in docs
        )
        
        # 2. RAG Local Generation Latency Tracking
        chain = prompt | llm
        start_gen = time.time()
        answer_msg = chain.invoke({"context": formatted_context, "question": q})
        gen_latency = time.time() - start_gen
        
        answer_text = answer_msg if isinstance(answer_msg, str) else answer_msg.content
        
        # 3. Log into Metrics Array
        result = {
            "Question": q,
            "Retrieval_Time_sec": round(ret_latency, 3),
            "Generation_Time_sec": round(gen_latency, 2),
            "Avg_L2_Vector_Distance": round(avg_distance, 3),
            "Context_Length_chars": len(formatted_context),
            "Generated_Answer": answer_text.strip()
        }
        results.append(result)
        
        print(f" -> Relevancy Distance: {avg_distance:.3f} | Retrieval Time: {ret_latency:.3f}s | Generation Time: {gen_latency:.2f}s")
        
    # Save the dataframe to a clean CSV for the college report
    df = pd.DataFrame(results)
    df.to_csv("benchmark_metrics.csv", index=False)
    print("\nBenchmark complete! Results automatically saved to 'benchmark_metrics.csv'.")

if __name__ == "__main__":
    run_benchmark()
