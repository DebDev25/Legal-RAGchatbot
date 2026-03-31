from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import os

load_dotenv()

CHROMA_PATH = "./chroma_db"

def get_retriever():
    """
    Initializes ChromaDB retriever for top-5 chunks.
    """
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2", 
        model_kwargs={"device": "cuda"}
    )
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
    retriever = db.as_retriever(search_kwargs={"k": 8})
    return retriever

def get_rag_chain():
    """
    Builds the RAG Prompt and LLM generation pipeline.
    """
    # Strict Prompt Design
    system_prompt = (
        "<|im_start|>system\n"
        "You are an objective legal reporter. Read the Document Context.\n"
        "Rules:\n"
        "1. Start your answer with 'Yes.' or 'No.' ONLY IF the context explicitly states the legality.\n"
        "2. If asked about a crime, list the punishment from the 'penal_code' context, and cite up to 3 'case_law' context examples.\n"
        "3. ALWAYS cite the Document name.\n"
        "4. **CRITICAL**: If the context does not explicitly contain the answer, say EXACTLY 'I don't know.' DO NOT GUESS OR INVENT REASONS.\n"
        "<|im_end|>\n"
        "<|im_start|>user\n"
        "Document Context:\n{context}\n\n"
        "User Query: {question}\n<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    
    prompt = PromptTemplate.from_template(system_prompt)
    
    # Initialize small local LLM via HuggingFace
    from transformers import pipeline
    
    print("Loading local Qwen 1.5B model (this may take a few moments)...")
    pipe = pipeline(
        "text-generation",
        model="Qwen/Qwen2.5-1.5B-Instruct",
        max_new_tokens=350,
        return_full_text=False,
        do_sample=False,
        device_map="auto"
    )
    llm = HuggingFacePipeline(pipeline=pipe)
    
    return get_retriever(), prompt, llm

def query_rag(question: str):
    """
    End-to-end query method that returns both the generated answer and the source chunks.
    """
    retriever, prompt, llm = get_rag_chain()
    
    # Retrieve top-5 most relevant chunks
    docs = retriever.invoke(question)
    
    # Format context chunks with metadata included directly into the text for the LLM
    formatted_context = "\n\n".join(
        f"--- Document: {doc.metadata.get('source', 'Unknown')} | "
        f"Page: {doc.metadata.get('page', 'Unknown')} ---\n{doc.page_content}"
        for doc in docs
    )
    
    # Generate answer with explicitly passed context
    chain = prompt | llm
    answer_msg = chain.invoke({
        "context": formatted_context,
        "question": question
    })
    
    # Extract just the raw text output 
    answer_text = answer_msg if isinstance(answer_msg, str) else answer_msg.content
    
    return answer_text, docs
