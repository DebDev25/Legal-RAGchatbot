import os
import fitz  # PyMuPDF
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

DATA_DIRS = {
    "./data/indian_supreme_court_pdfs": "case_law",
    "./data/penal_code": "penal_code"
}
CHROMA_PATH = "./chroma_db"

def extract_text_from_pdf(pdf_path):
    """
    Extracts text from a PDF block by block to preserve order, returning a list of dicts.
    Each dict contains page number and text.
    """
    doc = fitz.open(pdf_path)
    pages_data = []
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        
        # Extract blocks of text. Using "blocks" manages multi-column/complex layouts better.
        blocks = page.get_text("blocks")
        # Blocks are tuples: (x0, y0, x1, y1, "lines in block", block_no, block_type)
        if not blocks:
            continue
            
        # Sort vertically then horizontally
        blocks.sort(key=lambda b: (b[1], b[0])) 
        
        text_content = "\n".join([b[4] for b in blocks if len(b) >= 7 and b[6] == 0]) # 0 means text block
        
        if text_content.strip():
             pages_data.append({"page": page_num + 1, "text": text_content.strip()})
             
    return pages_data

def chunk_pdf_data(pages_data, filename):
    """
    Splits text into ~800 token chunks with ~150 token overlap.
    """
    # Using from_tiktoken_encoder to split exactly by tokens approximation
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=800,
        chunk_overlap=150
    )
    
    chunks = []
    for data in pages_data:
        # Split text of the current page
        page_chunks = text_splitter.split_text(data["text"])
        for chunk in page_chunks:
            # Add chunk with metadata
            chunks.append({
                "text": chunk,
                "metadata": {
                    "source": filename,
                    "page": data["page"],
                    "case_title": filename.replace(".pdf", "").replace("_", " ").title()
                }
            })
    return chunks

def ingest_pdfs():
    """
    Iterates over PDF directory, extracts, chunks, and stores into ChromaDB.
    """
    all_chunks = []
    
    for current_dir, source_type in DATA_DIRS.items():
        if not os.path.exists(current_dir):
            print(f"Directory {current_dir} does not exist. Skipping.")
            continue
            
        pdf_files = [f for f in os.listdir(current_dir) if f.lower().endswith(".pdf")]
        if not pdf_files:
            print(f"No PDFs found in {current_dir}.")
            continue
            
        for pdf_file in pdf_files:
            print(f"Processing {pdf_file} [{source_type}]...")
            pdf_path = os.path.join(current_dir, pdf_file)
            pages_data = extract_text_from_pdf(pdf_path)
            chunks = chunk_pdf_data(pages_data, pdf_file)
            
            # Tag the source type explicitly into metadata
            for c in chunks:
                c["metadata"]["source_type"] = source_type
                
            all_chunks.extend(chunks)
            
    print(f"Total chunks generated: {len(all_chunks)}")
    if len(all_chunks) == 0:
        print("No text could be extracted.")
        return
        
    # Initialize local HuggingFace Embeddings
    print("Loading embedding model (all-MiniLM-L6-v2)...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Store in ChromaDB
    print("Storing chunks in local ChromaDB...")
    texts = [c["text"] for c in all_chunks]
    metadatas = [c["metadata"] for c in all_chunks]
    
    # Store them in vector store
    vectorstore = Chroma.from_texts(
        texts=texts,
        embedding=embeddings,
        metadatas=metadatas,
        persist_directory=CHROMA_PATH
    )
    print("Ingestion complete! Run the RAG retriever next.")

if __name__ == "__main__":
    ingest_pdfs()
