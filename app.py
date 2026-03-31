import streamlit as st
from retrieval import query_rag
from dotenv import load_dotenv
import os

load_dotenv()

st.set_page_config(page_title="Legal Case Law RAG", layout="wide", page_icon="⚖️")

# Custom CSS for UI styling
st.markdown("""
<style>
    .chat-container { border-radius: 10px; padding: 10px; margin-bottom: 20px; }
    .source-box { background-color: #f1f3f4; padding: 15px; border-radius: 8px; margin-bottom: 15px; font-family: monospace; font-size: 13px; color: #202124; white-space: pre-wrap; word-wrap: break-word;}
    .source-header { font-weight: bold; color: #1a73e8; margin-bottom: 5px; }
    
    @media (prefers-color-scheme: dark) {
        .source-box { background-color: #2b2b2b; color: #e8eaed; }
        .source-header { color: #8ab4f8; }
    }
</style>
""", unsafe_allow_html=True)

st.title("⚖️ Legal RAG Assistant")
st.caption("Ask complex legal queries. Answers are synthesized locally using up to 1GB of case law.")

# Session State Initialization
if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_context" not in st.session_state:
    st.session_state.last_context = None

# Sidebar for Context Visualization
with st.sidebar:
    st.header("📄 Context Visualization")
    st.markdown("Raw PDF snippets retrieved from ChromaDB:")
    context_container = st.empty()

def render_context(context_docs):
    """Helper to render context HTML snippet securely"""
    if context_docs:
        html = ""
        for i, doc in enumerate(context_docs):
            source = doc.metadata.get("source", "Unknown")
            page = doc.metadata.get("page", "Unknown")
            # Replace script tags just in case
            safe_content = doc.page_content.replace("<", "&lt;").replace(">", "&gt;")
            
            html += f'<div class="source-header">Source: {source} | Page: {page}</div>'
            html += f'<div class="source-box">{safe_content}</div>'
        return html
    return "<p style='color: grey;'>No context retrieved yet. Send a query!</p>"

# Render initial sidebar context
context_container.markdown(render_context(st.session_state.last_context), unsafe_allow_html=True)

# Main Chat Interface Rendering
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Enter your legal query (e.g., 'What was the ruling regarding the limitation period?'):"):
    # Display user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Process answer dynamically
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        with st.spinner("Retrieving judicial context and reasoning..."):
            try:
                answer, docs = query_rag(prompt)
                
                # Render answer in chat
                message_placeholder.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
                
                # Update sidebar context container immediately without rerun
                st.session_state.last_context = docs
                context_container.markdown(render_context(docs), unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"Error executing Local LLM or accessing Database: {str(e)}")
