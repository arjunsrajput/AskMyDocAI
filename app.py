import os
import shutil
import tempfile
import streamlit as st
from src.document_loader import MultiFormatDocumentLoader
from src.text_splitter import DocumentSplitter
from src.vector_store import VectorStoreManager
from src.rag_engine import RAGEngine

# Set Streamlit Page Config
st.set_page_config(page_title="AskMyDoc AI", page_icon="📄", layout="wide")

# Create a unique, private database folder for THIS specific user session
if "session_dir" not in st.session_state:
    st.session_state.session_dir = tempfile.mkdtemp(prefix="askmydoc_")
    st.session_state.upload_dir = os.path.join(st.session_state.session_dir, "uploads")
    st.session_state.chroma_dir = os.path.join(st.session_state.session_dir, "chroma_db")
    os.makedirs(st.session_state.upload_dir, exist_ok=True)
    os.makedirs(st.session_state.chroma_dir, exist_ok=True)

# Initialize Session-Isolated Services
if "vector_manager" not in st.session_state:
    st.session_state.vector_manager = VectorStoreManager(persist_dir=st.session_state.chroma_dir)
if "rag_engine" not in st.session_state:
    st.session_state.rag_engine = RAGEngine(st.session_state.vector_manager)
if "splitter" not in st.session_state:
    st.session_state.splitter = DocumentSplitter()
if "messages" not in st.session_state:
    st.session_state.messages = []
if "processed_files" not in st.session_state:
    st.session_state.processed_files = []

# --- SIDEBAR: Document Management ---
with st.sidebar:
    st.title("📄 AskMyDoc Control")
    st.caption("Multi-Document RAG with Page Citations")

    # File Upload Widget
    uploaded_files = st.file_uploader(
        "Upload Documents",
        type=["pdf", "docx", "txt", "md"],
        accept_multiple_files=True
    )

    if st.button("📤 Process & Index Documents", type="primary"):
        if uploaded_files:
            with st.spinner("Processing documents into vector store..."):
                all_chunks = []
                for file in uploaded_files:
                    file_path = os.path.join(st.session_state.upload_dir, file.name)
                    with open(file_path, "wb") as f:
                        f.write(file.getbuffer())

                    # Parse and Chunk
                    docs = MultiFormatDocumentLoader.load_file(file_path)
                    chunks = st.session_state.splitter.split_documents(docs)
                    all_chunks.extend(chunks)
                    if file.name not in st.session_state.processed_files:
                        st.session_state.processed_files.append(file.name)

                # Store Embeddings in ChromaDB
                st.session_state.vector_manager.add_documents(all_chunks)
                st.success(f"✅ Indexed {len(all_chunks)} chunks from {len(uploaded_files)} file(s)!")
        else:
            st.warning("Please upload at least one file first.")

    st.divider()

    # List Indexed Documents
    st.subheader("📚 Indexed Documents")
    indexed_sources = st.session_state.vector_manager.list_sources()
    if indexed_sources:
        for doc in indexed_sources:
            st.write(f"📄 `{doc}`")
    else:
        st.caption("No documents indexed yet.")

    st.divider()

    # Reset DB
    if st.button("🗑️ Reset Vector Database"):
        st.session_state.vector_manager.clear_database()
        st.session_state.processed_files = []
        st.session_state.messages = []
        st.success("Vector database reset!")
        st.rerun()

# --- MAIN CHAT INTERFACE ---
st.title("📄 AskMyDoc AI")
st.caption("Ask questions across your documents with exact page-level citations.")

# Render Chat History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg and msg["sources"]:
            with st.expander("🔍 View Retrieved Sources"):
                for idx, src in enumerate(msg["sources"], 1):
                    st.markdown(f"**[{idx}] {src['file']} (Page {src['page']})**")
                    st.caption(src["snippet"])
                    st.divider()

# Handle User Input
if prompt := st.chat_input("Ask a question about your uploaded documents..."):
    # Append user question
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate RAG response
    with st.chat_message("assistant"):
        with st.spinner("Retrieving context & generating answer..."):
            result = st.session_state.rag_engine.answer_query(prompt, top_k=6)
            answer = result["answer"]
            sources = result["sources"]

            st.markdown(answer)
            if sources:
                with st.expander("🔍 View Retrieved Sources"):
                    for idx, src in enumerate(sources, 1):
                        st.markdown(f"**[{idx}] {src['file']} (Page {src['page']})**")
                        st.caption(src["snippet"])
                        st.divider()

            # Save assistant message to state
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "sources": sources
            })