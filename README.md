# 📄 AskMyDoc AI — Decoupled Multi-Document RAG Microservice

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/FastAPI-0.115%2B-009688?style=for-the-badge&logo=fastapi&logoColor=white" alt="FastAPI" />
  <img src="https://img.shields.io/badge/LangChain-0.3%2B-1C3C3C?style=for-the-badge" alt="LangChain" />
  <img src="https://img.shields.io/badge/ChromaDB-Vector_Store-FF6F00?style=for-the-badge" alt="ChromaDB" />
  <img src="https://img.shields.io/badge/Google_Gemini-3.6_Flash-4285F4?style=for-the-badge&logo=google&logoColor=white" alt="Google Gemini" />
  <img src="https://img.shields.io/badge/Streamlit-UI-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit" />
</p>

---

## 🌟 Overview

**AskMyDoc AI** is a production-grade, decoupled Retrieval-Augmented Generation (RAG) system built with **FastAPI**, **LangChain**, **ChromaDB**, and **Google Gemini**. 

It allows users to ingest complex, multi-format documents (PDFs, Word documents, text files), automatically parses and chunks them into semantic units, embeds them locally into a vector database, and synthesizes hallucination-resistant answers with **exact page-level citations**.

---

## 🏛️ System Architecture

```
                       ┌──────────────────────────────────────────────┐
                       │          Streamlit Web Client (UI)           │
                       │           (Running on Port 8501)             │
                       └──────────────────────┬───────────────────────┘
                                              │ HTTP JSON Requests
                                              ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│                                 FastAPI Backend (Port 8000)                                 │
│                                                                                             │
│   Endpoints:                                                                                │
│   ├── POST /api/v1/upload  ──> DocumentLoader ──> TextSplitter ──> ChromaDB (Embed & Store) │
│   ├── POST /api/v1/query   ──> ChromaDB Search ──> RAG Prompt ──> Gemini LLM ──> Citations  │
│   ├── GET  /api/v1/docs    ──> List Indexed Files                                           │
│   ├── DELETE /api/v1/reset ──> Safe Native Collection Reset                                 │
│   └── GET  /docs           ──> Interactive Swagger UI (OpenAPI)                             │
└─────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## ✨ Key Features

- **🚀 Decoupled Microservice Architecture**: The FastAPI AI backend is completely decoupled from the Streamlit frontend, exposing standard REST API endpoints suitable for any web/mobile client.
- **📑 Multi-Format Parsing**: Unified ingestion pipeline supporting `.pdf`, `.docx`, `.txt`, and `.md` formats with automated metadata enrichment (filename, 1-based page numbers).
- **🧩 Smart Semantic Chunking**: Implements `RecursiveCharacterTextSplitter` (1500 character window with 300 character overlap) preventing contextual fragmentation across headings and body sections.
- **⚡ Local HuggingFace Embeddings**: Vectorizes text using `sentence-transformers/all-MiniLM-L6-v2` locally on CPU/GPU with **zero API costs**, zero rate limits, and ultra-low embedding latency.
- **💾 Persistent Vector Storage**: ChromaDB persistent client integration with safe collection lifecycles preventing SQLite database lock issues.
- **🛡️ Anti-Hallucination Guardrails**: Low-temperature prompt constraints forcing the model to strictly ground its answers in retrieved context and generate structured citations `[Doc: <name>, Page: <number>]`.
- **📖 Interactive Swagger Documentation**: Live API testbed and schema contracts available out of the box at `/docs`.

---

## 🛠️ Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| **Backend Framework** | **FastAPI** | High-performance asynchronous REST API, CORS middleware, Pydantic validation |
| **Web Server** | **Uvicorn** | ASGI server for asynchronous request handling |
| **RAG Orchestration** | **LangChain (LCEL)** | Document loaders, recursive text splitters, runnable prompt chains |
| **Embeddings** | **HuggingFace (`all-MiniLM-L6-v2`)** | 384-dimensional dense semantic vectors (runs locally) |
| **Vector Database** | **ChromaDB** | Vector similarity search using HNSW indexing |
| **LLM Inference** | **Google Gemini (`gemini-3.6-flash`)** | Context synthesis, reasoning, and citation formatting |
| **Frontend UI** | **Streamlit** | Multi-turn chat interface, document uploader, expandable citation cards |

---

## 📂 Project Structure

```text
AskMyDoc-RAG/
├── .env.example          # Template for environment variables
├── requirements.txt      # Python package dependencies
├── main.py               # FastAPI backend microservice application
├── app.py                # Streamlit interactive frontend client
├── src/
│   ├── __init__.py
│   ├── config.py         # Centralized hyperparameters & environment configuration
│   ├── document_loader.py# Multi-format document parser & metadata extractor
│   ├── text_splitter.py  # Recursive character chunking engine
│   ├── vector_store.py   # ChromaDB vector store manager & embedding generator
│   └── rag_engine.py     # RAG prompt template, Gemini LLM chain & citation extractor
└── data/
    ├── uploads/          # Temporary storage for uploaded raw documents
    └── chroma_db/        # Persistent SQLite and vector index storage
```

---

## ⚡ Quickstart Guide

### 1. Prerequisites
- Python 3.10 or higher
- A free Google Gemini API key from [Google AI Studio](https://aistudio.google.com/)

### 2. Installation & Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/askmydoc-rag.git
cd askmydoc-rag

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Environment Configuration
Create a `.env` file in the root directory:
```ini
GOOGLE_API_KEY=your_actual_gemini_api_key_here
```

### 4. Running the Application

Open two separate terminal windows:

#### Terminal 1: Start FastAPI Backend
```bash
uvicorn main:app --reload --port 8000
```
> 📍 **Swagger API Docs**: Visit [http://localhost:8000/docs](http://localhost:8000/docs) to test endpoints interactively.

#### Terminal 2: Start Streamlit Frontend
```bash
streamlit run app.py
```
> 📍 **Web Application**: Visit [http://localhost:8501](http://localhost:8501) to interact with the UI.

---

## 📡 REST API Reference

### 1. Ingest Documents
`POST /api/v1/upload`
- **Request**: Multipart Form Data (`files: List[UploadFile]`)
- **Response**:
```json
{
  "status": "success",
  "message": "Successfully indexed 18 chunks from 2 file(s).",
  "files_processed": ["project_report.pdf", "resume.docx"],
  "total_chunks": 18
}
```

### 2. Query Documents
`POST /api/v1/query`
- **Request Body**:
```json
{
  "query": "What machine learning models were implemented?",
  "top_k": 6
}
```
- **Response**:
```json
{
  "query": "What machine learning models were implemented?",
  "answer": "The project implemented an ML pipeline using SMOTE for rebalancing and deployed it via Hugging Face Spaces [Doc: resume.pdf, Page: 1].",
  "sources": [
    {
      "file": "resume.pdf",
      "page": 1,
      "snippet": "Engineered an ML pipeline for data preprocessing and feature extraction..."
    }
  ]
}
```

### 3. List Indexed Files
`GET /api/v1/documents`
- **Response**:
```json
{
  "total_documents": 2,
  "documents": ["project_report.pdf", "resume.pdf"]
}
```

### 4. Reset Vector Database
`DELETE /api/v1/reset`
- **Response**:
```json
{
  "status": "success",
  "message": "Vector database has been reset."
}
```

---

## 📈 Key Design Choices & RAG Optimizations

1. **Chunk Boundary Tuning**: Configured a `1500` character window with `300` character overlap to preserve semantic context across multi-line resume project titles and section headers.
2. **Local Vectorization**: Decoupled embeddings from external cloud APIs by leveraging `all-MiniLM-L6-v2`, eliminating rate limits and billing overhead.
3. **Safe Database Lifecycles**: Utilized Chroma's native collection deletion APIs (`delete_collection`) rather than filesystem deletions to prevent SQLite file-lock conflicts.
4. **Source Deduplication**: Implemented citation filtering to prevent redundant page citations while preserving provenance.

---

## 📜 License

Distributed under the MIT License. See `LICENSE` for more information.

---

## 👨‍💻 Author

**Arjun Singh Panwar**  
- GitHub: [@yourusername](https://github.com/arjunsrajput)
- LinkedIn: [linkedin.com/in/yourprofile](https://linkedin.com/in/arjun-s-rajput)
