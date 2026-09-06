# 🎓 AskMyDoc AI — Master Technical Interview Guide & Architecture Breakdown

This comprehensive guide breaks down the entire **AskMyDoc** RAG project file-by-file, explaining every architectural decision, internal function, data flow, and potential interview questions with model answers.

---

## 📌 Table of Contents
1. [30-Second Elevator Pitch & 2-Minute Deep Dive](#1-elevator-pitch--deep-dive)
2. [End-to-End System Architecture & Data Flow](#2-system-architecture--data-flow)
3. [Deep-Dive File-by-File Technical Breakdown](#3-deep-dive-file-by-file-breakdown)
   - [A. `requirements.txt` & `.env`](#a-requirementstxt--env)
   - [B. `src/config.py`](#b-srcconfigpy)
   - [C. `src/document_loader.py`](#c-srcdocument_loaderpy)
   - [D. `src/text_splitter.py`](#d-srctext_splitterpy)
   - [E. `src/vector_store.py`](#e-srcvector_storepy)
   - [F. `src/rag_engine.py`](#f-srcrag_enginepy)
   - [G. `main.py` (FastAPI Microservice Backend)](#g-mainpy-fastapi-backend)
   - [H. `app.py` (Streamlit Web Client)](#h-apppy-streamlit-frontend)
4. [Key Engineering Trade-offs & Design Choices](#4-key-engineering-trade-offs)
5. [Real Bugs Encountered & How We Solved Them (Great Interview Stories)](#5-real-debugging-stories)
6. [Top 10 Technical Interview Q&A](#6-top-10-technical-interview-qa)

---

## 1. Elevator Pitch & Deep Dive

### ⏱️ The 30-Second Elevator Pitch
> *"I built **AskMyDoc AI**, a production-grade, decoupled Retrieval-Augmented Generation (RAG) microservice. It allows users to upload multiple documents across formats like PDF and DOCX, indexes them into a local ChromaDB vector store with semantic embeddings, and uses FastAPI and Google Gemini to generate precise, hallucination-free answers backed by verifiable page-level citations."*

### 🎙️ The 2-Minute Technical Summary
> *"The project is architected as a clean client-server microservice. The backend is built with **FastAPI**, handling asynchronous document uploads, metadata extraction, recursive character chunking (1500 characters with 300 overlap), and ChromaDB vector indexing. 
> For embeddings, I chose **HuggingFace’s `all-MiniLM-L6-v2`** to run vectorization locally with zero API latency and zero cost. For generation, I engineered strict anti-hallucination prompt guardrails with **Google Gemini**, enforcing cross-document source citations formatted as `[Doc: <name>, Page: <page>]`. 
> The presentation layer is an interactive **Streamlit** client communicating via REST endpoints. I also addressed critical RAG edge cases like chunk boundary fragmentation and SQLite persistent lock issues."*

---

## 2. System Architecture & Data Flow

```
========================================================================================
                                1. INGESTION PIPELINE
========================================================================================
[User Uploads Files in Streamlit]
        │
        ▼ (HTTP Multipart POST /api/v1/upload)
[FastAPI async handler in main.py] ──> Writes buffer to disk
        │
        ▼
[src/document_loader.py] ──> PyPDFLoader / Docx2txtLoader / TextLoader
                             • Extracts text per page
                             • Attaches metadata: {source_file, page}
        │
        ▼
[src/text_splitter.py]   ──> RecursiveCharacterTextSplitter
                             • Chunk Size: 1500 chars | Overlap: 300 chars
                             • Attaches chunk_id
        │
        ▼
[src/vector_store.py]    ──> HuggingFaceEmbeddings ('all-MiniLM-L6-v2')
                             • Computes 384-dimensional dense vectors
                             • Stores in ChromaDB collection 'rag_documents'

========================================================================================
                                2. QUERY & GENERATION PIPELINE
========================================================================================
[User Question in Streamlit UI]
        │
        ▼ (HTTP JSON POST /api/v1/query {"query": "...", "top_k": 6})
[FastAPI /api/v1/query in main.py]
        │
        ▼
[src/rag_engine.py]
        │
        ├──> 1. Vector Search: ChromaDB similarity_search(query, k=6)
        │       Retrieves Top-6 most semantically relevant text chunks
        │
        ├──> 2. Context Aggregation: Formats chunks with headers:
        │       "--- [Source: doc.pdf | Page: 1] --- \n <text>"
        │
        ├──> 3. Prompt Injection: Injects context & query into PROMPT_TEMPLATE
        │
        ├──> 4. LLM Synthesis: Gemini 3.6 Flash (temperature=0.2)
        │       Synthesizes answer and cites sources
        │
        └──> 5. Source Extraction: Deduplicates and packages citations
        │
        ▼ (HTTP JSON Response)
[Streamlit app.py] ──> Displays markdown answer + Expandable Citation Cards
```

---

## 3. Deep-Dive File-by-File Breakdown

---

### A. `requirements.txt` & `.env`

#### Purpose:
Declares dependencies and isolates private environment variables.

#### Key Packages Explained:
- `fastapi` & `uvicorn`: High-performance ASGI framework and web server for asynchronous REST APIs.
- `python-multipart`: Required for handling binary file uploads (`multipart/form-data`) in FastAPI.
- `pydantic`: Enforces strict data types for incoming requests and outgoing API responses.
- `langchain` & `langchain-community`: Core abstraction framework for document loaders, text splitters, and vector stores.
- `langchain-google-genai`: LangChain integration for Google Gemini models.
- `sentence-transformers`: Local neural network models to generate embeddings directly on CPU/GPU without external API calls.
- `chromadb`: High-performance, open-source embedded vector database.
- `pypdf` & `python-docx`: Document parsing engines.
- `streamlit`: Pure Python frontend interface.

---

### B. `src/config.py`

#### Purpose:
Central source of truth for all configurable hyperparameters and environment variables.

#### Key Code Components:
```python
CHUNK_SIZE = 1500        # Chunk length in characters
CHUNK_OVERLAP = 300      # Overlapping characters between adjacent chunks
TOP_K_RETRIEVAL = 6      # Number of nearest neighbors to retrieve
LLM_MODEL = "gemini-3.6-flash"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
```

#### Why it matters for interviews:
- Decouples configuration from application logic (Factor III of the *Twelve-Factor App* methodology).
- Makes it easy to tune RAG parameters during experiments without touching core engine code.

---

### C. `src/document_loader.py`

#### Purpose:
Uniformly ingests diverse file formats (`.pdf`, `.docx`, `.txt`, `.md`) and standardizes them into LangChain `Document` objects.

#### How it works:
1. Examines file extension via `os.path.splitext(file_path)[1]`.
2. Dispatches to the appropriate parser:
   - `PyPDFLoader`: Splits PDF into individual page objects.
   - `Docx2txtLoader`: Parses Microsoft Word `.docx` documents.
   - `TextLoader`: Reads raw text and markdown files with UTF-8 encoding.
3. **Metadata Normalization**: Normalizes page numbers from 0-indexed to 1-indexed so citations match physical document pages.

---

### D. `src/text_splitter.py`

#### Purpose:
Breaks large documents into semantically coherent chunks while maintaining overlap across boundaries.

#### How it works:
Uses `RecursiveCharacterTextSplitter` with separator hierarchy:
```python
separators=["\n\n", "\n", ". ", " ", ""]
```
1. It first attempts to split on paragraph boundaries (`\n\n`).
2. If a paragraph is longer than 1500 characters, it splits on single line breaks (`\n`).
3. If still too long, it splits on sentences (`. `).
4. As a last resort, it splits on spaces (` `) or individual characters.

#### Why `CHUNK_SIZE=1500` and `CHUNK_OVERLAP=300`?
- **1500 characters (~200–250 words)** encapsulates complete ideas (e.g. an entire resume project description or a full technical explanation).
- **300 characters overlap** prevents **Boundary Information Loss**, ensuring titles/headers in one chunk remain attached to the context in the next.

---

### E. `src/vector_store.py`

#### Purpose:
Embeds text chunks into mathematical vectors and manages indexing, similarity retrieval, and database resets in ChromaDB.

#### How it works:
1. **Embedding**: Uses `HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")`. Each text chunk is converted into a **384-dimensional vector**.
2. **Persistent Client**: Initializes ChromaDB using `chromadb.PersistentClient(path="./data/chroma_db")` to persist embeddings to disk.
3. **Similarity Search**: Performs **Cosine Similarity / Approximate Nearest Neighbor (ANN)** search using HNSW (Hierarchical Navigable Small World) graphs to find the top-$k$ closest chunks to a query vector.
4. **Safe Reset**: Uses `self.client.delete_collection("rag_documents")` instead of deleting the directory with `shutil.rmtree`, preventing SQLite database file-lock corruption (Code 1032).

---

### F. `src/rag_engine.py`

#### Purpose:
Orchestrates retrieval, context formatting, prompt engineering, anti-hallucination guardrails, and Gemini LLM inference.

#### Key Components:
1. **Prompt Guardrail**:
   ```text
   Rules:
   1. If the answer cannot be found in the context, explicitly say:
      "I cannot find the answer to this question in the uploaded documents."
   2. For every key point in your answer, cite the source document and page number:
      [Doc: <filename>, Page: <page_number>].
   ```
2. **Temperature = 0.2**: Low temperature forces the LLM to be deterministic and strictly factual.
3. **LangChain Expression Language (LCEL)**:
   ```python
   chain = self.prompt | self.llm | StrOutputParser()
   ```
   Pipes the prompt into Gemini and parses the output into a clean string.
4. **Source Deduplication**: Tracks seen `file_name + page_number` pairs so citations in the UI are clean and unique.

---

### G. `main.py` (FastAPI Backend)

#### Purpose:
Exposes the RAG system as high-throughput, asynchronous REST API endpoints with auto-generated OpenAPI / Swagger docs.

#### Key Endpoints:
| HTTP Verb | Endpoint | Purpose |
|---|---|---|
| `GET` | `/health` | Liveness check |
| `POST` | `/api/v1/upload` | Async multipart file upload, chunking, and ChromaDB indexing |
| `POST` | `/api/v1/query` | RAG query pipeline returning generated answer and source citations |
| `GET` | `/api/v1/documents` | Lists all indexed document filenames |
| `DELETE` | `/api/v1/reset` | Resets the vector database |

#### Technical Highlights:
- **Pydantic Models**: `QueryRequest`, `QueryResponse`, `SourceCitation`, and `UploadResponse` provide automated request validation and response serialization.
- **Asynchronous I/O**: `async def upload_documents` uses `await file.read()` for non-blocking file streaming.
- **CORS Middleware**: Allows web frontends hosted on different domains/ports to interact with the API safely.
- **Swagger Documentation**: Interactive documentation automatically rendered at `/docs`.

---

### H. `app.py` (Streamlit Frontend Client)

#### Purpose:
Provides a clean, reactive chat UI for users to upload files, manage documents, ask questions, and inspect citation cards.

#### Key Features:
- **Session State Management**: Persists chat history across re-renders (`st.session_state.messages`).
- **REST Communication**: Communicates with the FastAPI backend over HTTP using Python's `requests` library.
- **Expandable Citations**: Renders collapsible cards (`st.expander`) displaying exact file names, page numbers, and text snippets used by the LLM.

---

## 4. Key Engineering Trade-offs

| Decision | What We Chose | Alternative Considered | Why Our Choice Is Better |
|---|---|---|---|
| **Architecture** | Decoupled Microservice (FastAPI + Streamlit) | Monolithic Streamlit Script | Industry standard; allows backend reuse for React, mobile apps, or enterprise integrations. |
| **Embeddings** | Local `all-MiniLM-L6-v2` | Cloud API Embeddings | Zero API costs, zero rate limits, runs offline, lower latency. |
| **Vector DB** | ChromaDB (Embedded Persistent) | In-Memory FAISS or Cloud Pinecone | Lightweight, no cloud subscription required, persists to disk across server restarts. |
| **LLM Model** | Gemini 3.6 Flash (Temp: 0.2) | GPT-4 / Llama 3 | Fast inference speed, generous free-tier limits, strong reasoning. |
| **Text Splitter** | Recursive Character (1500 / 300) | Fixed-size Character Splitter | Respects natural paragraph and sentence boundaries, avoiding fragmented clauses. |

---

## 5. Real Debugging Stories

When interviewers ask: *"Tell me about a technical bug you faced and how you solved it"*, you have 3 real stories:

### Story 1: Information Fragmentation at Chunk Boundaries
- **Problem**: When querying project titles from a resume, the LLM returned project descriptions but said titles were missing.
- **Root Cause**: Fixed chunk size of 1000 characters split the project heading into chunk $N$ and the bullet points into chunk $N+1$.
- **Resolution**: Increased `CHUNK_SIZE` to 1500 and `CHUNK_OVERLAP` to 300, and increased retrieval `TOP_K` from 4 to 6.

### Story 2: SQLite File Lock (Database Readonly Error 1032)
- **Problem**: Calling the reset endpoint crashed with `attempt to write a readonly database (code: 1032)`.
- **Root Cause**: Deleting the folder on disk using `shutil.rmtree` while SQLite connections remained open corrupted the file lock.
- **Resolution**: Switched to Chroma's native client API (`client.delete_collection()`), cleanly resetting vector data without disturbing the file system handles.

### Story 3: Non-blocking Async File Uploads in FastAPI
- **Problem**: Uploaded files generated 0 chunks in ChromaDB.
- **Root Cause**: `shutil.copyfileobj` on FastAPI's `UploadFile` in an `async` route failed to read the unbuffered stream properly.
- **Resolution**: Replaced with `content = await file.read()` followed by a clean buffer write.

---

## 6. Top 10 Technical Interview Q&A

### Q1: What is RAG and why is it preferred over fine-tuning for knowledge retrieval?
> **Answer:** RAG combines an information retrieval system with a generative LLM. It retrieves factual context dynamically at runtime and injects it into the prompt. 
> - **RAG vs Fine-Tuning**: Fine-tuning adjusts model weights to learn styles or domain vocabulary but struggles with dynamic facts and causes hallucinations. RAG allows real-time data updates without retraining, guarantees source citations, and costs significantly less.

### Q2: How does Dense Vector Search work under the hood?
> **Answer:** An embedding model maps text into a high-dimensional vector space where semantically similar phrases are located close to each other. When a user asks a question, ChromaDB embeds the query into the same vector space and calculates the **Cosine Similarity** against stored vectors using Approximate Nearest Neighbor (ANN) indexing like **HNSW (Hierarchical Navigable Small World)** graphs.

### Q3: How do you prevent LLM hallucinations in this project?
> **Answer:**
> 1. **Prompt Guardrails**: Explicitly directing the model to answer *only* using the provided context and return a standard refusal if information is absent.
> 2. **Low Temperature (0.2)**: Reduces token sampling randomness, forcing factual responses.
> 3. **Source Attribution**: Forcing `[Doc: <name>, Page: <page>]` citation formats so claims can be audited against raw text.

### Q4: Why did you choose Recursive Character Splitting over simple character splitting?
> **Answer:** Simple character splitting slices text at arbitrary character limits, cutting words and sentences in half. `RecursiveCharacterTextSplitter` tries a hierarchy of natural boundaries (`\n\n` -> `\n` -> `. ` -> ` `), preserving paragraph integrity and semantic coherence.

### Q5: What is the purpose of Chunk Overlap?
> **Answer:** Chunk overlap ensures that context spanning across chunk boundaries is not lost. For example, if a sentence introduces a concept at the end of Chunk 1 and explains it at the start of Chunk 2, a 300-character overlap guarantees both chunks retain the complete context.

### Q6: What is the difference between Top-K retrieval and Reranking?
> **Answer:** Top-K retrieval uses Bi-Encoder embeddings to quickly find the top-$K$ candidates from millions of vectors using cosine similarity. A **Reranker (Cross-Encoder)** takes those top-$K$ candidates along with the query, compares all token-level cross-attentions, and reorders them by true relevance before passing to the LLM.

### Q7: What are the advantages of using FastAPI over Flask or Django for this project?
> **Answer:**
> 1. **Asynchronous Support (`async/await`)**: Handles concurrent I/O-bound operations (file uploads, embedding computations, LLM API calls) efficiently.
> 2. **Pydantic Data Validation**: Validates request/response schemas automatically.
> 3. **Auto-Generated OpenAPI/Swagger Docs**: Live interactive API documentation at `/docs`.

### Q8: What is Mean Reciprocal Rank (MRR) and Hit Rate in RAG evaluation?
> **Answer:**
> - **Hit Rate@K**: The percentage of test queries for which the correct document chunk appeared anywhere within the top-$K$ retrieved results.
> - **Mean Reciprocal Rank (MRR)**: Evaluates the exact ranking position of the first relevant chunk ($1/\text{rank}$). If the correct document is ranked 1st, score is 1.0; if 2nd, 0.5.

### Q9: How would you scale this system to handle millions of documents?
> **Answer:**
> 1. Migrate from embedded ChromaDB to a distributed vector database like **Milvus, Qdrant, or Pinecone**.
> 2. Implement **Hybrid Search** (BM25 keyword search + Dense Vector Search with Reciprocal Rank Fusion).
> 3. Offload document ingestion and embedding computation to background worker queues using **Celery / Redis**.
> 4. Add semantic caching (e.g. **GPTCache**) to return cached responses for semantically identical queries.

### Q10: How do you handle non-text files like scanned PDFs or images?
> **Answer:** Standard PDF loaders extract digital text streams. For scanned or image-based PDFs, an **OCR (Optical Character Recognition) pipeline** using Tesseract or a multimodal Vision LLM (e.g. Gemini Vision) would be integrated into `src/document_loader.py` to extract text prior to chunking.
