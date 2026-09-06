import os
from dotenv import load_dotenv

load_dotenv()

# API Keys
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# # Chunking Configuration
# CHUNK_SIZE = 1000        # Characters per chunk (~150-200 words)
# CHUNK_OVERLAP = 200      # Boundary overlap to preserve context

# # Retrieval Configuration
# TOP_K_RETRIEVAL = 4      # Number of closest chunks to retrieve

# In src/config.py:

CHUNK_SIZE = 1500        # Increased from 1000 to keep full sections together
CHUNK_OVERLAP = 300      # Increased from 200 so headers carry over into next chunk
TOP_K_RETRIEVAL = 6      # Increased from 4 to retrieve more context for broad questions

# Model Names
LLM_MODEL = "gemini-3.6-flash"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Storage Paths
PERSIST_DIRECTORY = "./data/chroma_db"
UPLOAD_DIRECTORY = "./data/uploads"

# API URL for Frontend
FASTAPI_BACKEND_URL = "http://localhost:8000"