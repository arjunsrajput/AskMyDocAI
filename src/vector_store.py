import os
import chromadb
from typing import List
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from src.config import EMBEDDING_MODEL, PERSIST_DIRECTORY

class VectorStoreManager:
    """Manages ChromaDB vector indexing and safe resets without SQLite lock issues."""

    def __init__(self, persist_dir: str = PERSIST_DIRECTORY):
        self.persist_dir = persist_dir
        self.collection_name = "rag_documents"
        
        # Load local HuggingFace embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL
        )
        self._init_db()

    def _init_db(self):
        """Initializes persistent Chroma client and vector store."""
        os.makedirs(self.persist_dir, exist_ok=True)
        self.client = chromadb.PersistentClient(path=self.persist_dir)
        self.vector_store = Chroma(
            client=self.client,
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
        )

    def add_documents(self, chunks: List[Document]):
        """Adds and embeds document chunks into the vector store."""
        if chunks:
            self.vector_store.add_documents(chunks)

    def search_similar(self, query: str, top_k: int = 4) -> List[Document]:
        """Performs cosine similarity search against stored embeddings."""
        return self.vector_store.similarity_search(query, k=top_k)

    def list_sources(self) -> List[str]:
        """Returns list of unique document filenames in the store."""
        try:
            data = self.vector_store.get()
            if not data or "metadatas" not in data or not data["metadatas"]:
                return []
            sources = set(m.get("source_file") for m in data["metadatas"] if m and "source_file" in m)
            return sorted(list(sources))
        except Exception:
            return []

    def clear_database(self):
        """Safely resets the collection without destroying SQLite file locks."""
        try:
            # Delete collection using Chroma's native client API
            self.client.delete_collection(self.collection_name)
        except Exception:
            pass  # If collection doesn't exist yet, ignore
            
        # Re-initialize collection cleanly
        self.vector_store = Chroma(
            client=self.client,
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
        )