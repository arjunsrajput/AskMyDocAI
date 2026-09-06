import os
from typing import List
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader

class MultiFormatDocumentLoader:
    """Loads PDF, TXT, and DOCX files into LangChain Documents with enriched metadata."""

    @staticmethod
    def load_file(file_path: str) -> List[Document]:
        ext = os.path.splitext(file_path)[1].lower()
        
        if ext == ".pdf":
            loader = PyPDFLoader(file_path)
        elif ext in [".txt", ".md"]:
            loader = TextLoader(file_path, encoding="utf-8")
        elif ext == ".docx":
            loader = Docx2txtLoader(file_path)
        else:
            raise ValueError(f"Unsupported file format: {ext}")
            
        docs = loader.load()
        
        # Add uniform metadata
        for doc in docs:
            doc.metadata["source_file"] = os.path.basename(file_path)
            if "page" not in doc.metadata:
                doc.metadata["page"] = 1
            else:
                doc.metadata["page"] = doc.metadata["page"] + 1  # 1-based indexing
                
        return docs