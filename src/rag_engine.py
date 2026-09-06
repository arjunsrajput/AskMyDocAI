# from typing import Dict, Any, List
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.output_parsers import StrOutputParser
# from src.config import GEMINI_API_KEY, LLM_MODEL, TOP_K_RETRIEVAL
# from src.vector_store import VectorStoreManager

# PROMPT_TEMPLATE = """You are an expert AI research assistant. Answer the user's question using ONLY the provided context below.

# Rules to follow:
# 1. If the answer cannot be found in the context, explicitly say: "I cannot find the answer to this question in the uploaded documents." Do NOT make up information.
# 2. For every key point in your answer, cite the source document and page number in this format: [Doc: <filename>, Page: <page_number>].
# 3. Be clear, concise, and structured.

# ---
# CONTEXT:
# {context}
# ---

# USER QUESTION: {question}

# DETAILED ANSWER:"""

# def format_docs(docs):
#     formatted = []
#     for doc in docs:
#         source = doc.metadata.get("source_file", "Unknown")
#         page = doc.metadata.get("page", 1)
#         formatted.append(f"--- [Source: {source} | Page: {page}] ---\n{doc.page_content}")
#     return "\n\n".join(formatted)

# class RAGEngine:
#     """Orchestrates document retrieval, context formatting, and LLM synthesis."""

#     def __init__(self, vector_store_manager: VectorStoreManager):
#         self.vector_store_manager = vector_store_manager
#         self.llm = ChatGoogleGenerativeAI(
#             model=LLM_MODEL,
#             google_api_key=GEMINI_API_KEY,
#             temperature=0.2
#         )
#         self.prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

#     def answer_query(self, query: str, top_k: int = TOP_K_RETRIEVAL) -> Dict[str, Any]:
#         retrieved_docs = self.vector_store_manager.search_similar(query, top_k=top_k)
        
#         if not retrieved_docs:
#             return {
#                 "answer": "No documents found in database. Please upload documents first.",
#                 "sources": []
#             }

#         context_text = format_docs(retrieved_docs)
#         chain = self.prompt | self.llm | StrOutputParser()

#         response = chain.invoke({
#             "context": context_text,
#             "question": query
#         })

#         sources = []
#         for doc in retrieved_docs:
#             sources.append({
#                 "file": doc.metadata.get("source_file", "Unknown"),
#                 "page": doc.metadata.get("page", 1),
#                 "snippet": doc.page_content[:250] + "..." if len(doc.page_content) > 250 else doc.page_content
#             })

#         return {
#             "answer": response,
#             "sources": sources
#         }


import os
from typing import Dict, Any, List
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.config import GEMINI_API_KEY, LLM_MODEL, TOP_K_RETRIEVAL
from src.vector_store import VectorStoreManager

# Strict anti-hallucination prompt template
PROMPT_TEMPLATE = """You are an expert AI research assistant. Answer the user's question using ONLY the provided context below.

Rules to follow:
1. If the answer cannot be found in the context, explicitly say: "I cannot find the answer to this question in the uploaded documents." Do NOT make up information.
2. For every key point in your answer, cite the source document and page number in this format: [Doc: <filename>, Page: <page_number>].
3. Be clear, concise, and structured.

---
CONTEXT:
{context}
---

USER QUESTION: {question}

DETAILED ANSWER:"""


def format_docs(docs: List[Any]) -> str:
    """Formats retrieved chunks with their source file and page metadata."""
    formatted = []
    for doc in docs:
        source = doc.metadata.get("source_file", "Unknown")
        page = doc.metadata.get("page", 1)
        formatted.append(f"--- [Source: {source} | Page: {page}] ---\n{doc.page_content}")
    return "\n\n".join(formatted)


class RAGEngine:
    """Orchestrates document retrieval, context formatting, and LLM synthesis."""

    def __init__(self, vector_store_manager: VectorStoreManager):
        self.vector_store_manager = vector_store_manager

        # Ensure API key is found from either variable name
        api_key = GEMINI_API_KEY or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("Google API key not found. Please set GOOGLE_API_KEY or GEMINI_API_KEY in .env")

        # Initialize Gemini LLM
        self.llm = ChatGoogleGenerativeAI(
            model=LLM_MODEL,
            google_api_key=api_key,
            temperature=0.2  # Low temperature for factual consistency
        )
        self.prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

    def answer_query(self, query: str, top_k: int = TOP_K_RETRIEVAL) -> Dict[str, Any]:
        # 1. Retrieve relevant chunks from ChromaDB
        retrieved_docs = self.vector_store_manager.search_similar(query, top_k=top_k)

        if not retrieved_docs:
            return {
                "answer": "No documents found in database. Please upload and index documents first.",
                "sources": []
            }

        # 2. Format context for prompt
        context_text = format_docs(retrieved_docs)

        # 3. Build the LangChain chain
        chain = self.prompt | self.llm | StrOutputParser()

        # 4. Generate response with error safety
        try:
            response = chain.invoke({
                "context": context_text,
                "question": query
            })
        except Exception as e:
            return {
                "answer": f"Error generating answer from LLM: {str(e)}",
                "sources": []
            }

        # 5. Extract unique sources for citations
        sources = []
        seen = set()
        for doc in retrieved_docs:
            file_name = doc.metadata.get("source_file", "Unknown")
            page_num = doc.metadata.get("page", 1)
            identifier = f"{file_name}_p{page_num}"
            
            if identifier not in seen:
                seen.add(identifier)
                sources.append({
                    "file": file_name,
                    "page": page_num,
                    "snippet": doc.page_content[:250] + "..." if len(doc.page_content) > 250 else doc.page_content
                })

        return {
            "answer": response,
            "sources": sources
        }