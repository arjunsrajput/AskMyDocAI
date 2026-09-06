import streamlit as st
import requests
from src.config import FASTAPI_BACKEND_URL

st.set_page_config(page_title="AskMyDoc AI", page_icon="📄", layout="wide")

# Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- SIDEBAR ---
with st.sidebar:
    st.title("📄 AskMyDoc AI")
    st.caption("Ask anything across your documents with exact page-level citations.")

    # Upload Section
    uploaded_files = st.file_uploader(
        "Upload Documents",
        type=["pdf", "docx", "txt", "md"],
        accept_multiple_files=True
    )

    if st.button("📤 Upload & Index to API", type="primary"):
        if uploaded_files:
            with st.spinner("Sending documents to FastAPI backend..."):
                files_payload = [("files", (f.name, f.getvalue(), f.type)) for f in uploaded_files]
                try:
                    res = requests.post(f"{FASTAPI_BACKEND_URL}/api/v1/upload", files=files_payload)
                    if res.status_code == 200:
                        data = res.json()
                        st.success(f"✅ {data['message']}")
                    else:
                        st.error(f"Error {res.status_code}: {res.text}")
                except Exception as e:
                    st.error(f"Failed to connect to FastAPI backend: {e}")
        else:
            st.warning("Please select files first.")

    st.divider()

    # View Indexed Docs
    st.subheader("📚 Indexed Documents")
    try:
        docs_res = requests.get(f"{FASTAPI_BACKEND_URL}/api/v1/documents")
        if docs_res.status_code == 200:
            docs = docs_res.json().get("documents", [])
            if docs:
                for doc in docs:
                    st.write(f"📄 `{doc}`")
            else:
                st.caption("No documents indexed yet.")
    except Exception:
        st.caption("Backend offline. Please start FastAPI.")

    st.divider()

    # Reset DB Button
    if st.button("🗑️ Reset Vector DB"):
        try:
            res = requests.delete(f"{FASTAPI_BACKEND_URL}/api/v1/reset")
            if res.status_code == 200:
                st.session_state.messages = []
                st.success("Vector database reset!")
                st.rerun()
        except Exception as e:
            st.error(f"Error: {e}")

# --- MAIN CHAT INTERFACE ---
st.title("Multi-Document Research Copilot")
st.caption("Powered by **FastAPI**, **LangChain**, **ChromaDB**, and **Google Gemini**.")

# Display Chat History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg and msg["sources"]:
            with st.expander("🔍 View Retrieved Sources"):
                for idx, src in enumerate(msg["sources"], 1):
                    st.markdown(f"**[{idx}] {src['file']} (Page {src['page']})**")
                    st.caption(src["snippet"])
                    st.divider()

# User Input
if prompt := st.chat_input("Ask a question about your uploaded documents..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("FastAPI is retrieving context and generating answer..."):
            try:
                response = requests.post(
                    f"{FASTAPI_BACKEND_URL}/api/v1/query",
                    json={"query": prompt, "top_k": 4}
                )
                if response.status_code == 200:
                    data = response.json()
                    answer = data["answer"]
                    sources = data["sources"]

                    st.markdown(answer)
                    if sources:
                        with st.expander("🔍 View Retrieved Sources"):
                            for idx, src in enumerate(sources, 1):
                                st.markdown(f"**[{idx}] {src['file']} (Page {src['page']})**")
                                st.caption(src["snippet"])
                                st.divider()

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources
                    })
                else:
                    st.error(f"Backend error: {response.status_code} - {response.text}")
            except Exception as e:
                st.error(f"Could not connect to FastAPI backend at {FASTAPI_BACKEND_URL}. Ensure it is running.")