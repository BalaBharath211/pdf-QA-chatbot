# app.py
import streamlit as st
import os
from rag_core import build_vector_store, answer_query

st.set_page_config(page_title="PDF Q&A – Gemini RAG", layout="wide", page_icon="📄")
st.title("📄 PDF Question Answering – Gemini RAG Agent")

with st.sidebar:
    st.subheader("🔐 API Key")
    api_key = st.text_input("Enter Gemini API Key", type="password")
    st.subheader("📄 Upload PDF")
    pdf_file = st.file_uploader("Upload a PDF", type=["pdf"])
    process_btn = st.button("⚙️ Process Document")

# Put key into env for rag_core
if api_key:
    os.environ["GOOGLE_API_KEY"] = api_key

if process_btn:
    if not pdf_file:
        st.warning("Please upload a PDF first.")
        st.stop()
    with st.spinner("Indexing PDF…"):
        pdf_path = f"data/{pdf_file.name}"
        with open(pdf_path, "wb") as f:
            f.write(pdf_file.getvalue())
        st.session_state.vector_store = build_vector_store(pdf_path)
    st.success("PDF processed! You can now ask questions.")

if "vector_store" in st.session_state:
    st.subheader("💬 Ask Your Questions")
    user_input = st.chat_input("Ask something about the PDF...")
    if user_input:
        with st.spinner("Thinking..."):
            answer, sources = answer_query(st.session_state.vector_store, user_input)
        st.chat_message("assistant").markdown(answer)
        if sources:
            st.markdown("### 📚 Sources")
            for s in sources:
                st.markdown(f"- {s}")
