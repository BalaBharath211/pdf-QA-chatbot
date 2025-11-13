# rag_core.py
import os
import streamlit as st
import google.generativeai as genai
from typing import List, Tuple

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# --------------------
# Config - ADJUST THESE VALUES TO CONTROL CHUNK SIZE
# --------------------
CHUNK_SIZE = 2000  # Increased from 1000 to 2000 characters
CHUNK_OVERLAP = 200  # Increased from 100 to maintain context between chunks
EMBED_MODEL = "sentence-transformers/all-mpnet-base-v2"
TOP_K = 4  # Number of chunks to retrieve
FINAL_K = 3  # Number of chunks to use in final context

# Note: Larger chunks provide more context but may include less relevant information
# Smaller chunks are more precise but may miss important context
# Recommended ranges:
# - CHUNK_SIZE: 500-3000 characters
# - CHUNK_OVERLAP: 10-20% of CHUNK_SIZE


# --------------------
# Build vector store (cached)
# --------------------
@st.cache_resource
def build_vector_store(pdf_path: str) -> FAISS:
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    if not docs:
        raise ValueError("No text loaded from PDF (maybe scanned image?).")

    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". ", " "],
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(docs)
    
    # Print chunk statistics for debugging
    print(f"Created {len(chunks)} chunks")
    print(f"Average chunk size: {sum(len(c.page_content) for c in chunks) / len(chunks):.0f} characters")

    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore


# --------------------
# Simple answer function using direct Gemini API
# --------------------
def answer_query(vectorstore: FAISS, query: str) -> Tuple[str, List[str]]:
    # Get API key from environment
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        return "Please enter your Gemini API key in the sidebar.", []
    
    # Configure genai
    genai.configure(api_key=api_key)
    
    # Retriever
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": TOP_K})
    candidate_docs = retriever.invoke(query)
    if not candidate_docs:
        return "The answer is not found in the PDF.", []

    # choose top FINAL_K docs (no reranker)
    top_docs = candidate_docs[:FINAL_K]

    # Prepare context and sources
    context_parts = []
    sources = []
    total_context_size = 0
    
    for d in top_docs:
        meta = d.metadata or {}
        src = os.path.basename(meta.get("source", "unknown"))
        page = meta.get("page", None)
        header = f"[Source: {src}" + (f" | page {page+1}]" if page is not None else "]")
        context_parts.append(f"{header}\n{d.page_content}")
        total_context_size += len(d.page_content)
        
        if page is not None:
            sources.append(f"{src} (page {page+1})")
        else:
            sources.append(src)

    context = "\n\n---\n\n".join(context_parts)
    
    # Debug info (optional - remove in production)
    print(f"Total context size: {total_context_size} characters across {len(top_docs)} chunks")

    # Prompt
    prompt = f"""You are a helpful assistant. Use ONLY the provided context to answer the user's question.
If the answer is not contained in the context, reply: "The answer is not found in the PDF."

Context:
{context}

Question:
{query}

Answer:
"""

    # Use the CONFIRMED WORKING model: gemini-2.0-flash
    try:
        model = genai.GenerativeModel('gemini-2.0-flash')
        response = model.generate_content(prompt)
        answer_text = response.text
    except Exception as e:
        return f"LLM error: {str(e)}", sources

    return answer_text.strip(), sources