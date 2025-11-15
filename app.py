# app.py - IMPROVED WITH RATE LIMITING
import streamlit as st
import os
import time
from dotenv import load_dotenv
from rag_core import build_vector_store, answer_query

# Load .env API key
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    st.error("❌ GOOGLE_API_KEY not found in .env file")
    st.stop()

os.environ["GOOGLE_API_KEY"] = api_key

st.set_page_config(page_title="PDF Q&A – Gemini RAG", layout="wide", page_icon="📚")
st.title("📚 Multi-PDF Q&A – Gemini RAG Agent")

# Initialize session state for rate limiting
if "last_query_time" not in st.session_state:
    st.session_state.last_query_time = 0

if "query_count" not in st.session_state:
    st.session_state.query_count = 0

if "query_cache" not in st.session_state:
    st.session_state.query_cache = {}

# Sidebar
with st.sidebar:
    st.subheader("📚 Upload PDFs")
    pdf_files = st.file_uploader("Upload one or more PDFs", type=["pdf"], accept_multiple_files=True)
    process_btn = st.button("⚙️ Process Documents")


# Process PDFs
if process_btn:
    if not pdf_files:
        st.warning("Please upload at least one PDF first.")
        st.stop()

    os.makedirs("data", exist_ok=True)

    saved_paths = []
    for pdf in pdf_files:
        path = f"data/{pdf.name}"
        with open(path, "wb") as f:
            f.write(pdf.getvalue())
        saved_paths.append(path)

    with st.spinner("📊 Processing PDFs…"):
        st.session_state.vector_store = build_vector_store(saved_paths)

    st.success("✅ All PDFs processed! Ask your questions below.")
    
    # Reset query stats when new docs are processed
    st.session_state.query_count = 0
    st.session_state.query_cache = {}

# Q&A Interface
if "vector_store" in st.session_state:
    st.subheader("💬 Ask Your Questions")
    
    # Show rate limit warning if needed
    time_since_last = time.time() - st.session_state.last_query_time
    if st.session_state.last_query_time > 0 and time_since_last < 5:
        remaining = 5 - time_since_last
        st.warning(f"⏳ Please wait {remaining:.1f} more seconds to avoid rate limits...")
    
    user_input = st.chat_input("Ask something about your uploaded documents...")

    if user_input:
        # Check if query is cached
        query_hash = hash(user_input.lower().strip())
        
        if query_hash in st.session_state.query_cache:
            st.info("💾 Using cached answer (asked this before)")
            answer, sources = st.session_state.query_cache[query_hash]
        else:
            # Rate limiting check
            time_since_last = time.time() - st.session_state.last_query_time
            
            # Enforce 5-second minimum delay (except for first query)
            if st.session_state.last_query_time > 0 and time_since_last < 5:
                wait_time = 5 - time_since_last
                st.warning(f"⏳ Rate limiting: waiting {wait_time:.1f} seconds...")
                
                # Show countdown
                progress_bar = st.progress(0)
                for i in range(int(wait_time * 10)):
                    time.sleep(0.1)
                    progress_bar.progress((i + 1) / (wait_time * 10))
                progress_bar.empty()
            
            # Query the RAG system
            with st.spinner("🤔 Analyzing documents…"):
                try:
                    answer, sources = answer_query(st.session_state.vector_store, user_input)
                    
                    # Cache the answer
                    st.session_state.query_cache[query_hash] = (answer, sources)
                    
                    # Update stats
                    st.session_state.query_count += 1
                    st.session_state.last_query_time = time.time()
                    
                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    st.stop()

        # Display answer
        st.chat_message("assistant").markdown(answer)

        # Display sources
        if sources:
            with st.expander("📚 View Sources"):
                for i, source in enumerate(sources, 1):
                    st.markdown(f"**{i}.** {source}")
        
        # Show query stats
        if st.session_state.query_count > 10:
            st.info(f"💡 **Tip:** You've asked {st.session_state.query_count} questions. "
                   f"Consider waiting a minute if you encounter rate limits.")

else:
    st.info("👆 Please upload and process PDFs first using the sidebar.")