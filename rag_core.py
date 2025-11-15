# rag_core.py - IMPROVED GENERAL-PURPOSE VERSION
import os
import re
import time
from typing import List, Tuple, Dict
import google.generativeai as genai

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings


# 🔥 IMPROVED RAG PARAMETERS
CHUNK_SIZE = 3000
CHUNK_OVERLAP = 400
TOP_K = 15              # Retrieve more candidates for better coverage
FINAL_K = 8             # Use more chunks for context
EMBED_MODEL = "sentence-transformers/all-mpnet-base-v2"

# Query type detection keywords
COMPARISON_KEYWORDS = ['compare', 'comparison', 'difference', 'versus', 'vs', 'vs.', 
                       'distinguish', 'contrast', 'better', 'worse', 'trade-off', 'tradeoff']

EXPLANATION_KEYWORDS = ['explain', 'what is', 'what are', 'how does', 'describe', 'define']

MULTI_ASPECT_KEYWORDS = ['all', 'every', 'complete', 'comprehensive', 'detailed', 
                         'various', 'different types', 'kinds of']


# ------------------------------------------------------------
#  RETRY LOGIC FOR API CALLS
# ------------------------------------------------------------

def get_available_gemini_models() -> List[str]:
    """
    Get list of available Gemini models for content generation.
    Returns models in order of preference.
    """
    try:
        api_key = os.getenv("GOOGLE_API_KEY")
        genai.configure(api_key=api_key)
        
        # List all available models
        available_models = []
        for model in genai.list_models():
            # Check if model supports generateContent
            if 'generateContent' in model.supported_generation_methods:
                available_models.append(model.name)
        
        # Prefer flash models, then pro, then any gemini model
        preferred_order = []
        
        # First priority: Flash models
        for model_name in available_models:
            if 'flash' in model_name.lower():
                preferred_order.append(model_name.replace('models/', ''))
        
        # Second priority: Pro models
        for model_name in available_models:
            if 'pro' in model_name.lower() and model_name not in preferred_order:
                preferred_order.append(model_name.replace('models/', ''))
        
        # Third priority: Any other gemini model
        for model_name in available_models:
            if 'gemini' in model_name.lower() and model_name not in preferred_order:
                preferred_order.append(model_name.replace('models/', ''))
        
        return preferred_order if preferred_order else ['gemini-pro']
        
    except Exception as e:
        # Fallback to known working models
        print(f"Warning: Could not list models ({e}). Using fallback list.")
        return [
            'gemini-pro',
            'gemini-1.5-pro-latest', 
            'gemini-1.5-flash-latest',
            'gemini-1.0-pro'
        ]


def query_gemini_with_retry(prompt: str, max_retries: int = 3) -> str:
    """
    Query Gemini API with automatic retry on rate limit errors.
    Uses exponential backoff for retries.
    """
    api_key = os.getenv("GOOGLE_API_KEY")
    genai.configure(api_key=api_key)
    
    # Get available models dynamically
    models_to_try = get_available_gemini_models()[:3]  # Try top 3
    
    if not models_to_try:
        raise Exception("No Gemini models available. Check your API key and billing status.")
    
    print(f"🤖 Available models: {models_to_try}")
    
    for model_name in models_to_try:
        for attempt in range(max_retries):
            try:
                model = genai.GenerativeModel(model_name)
                response = model.generate_content(prompt)
                return response.text
                
            except Exception as e:
                error_msg = str(e)
                
                # Check if it's a rate limit error
                if "429" in error_msg or "quota" in error_msg.lower() or "rate limit" in error_msg.lower():
                    wait_time = (attempt + 1) * 20  # 20, 40, 60 seconds
                    
                    if attempt < max_retries - 1:
                        print(f"⏳ Rate limit hit. Waiting {wait_time} seconds before retry {attempt + 2}/{max_retries}...")
                        time.sleep(wait_time)
                        continue
                    else:
                        # Last retry failed, try next model
                        break
                else:
                    # Not a rate limit error, raise immediately
                    raise Exception(f"Gemini API Error ({model_name}): {error_msg}")
    
    # All retries and models failed
    raise Exception(
        "⚠️ **Rate Limit Reached - All Retries Exhausted**\n\n"
        "The Gemini API free tier limits have been exceeded.\n\n"
        "**Immediate Solutions:**\n"
        "1. Wait 2-3 minutes before trying again\n"
        "2. Reduce the number of questions you ask per minute (max ~15)\n"
        "3. Get a paid API key for higher limits\n\n"
        "**Free Tier Limits:**\n"
        "- 15 requests per minute\n"
        "- 1,500 requests per day\n"
        "- 1 million tokens per day\n\n"
        "Monitor your usage: https://ai.dev/usage"
    )


# ------------------------------------------------------------
#  HELPER FUNCTIONS FOR QUERY PROCESSING
# ------------------------------------------------------------

def detect_query_type(query: str) -> str:
    """Detect the type of query to apply appropriate strategy."""
    query_lower = query.lower()
    
    if any(kw in query_lower for kw in COMPARISON_KEYWORDS):
        return 'comparison'
    elif any(kw in query_lower for kw in MULTI_ASPECT_KEYWORDS):
        return 'multi_aspect'
    elif any(kw in query_lower for kw in EXPLANATION_KEYWORDS):
        return 'explanation'
    else:
        return 'general'


def extract_key_terms(query: str) -> str:
    """Extract important terms from query, removing common stop words."""
    stop_words = {'what', 'is', 'are', 'the', 'how', 'does', 'do', 'can', 'could', 
                  'would', 'should', 'explain', 'describe', 'tell', 'me', 'about', 
                  'a', 'an', 'of', 'in', 'on', 'at', 'to', 'for'}
    
    # Split and clean
    words = re.findall(r'\b\w+\b', query.lower())
    key_terms = [w for w in words if w not in stop_words and len(w) > 2]
    
    return ' '.join(key_terms)


def create_query_variants(query: str, query_type: str) -> List[str]:
    """
    Create multiple query variants for hybrid retrieval.
    This helps capture different aspects and phrasings.
    """
    variants = [query]  # Always include original
    
    # Add key terms version
    key_terms = extract_key_terms(query)
    if key_terms and key_terms != query:
        variants.append(key_terms)
    
    # Add type-specific variants
    if query_type == 'comparison':
        # For comparisons, also search for individual terms
        parts = re.split(r'\b(?:vs\.?|versus|and|compared to|compared with)\b', query, flags=re.IGNORECASE)
        for part in parts:
            cleaned = part.strip()
            if cleaned and len(cleaned) > 3:
                variants.append(cleaned)
    
    elif query_type == 'multi_aspect':
        # For multi-aspect queries, add a more focused version
        focused = query.replace('all', '').replace('every', '').replace('various', '').strip()
        if focused and focused != query:
            variants.append(focused)
    
    elif query_type == 'explanation':
        # For explanations, add definition-focused variant
        # Extract the main term being asked about
        term_match = re.search(r'(?:what is|what are|explain|describe)\s+(.+?)(?:\?|$)', query, re.IGNORECASE)
        if term_match:
            main_term = term_match.group(1).strip()
            variants.append(f"definition {main_term}")
            variants.append(f"{main_term} explanation")
    
    # Remove duplicates while preserving order
    seen = set()
    unique_variants = []
    for v in variants:
        v_lower = v.lower().strip()
        if v_lower and v_lower not in seen:
            seen.add(v_lower)
            unique_variants.append(v)
    
    return unique_variants[:4]  # Limit to 4 variants to avoid overwhelming


def expand_query_intelligently(query: str, query_type: str) -> str:
    """
    Expand query based on its type to improve retrieval.
    Works generically without assuming specific document topics.
    """
    base_query = query
    
    if query_type == 'comparison':
        expansion = (
            f"{base_query}\n\n"
            f"Include information covering:\n"
            f"- Key differences and similarities\n"
            f"- Strengths and weaknesses of each\n"
            f"- Performance or efficiency comparisons\n"
            f"- Use cases or applications for each\n"
            f"- Any trade-offs between them"
        )
    
    elif query_type == 'multi_aspect':
        expansion = (
            f"{base_query}\n\n"
            f"Provide comprehensive coverage including:\n"
            f"- All relevant types or categories\n"
            f"- Key characteristics of each\n"
            f"- How they relate to each other\n"
            f"- Practical applications or examples"
        )
    
    elif query_type == 'explanation':
        expansion = (
            f"{base_query}\n\n"
            f"Provide detailed explanation covering:\n"
            f"- Core definition and concept\n"
            f"- How it works or functions\n"
            f"- Key components or principles\n"
            f"- Practical examples or applications\n"
            f"- Any important context or background"
        )
    
    else:  # general
        expansion = (
            f"{base_query}\n\n"
            f"Include relevant details such as definitions, examples, "
            f"applications, and any important context."
        )
    
    return expansion


# ------------------------------------------------------------
#  BUILD VECTOR STORE (Multi-PDF)
# ------------------------------------------------------------

def build_vector_store(pdf_paths: List[str]) -> FAISS:
    """Load and combine multiple PDFs into a single FAISS vector store."""
    
    all_chunks = []
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " "]
    )
    
    for pdf_path in pdf_paths:
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
        
        # Add filename metadata for citations
        filename = os.path.basename(pdf_path)
        for doc in docs:
            doc.metadata["filename"] = filename
        
        # Split into chunks
        chunks = splitter.split_documents(docs)
        all_chunks.extend(chunks)
    
    # Build vector DB
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    vector_store = FAISS.from_documents(all_chunks, embeddings)
    
    return vector_store


# ------------------------------------------------------------
#  HYBRID RETRIEVAL STRATEGY
# ------------------------------------------------------------

def hybrid_retrieval(vectorstore: FAISS, query: str, query_type: str) -> List:
    """
    Use multiple retrieval strategies and combine results.
    This significantly improves coverage for complex queries.
    """
    all_docs = []
    seen_content = set()
    
    # Strategy 1: Original query
    docs1 = vectorstore.similarity_search(query, k=TOP_K)
    
    # Strategy 2: Expanded query
    expanded_query = expand_query_intelligently(query, query_type)
    docs2 = vectorstore.similarity_search(expanded_query, k=TOP_K)
    
    # Strategy 3: Query variants (different phrasings)
    query_variants = create_query_variants(query, query_type)
    docs3 = []
    for variant in query_variants:
        if variant != query:  # Don't repeat original
            variant_docs = vectorstore.similarity_search(variant, k=min(TOP_K, 10))
            docs3.extend(variant_docs)
    
    # Strategy 4: Key terms only (broader search)
    key_terms = extract_key_terms(query)
    if key_terms and key_terms != query:
        docs4 = vectorstore.similarity_search(key_terms, k=TOP_K)
    else:
        docs4 = []
    
    # Combine all results
    all_docs = docs1 + docs2 + docs3 + docs4
    
    # Deduplicate while preserving order (earlier results = more relevant)
    unique_docs = []
    for doc in all_docs:
        content_hash = hash(doc.page_content)
        if content_hash not in seen_content:
            seen_content.add(content_hash)
            unique_docs.append(doc)
    
    # Return top FINAL_K documents
    return unique_docs[:FINAL_K]


# ------------------------------------------------------------
#  BUILD CONTEXT FROM RETRIEVED DOCUMENTS
# ------------------------------------------------------------

def build_context_from_docs(docs: List) -> Tuple[str, List[str]]:
    """
    Build formatted context string and source citations from documents.
    """
    context_parts = []
    sources = []
    source_set = set()  # Track unique sources
    
    for i, doc in enumerate(docs, 1):
        fname = doc.metadata.get("filename", "unknown.pdf")
        page = doc.metadata.get("page", None)
        
        # Build source reference
        if page is not None:
            source_ref = f"{fname} (page {page + 1})"
            header = f"[Source {i}: {fname}, page {page + 1}]"
        else:
            source_ref = fname
            header = f"[Source {i}: {fname}]"
        
        # Add to sources list (avoid duplicates)
        if source_ref not in source_set:
            source_set.add(source_ref)
            sources.append(source_ref)
        
        # Add to context
        context_parts.append(f"{header}\n{doc.page_content}")
    
    context = "\n\n" + "="*80 + "\n\n".join(context_parts)
    
    return context, sources


# ------------------------------------------------------------
#  CREATE OPTIMIZED PROMPT BASED ON QUERY TYPE
# ------------------------------------------------------------

def create_prompt(query: str, query_type: str, context: str) -> str:
    """
    Create an optimized prompt based on query type.
    """
    base_instructions = """You are an expert AI assistant analyzing documents. Follow these rules strictly:

1. **Use ONLY the provided context** - Never add information from outside sources
2. **Be accurate and precise** - Quote or paraphrase from the context
3. **Cite sources** - Reference specific documents when making claims
4. **Admit limitations** - If information is not in the context, clearly state this
"""
    
    if query_type == 'comparison':
        specific_instructions = """
5. **Structure comparisons clearly** - Use tables, bullet points, or organized sections
6. **Cover all aspects** - Include similarities, differences, strengths, weaknesses
7. **Be balanced** - Give fair treatment to all items being compared
8. **Highlight trade-offs** - Explain when one is better for specific use cases
"""
    
    elif query_type == 'multi_aspect':
        specific_instructions = """
5. **Be comprehensive** - Cover all types/categories mentioned in the documents
6. **Organize clearly** - Use structured format (numbered lists, sections, etc.)
7. **Explain relationships** - Show how different aspects relate to each other
8. **Provide examples** - Use concrete examples from the documents
"""
    
    elif query_type == 'explanation':
        specific_instructions = """
5. **Start with core definition** - Begin with what it is
6. **Explain how it works** - Describe mechanisms or principles
7. **Provide context** - Explain why it matters or when it's used
8. **Use examples** - Include concrete examples if available in context
"""
    
    else:  # general
        specific_instructions = """
5. **Be clear and direct** - Answer the question straightforwardly
6. **Provide supporting details** - Include relevant context and examples
7. **Stay focused** - Don't include unrelated information
"""
    
    prompt = f"""{base_instructions}{specific_instructions}

CONTEXT FROM DOCUMENTS:
{context}

USER QUESTION:
{query}

DETAILED ANSWER:
"""
    
    return prompt


# ------------------------------------------------------------
#  MAIN QUERY ANSWERING FUNCTION
# ------------------------------------------------------------

def answer_query(vectorstore: FAISS, query: str) -> Tuple[str, List[str]]:
    """
    Main function: Retrieve relevant chunks, build context, and query Gemini.
    Uses intelligent strategies based on query type.
    """
    
    # Detect query type for appropriate handling
    query_type = detect_query_type(query)
    
    # Use hybrid retrieval for better coverage
    docs = hybrid_retrieval(vectorstore, query, query_type)
    
    # Check if we found any relevant documents
    if not docs or len(docs) == 0:
        return (
            f"I couldn't find relevant information about '{query}' in the uploaded PDFs. "
            f"The documents may not contain information on this topic. "
            f"Please try rephrasing your question or ask about topics covered in the uploaded documents.",
            []
        )
    
    # Build context and sources
    context, sources = build_context_from_docs(docs)
    
    # Create optimized prompt based on query type
    prompt = create_prompt(query, query_type, context)
    
    # Query Gemini with automatic retry logic
    try:
        answer = query_gemini_with_retry(prompt, max_retries=3)
    except Exception as e:
        return str(e), sources
    
    # Post-process answer
    answer = answer.strip()
    
    # If answer is too short or generic, it might indicate poor retrieval
    if len(answer) < 100 and "not found" not in answer.lower():
        answer += "\n\n*Note: Limited information was found in the documents. Consider rephrasing your question for better results.*"
    
    return answer, sources