import os
from dotenv import load_dotenv

def load_api_key():
    """Load Gemini API key from .env or environment variables."""
    if os.path.exists(".env"):
        load_dotenv()

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("❌ GEMINI_API_KEY not found! Add it in .env or env vars.")
    return api_key


def format_citations(source_docs):
    """Format citations from retrieved LangChain Documents."""
    if not source_docs:
        return ""

    citations = []
    for doc in source_docs:
        meta = doc.metadata
        filename = os.path.basename(meta.get("source", ""))
        page = meta.get("page", None)

        if page is not None:
            citations.append(f"- **{filename}** → Page **{page + 1}**")

    if citations:
        return "\n\n### 📚 Sources\n" + "\n".join(citations)
    return ""
