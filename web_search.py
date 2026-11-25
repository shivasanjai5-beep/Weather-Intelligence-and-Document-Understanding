"""Web fallback: only for general questions. Always shows sources."""
from duckduckgo_search import DDGS
from llm import ask_llm


def web_answer(question: str) -> dict:
    try:
        results = DDGS().text(question, max_results=3)
    except Exception as e:
        return {"text": f"⚠️ Web search failed: {e}", "sources": []}
    context = "\n".join(f"{r['title']}: {r['body']}" for r in results)
    text = ask_llm(f"Answer briefly using these search results.\n\n{context}\n\nQuestion: {question}")
    return {"text": text, "sources": [r["href"] for r in results]}
