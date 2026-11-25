"""The 'agentic' part: the LLM decides which tool to use."""
import json
from llm import ask_llm

ROUTER_PROMPT = """You are a router for a chatbot. Pick ONE tool for the user question.

Tools:
- "weather": questions about weather (current, yesterday, tomorrow, forecast, last 10 days).
- "document": questions about the uploaded document / resume (skills, education, projects, experience).
- "web": general knowledge questions that are neither weather nor about the document.

Reply ONLY with JSON like:
{{"tool": "weather", "city": "Chennai", "when": "current"}}

"when" must be one of: current, yesterday, tomorrow, forecast, last_10_days.
"city" and "when" are only needed for the weather tool (use "" otherwise).
A document is {doc_status}.

Question: {question}"""


def route(question: str, has_document: bool) -> dict:
    status = "uploaded" if has_document else "NOT uploaded"
    raw = ask_llm(ROUTER_PROMPT.format(doc_status=status, question=question), json_mode=True)
    try:
        data = json.loads(raw)
        if data.get("tool") not in ("weather", "document", "web"):
            raise ValueError
        return data
    except (json.JSONDecodeError, ValueError):
        return {"tool": "web", "city": "", "when": ""}  # safe default
