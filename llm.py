"""Talks to a local open-source LLM served by Ollama."""
import os
import requests
from dotenv import load_dotenv

load_dotenv()
MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
OLLAMA_URL = "http://localhost:11434/api/chat"


def ask_llm(prompt: str, json_mode: bool = False) -> str:
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
    }
    if json_mode:
        body["format"] = "json"
    try:
        r = requests.post(OLLAMA_URL, json=body, timeout=120)
        r.raise_for_status()
        return r.json()["message"]["content"]
    except requests.RequestException as e:
        return f"LLM error: is Ollama running? ({e})"
