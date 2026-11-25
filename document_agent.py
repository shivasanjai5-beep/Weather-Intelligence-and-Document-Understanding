"""Agent 2: answers questions from an uploaded PDF/TXT (simple RAG)."""
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from llm import ask_llm


def read_file(uploaded_file) -> str:
    if uploaded_file.name.lower().endswith(".pdf"):
        reader = PdfReader(uploaded_file)
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    return uploaded_file.read().decode("utf-8", errors="ignore")


def split_into_chunks(text: str, size: int = 500, overlap: int = 100) -> list:
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start:start + size])
        start += size - overlap
    return [c for c in chunks if c.strip()]


class DocumentStore:
    """Holds chunks and finds the most relevant ones for a question."""

    def __init__(self, text: str):
        self.chunks = split_into_chunks(text)
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.matrix = self.vectorizer.fit_transform(self.chunks)

    def search(self, question: str, k: int = 3) -> list:
        scores = cosine_similarity(self.vectorizer.transform([question]), self.matrix)[0]
        best = scores.argsort()[::-1][:k]
        return [self.chunks[i] for i in best if scores[i] > 0]


def answer(question: str, store: DocumentStore) -> dict:
    passages = store.search(question)
    if not passages:
        return {"text": "I could not find this in the uploaded document.", "sources": []}
    context = "\n---\n".join(passages)
    text = ask_llm(
        "Answer using ONLY the document passages below. "
        "If the answer is not there, say 'Not found in the document'. Do not guess.\n\n"
        f"Passages:\n{context}\n\nQuestion: {question}"
    )
    return {"text": text, "sources": passages}
