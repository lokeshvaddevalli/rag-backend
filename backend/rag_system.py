import re
from typing import List, Tuple
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer


@dataclass
class Document:
    content: str
    metadata: dict = None


class SimpleRAG:
    def __init__(self):
        self.documents: List[Document] = []
        self.vectors = []
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def add_text(self, text: str, source: str = "manual"):
        chunks = [p.strip() for p in text.split("\n\n") if p.strip()]

        for i, chunk in enumerate(chunks):
            if len(chunk) > 50:
                self.documents.append(
                    Document(chunk, {"source": source, "chunk": i})
                )

        if self.documents:
            texts = [doc.content for doc in self.documents]
            self.vectors = self.model.encode(texts)

    def load_sample(self):
        sample = """
        TechCorp AI Company Information

        About:
        TechCorp AI is a leading AI company founded in 2020. 
        It provides AI chatbots, analytics, and ML consulting.

        Services:
        - Conversational AI
        - Predictive Analytics
        - Computer Vision
        - MLOps and AutoML

        Pricing:
        Starter: $499/month
        Professional: $1,999/month
        Enterprise: $9,999/month

        Contact:
        Email: hello@techcorp-ai.com
        Phone: +1-555-TECH-AI
        """
        self.add_text(sample, "sample_data")

    def search(self, query: str, k: int = 3) -> List[Tuple[Document, float]]:
        query_vec = self.model.encode([query])[0]

        scores = []
        for i, vec in enumerate(self.vectors):
            score = float(sum(a * b for a, b in zip(query_vec, vec)))
            scores.append((score, i))

        scores.sort(reverse=True)
        return [(self.documents[i], score) for score, i in scores[:k]]

    def simple_answer(self, query: str, context: str) -> str:
        query_words = query.lower().split()
        sentences = re.split(r"[.!?]+", context)

        relevant = []
        for sentence in sentences:
            s = sentence.strip()
            if len(s) > 20 and any(w in s.lower() for w in query_words):
                relevant.append(s)

        if relevant:
            return ". ".join(relevant[:2]) + "."

        return context[:250]

    def ask(self, question: str) -> str:
        if not self.documents:
            return "No documents loaded."

        results = self.search(question, k=3)
        if not results:
            return "No relevant information found."

        context = " ".join([doc.content for doc, _ in results])
        return self.simple_answer(question, context)
