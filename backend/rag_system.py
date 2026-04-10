import numpy as np

class SimpleRAG:
    def __init__(self):
        self.documents = []
        self.vectors = []

    def _chunk_text(self, text, chunk_size=100):
        words = text.split()
        chunks = []

        for i in range(0, len(words), chunk_size):
            chunk = " ".join(words[i:i + chunk_size])
            if len(chunk.strip()) > 20:
                chunks.append(chunk)

        return chunks

    def _embed(self, texts):
        # simple numeric embedding (length-based)
        return [np.array([len(t)]) for t in texts]

    def add_text(self, text):
        chunks = self._chunk_text(text)
        vectors = self._embed(chunks)

        self.documents.extend(chunks)
        self.vectors.extend(vectors)

        return len(self.documents)

    def ask(self, question):
        if not self.documents:
            return "No document uploaded."

        q_vec = np.array([len(question)])

        sims = []
        for vec in self.vectors:
            sim = np.dot(q_vec, vec) / (np.linalg.norm(q_vec) * np.linalg.norm(vec))
            sims.append(sim)

        best_idx = int(np.argmax(sims))
        context = self.documents[best_idx]

        return f"Answer based on document:\n\n{context[:300]}..."