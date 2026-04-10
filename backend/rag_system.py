import os
import requests
import numpy as np
from sentence_transformers import SentenceTransformer

# Free-tier compatible LLM — publicly hosted on HF Inference API
HF_LLM_MODEL = "google/flan-t5-base"
HF_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

TOP_K = 3                   # chunks to combine as context
MIN_CHUNK_WORDS = 20        # discard near-empty trailing chunks
SIMILARITY_THRESHOLD = 0.30 # minimum cosine score to consider a chunk relevant


class SimpleRAG:
    def __init__(self):
        self.local_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.documents: list[str] = []
        self.vectors: list[list[float]] = []

    # ──────────────────────────────────────────────
    # Chunking
    # ──────────────────────────────────────────────
    def _chunk_text(self, text: str, chunk_size: int = 300, overlap: int = 50) -> list[str]:
        """Split text into overlapping word-level chunks."""
        words = text.split()
        step = chunk_size - overlap          # advance by 250 words each time
        chunks = []
        for i in range(0, len(words), step):
            chunk = " ".join(words[i : i + chunk_size])
            if len(chunk.split()) >= MIN_CHUNK_WORDS:   # drop tiny trailing shards
                chunks.append(chunk)
        return chunks

    # ──────────────────────────────────────────────
    # Embeddings  (HF API → local fallback)
    # ──────────────────────────────────────────────
    def _embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts. Uses HF API when token is present, else local model."""
        hf_token = os.getenv("HF_TOKEN")
        if hf_token:
            try:
                response = requests.post(
                    f"https://api-inference.huggingface.co/pipeline/feature-extraction/{HF_EMBED_MODEL}",
                    headers={"Authorization": f"Bearer {hf_token}"},
                    json={"inputs": texts},
                    timeout=15,
                )
                response.raise_for_status()
                result = response.json()
                # HF returns a flat vector for a single input — normalise to list-of-vectors
                if result and isinstance(result[0], float):
                    return [result]
                return result
            except Exception as e:
                print(f"[embed] HF API failed, using local model: {e}")

        return self.local_model.encode(texts, show_progress_bar=False).tolist()

    # ──────────────────────────────────────────────
    # Indexing
    # ──────────────────────────────────────────────
    def add_text(self, text: str) -> int:
        """
        Chunk, embed, and APPEND text to the index.
        Calling this multiple times accumulates documents (no wipe).
        Returns the total number of indexed chunks.
        """
        new_chunks = self._chunk_text(text)
        if not new_chunks:
            return len(self.documents)
        new_vectors = self._embed(new_chunks)
        self.documents.extend(new_chunks)
        self.vectors.extend(new_vectors)
        print(f"[index] +{len(new_chunks)} chunks | total: {len(self.documents)}")
        return len(self.documents)

    def clear(self):
        """Reset the index."""
        self.documents.clear()
        self.vectors.clear()

    # ──────────────────────────────────────────────
    # Retrieval  (vectorised top-k cosine similarity)
    # ──────────────────────────────────────────────
    def _retrieve(self, question: str, top_k: int = TOP_K) -> list[str]:
        """Return the top-k most relevant chunks for the question."""
        if not self.documents:
            return []

        q_vec = np.array(self._embed([question])[0])
        matrix = np.array(self.vectors)                      # (N, D)
        norms = np.linalg.norm(matrix, axis=1) * np.linalg.norm(q_vec)
        norms = np.where(norms == 0, 1e-10, norms)           # avoid /0
        sims = matrix.dot(q_vec) / norms

        k = min(top_k, len(self.documents))
        top_indices = np.argsort(sims)[::-1][:k]

        return [self.documents[i] for i in top_indices if sims[i] >= SIMILARITY_THRESHOLD]

    # ──────────────────────────────────────────────
    # Fallback answer  (extractive keyword scoring)
    # ──────────────────────────────────────────────
    def _fallback_answer(self, question: str, context: str) -> str:
        """
        When the LLM is unavailable, extract the 2 most relevant sentences
        from retrieved context using simple keyword-overlap scoring.
        """
        question_words = set(question.lower().split())
        sentences = [s.strip() for s in context.replace("\n", " ").split(".") if s.strip()]

        # Sort FIRST, then threshold check — fixes original bug
        scored = sorted(
            [(len(question_words & set(s.lower().split())), s) for s in sentences],
            reverse=True,
        )

        if not scored or scored[0][0] < 2:
            return "The document does not contain information relevant to this question."

        top_sentences = [s for score, s in scored[:2] if score > 0]
        return ". ".join(top_sentences).strip() + "."

    # ──────────────────────────────────────────────
    # LLM generation  (HF Inference API → fallback)
    # ──────────────────────────────────────────────
    def _generate(self, question: str, context: str) -> str:
        """
        Try the HF Inference API first.
        Falls back to extractive answer on any failure (410, 503, timeout, no token).
        """
        hf_token = os.getenv("HF_TOKEN")
        if hf_token:
            # Zephyr uses ChatML-style tags
            prompt = f"""Answer the question using ONLY the context below.
            If the answer is not present, say:
            "The document does not contain this information."
            Context:
            {context}
            Question:
            {question}
            Answer:
            """
            try:
                response = requests.post(
                    f"https://api-inference.huggingface.co/models/{HF_LLM_MODEL}",
                    headers={"Authorization": f"Bearer {hf_token}"},
                    json={
                        "inputs": prompt,
                        "parameters": {
                            "max_new_tokens": 200,
                            "temperature": 0.1,
                            "do_sample": False,
                            "return_full_text": False,  # generated part only
                        },
                    },
                    timeout=30,
                )
                response.raise_for_status()
                data = response.json()

                # Normalise list vs dict response shapes
                if isinstance(data, list) and data:
                    generated = data[0].get("generated_text", "").strip()
                elif isinstance(data, dict):
                    generated = data.get("generated_text", "").strip()
                else:
                    generated = ""

                if generated:
                    return generated

            except requests.exceptions.HTTPError as e:
                status = e.response.status_code if e.response is not None else "?"
                print(f"[generate] HF HTTP {status} — switching to fallback")
            except Exception as e:
                print(f"[generate] HF API error — switching to fallback: {e}")

        return self._fallback_answer(question, context)

    # ──────────────────────────────────────────────
    # Public entry-point
    # ──────────────────────────────────────────────
    def ask(self, question: str) -> str:
        """Main Q&A method. Always returns a plain string."""
        if not self.documents:
            return "No document has been uploaded yet. Please upload a text file first."

        chunks = self._retrieve(question)
        if not chunks:
            return "The document does not contain information relevant to this question."

        context = "\n\n---\n\n".join(chunks)   # join top-k chunks as one context block
        return self._generate(question, context)