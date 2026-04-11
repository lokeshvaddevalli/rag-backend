import numpy as np
import os
import requests
import re

HF_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
HF_LLM_MODEL = "HuggingFaceH4/zephyr-7b-beta"


class SimpleRAG:
    def __init__(self):
        self.documents = []
        self.vectors = []

    # ---------- Better Chunking ----------
    def _chunk_text(self, text, chunk_size=120, overlap=30):
        words = text.split()
        chunks = []

        for i in range(0, len(words), chunk_size - overlap):
            chunk = " ".join(words[i:i + chunk_size])
            if len(chunk.strip()) > 40:
                chunks.append(chunk)

        return chunks

    # ---------- Embedding ----------
    def _embed(self, texts):
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

                return [np.array(v) for v in result]

            except Exception as e:
                print("HF embedding failed:", e)

        # fallback (consistent dimension)
        return [np.ones(384) * len(t) for t in texts]

    # ---------- Keyword Scoring ----------
    def _keyword_score(self, question, text):
        q_words = set(re.findall(r"\w+", question.lower()))
        t_words = set(re.findall(r"\w+", text.lower()))

        return len(q_words & t_words)

    # ---------- Add Document ----------
    def add_text(self, text):
        chunks = self._chunk_text(text)
        vectors = self._embed(chunks)

        self.documents.extend(chunks)
        self.vectors.extend(vectors)

        return len(self.documents)

    # ---------- Ask ----------
    def ask(self, question):
        if not self.documents:
            return "No document uploaded."

        q_vec = np.array(self._embed([question])[0])
        q_vec = q_vec / (np.linalg.norm(q_vec) + 1e-8)

        scores = []

        for i, vec in enumerate(self.vectors):
            vec = np.array(vec)
            vec = vec / (np.linalg.norm(vec) + 1e-8)

            # cosine similarity
            sim = np.dot(q_vec, vec)

            # keyword score
            kw = self._keyword_score(question, self.documents[i])

            # hybrid score (important)
            final_score = sim + (0.2 * kw)

            scores.append(final_score)

        # top 5 chunks
        top_indices = np.argsort(scores)[-5:][::-1]
        context = "\n\n".join([self.documents[i] for i in top_indices])

        return self._llm_answer(question, context)

    # ---------- LLM ----------
    def _llm_answer(self, question, context):
        hf_token = os.getenv("HF_TOKEN")

        print("LLM function called")
        print("HF TOKEN:", hf_token)

        if not hf_token:
            return "⚠️ LLM NOT USED — showing context:\n\n" + context[:300]

        prompt = f"""
You are a helpful AI assistant.

Answer ONLY using the context below.
If the answer is not present, say "Not found in document."

Context:
{context}

Question:
{question}

Answer clearly in 2-3 sentences:
"""

        try:
            response = requests.post(
                f"https://api-inference.huggingface.co/models/{HF_LLM_MODEL}",
                headers={"Authorization": f"Bearer {hf_token}"},
                json={
                    "inputs": prompt,
                    "parameters": {
                        "max_new_tokens": 150,
                        "return_full_text": False
                    }
                },
                timeout=20,
            )
            response.raise_for_status()
            result = response.json()

            if isinstance(result, list):
                return result[0]["generated_text"]

            return str(result)

        except Exception as e:
            print("LLM failed:", e)
            return context[:300] + "..."