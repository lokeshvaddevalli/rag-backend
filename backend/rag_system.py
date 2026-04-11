import numpy as np
import os
import requests
import re
import httpx
print(" NEW RAG CODE LOADED")
HF_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
HF_LLM_MODEL = "google/flan-t5-base"


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
    
    def _llm_answer(self, question, context):
        api_key = os.getenv("GROQ_API_KEY")
        print("🔥 GROQ FUNCTION CALLED")
        print("🔑 API KEY:", api_key)
        if not api_key:
            return "❌ GROQ API KEY NOT FOUND"
        prompt = f"""
        Use ONLY the context to answer.
        Context:
        {context}

        Question:
        {question}
        Answer:
        """
        try:
            response = httpx.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}"},
                json={
                    "model": "llama3-8b-8192",
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.1,
                    "max_tokens": 200
                    },
                    timeout=15
                    )
            print("📡 GROQ STATUS:", response.status_code)
            print("📡 GROQ RESPONSE:", response.text)
            return response.json()["choices"][0]["message"]["content"]
        except Exception as e:
            print("❌ GROQ ERROR:", e)
            return "❌ GROQ FAILED"