import numpy as np
import os
import requests

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

    import os
import requests

HF_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

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

            if result and isinstance(result[0], float):
                return [result]

            return result

        except Exception as e:
            print("HF embedding failed:", e)

    # fallback
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
    q_vec = np.array(self._embed([question])[0])

    sims = []
    for vec in self.vectors:
        vec = np.array(vec)
        sim = np.dot(q_vec, vec) / (np.linalg.norm(q_vec) * np.linalg.norm(vec))
        sims.append(sim)

    # top 3 chunks
    top_indices = np.argsort(sims)[-3:][::-1]
    context = "\n\n".join([self.documents[i] for i in top_indices])

    return self._llm_answer(question, context)

HF_LLM_MODEL = "HuggingFaceH4/zephyr-7b-beta"

def _llm_answer(self, question, context):
    hf_token = os.getenv("HF_TOKEN")

    print("LLM function called")           # 👈 ADD HERE
    print("HF TOKEN:", hf_token)           # 👈 ADD HERE

    if not hf_token:
        return "⚠️ LLM NOT USED — showing context:\n\n" + context[:300]
    prompt = f"""
Answer the question using the context below.

Context:
{context}

Question:
{question}

Answer clearly and briefly:
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