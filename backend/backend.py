from fastapi import FastAPI
from pydantic import BaseModel
from backend.rag_system import SimpleRAG

app = FastAPI(
    title="Simple RAG Backend",
    description="RAG API powered by SentenceTransformers",
    version="1.0.0",
)

# Initialize RAG system
rag = SimpleRAG()
rag.load_sample()


class AskRequest(BaseModel):
    question: str


class AskResponse(BaseModel):
    answer: str


@app.get("/")
def root():
    return {"status": "RAG backend running"}


@app.post("/ask", response_model=AskResponse)
def ask_question(req: AskRequest):
    answer = rag.ask(req.question)
    return {"answer": answer}
