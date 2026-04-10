from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from backend.rag_system import SimpleRAG

print("🚀 Backend starting...")

app = FastAPI()

@app.on_event("startup")
async def startup_event():
    global rag
    print("🔥 FastAPI started successfully")
    rag = SimpleRAG()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class QuestionRequest(BaseModel):
    question: str


@app.get("/")
def root():
    return {
        "status": "RAG backend running",
        "indexed_chunks": len(rag.documents) if rag else 0
    }


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    content = await file.read()
    text = content.decode("utf-8")

    chunks = rag.add_text(text)

    return {"message": "Uploaded", "chunks": chunks}


@app.post("/ask")
def ask(body: QuestionRequest):
    if not rag:
        raise HTTPException(status_code=500, detail="RAG not initialized")

    answer = rag.ask(body.question)
    return {"answer": answer}