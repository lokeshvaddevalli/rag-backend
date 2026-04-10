from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from backend.rag_system import SimpleRAG

print("🚀 Backend starting...")

app = FastAPI(title="RAG Document Q&A API")

@app.on_event("startup")
async def startup_event():
    global rag
    print("🔥 FastAPI started successfully")
    rag = None

# Allow browser-based front-ends to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Single shared RAG instance for the process lifetime
rag = None


class QuestionRequest(BaseModel):
    question: str


# ──────────────────────────────────────────────
# Health check
# ──────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "status": "RAG backend running",
        "indexed_chunks": len(rag.documents) if rag and hasattr(rag, "documents") else 0,
    }

# ──────────────────────────────────────────────
# Upload endpoint
# ──────────────────────────────────────────────
@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """
    Upload a plain-text (.txt) file and index it.
    Multiple uploads accumulate — they do NOT wipe previous documents.
    """
    if not file.filename.endswith(".txt"):
        raise HTTPException(status_code=400, detail="Only .txt files are supported.")

    try:
        content = await file.read()
        text = content.decode("utf-8")
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="File is not valid UTF-8 text.")

    if not text.strip():
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    total_chunks = 0
    return {
        "message": f"'{file.filename}' uploaded and indexed.",
        "total_indexed_chunks": total_chunks,
    }


# ──────────────────────────────────────────────
# Clear index
# ──────────────────────────────────────────────
@app.delete("/index")
def clear_index():
    """Remove all indexed documents."""
    rag.clear()
    return {"message": "Index cleared."}


# ──────────────────────────────────────────────
# Ask endpoint
# ──────────────────────────────────────────────
@app.post("/ask")
def ask(body: QuestionRequest):
    """
    Ask a question about the uploaded documents.
    Accepts a JSON body: { "question": "..." }
    """
    question = body.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question must not be empty.")

    try:
        if rag is None:
            raise HTTPException(status_code=500, detail="RAG not initialized")
        answer = "Backend is working (RAG temporarily disabled)"
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {e}")

    return {"answer": answer}
