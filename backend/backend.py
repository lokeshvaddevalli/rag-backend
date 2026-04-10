from fastapi import FastAPI

print("🚀 Backend starting...")

app = FastAPI()

@app.on_event("startup")
async def startup_event():
    print("🔥 FastAPI started successfully")

@app.get("/")
def root():
    return {"status": "working"}