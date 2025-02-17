import uvicorn
from fastapi import FastAPI, APIRouter, Depends
from app.core.ingest import RAG

# Initialize RAG only once globally
rag = RAG()

app = FastAPI(
    title="E-Learning Chatbot API",
    description="API hỗ trợ chatbot e-learning với RAG, ChromaDB và Ollama.",
    version="1.0",
    docs_url="/docs",       # Bật Swagger UI
    redoc_url="/redoc",     # Bật ReDocs
    openapi_url="/openapi.json"  # Bật OpenAPI JSON
)

# Startup event to call ingest_data() once during the app startup
@app.on_event("startup")
async def startup_event():
    rag.ingest_data()

# Function to provide rag as a dependency to all routes
def get_rag() -> RAG:
    return rag

router = APIRouter()

@router.post("/chat")
async def ask_chatbot(question: str, rag: RAG = Depends(get_rag)):
    response = rag.chat(question)
    return {"answer": response}

@router.post("/update_lesson_data")
async def update_lesson_data(lesson_id: str, rag: RAG = Depends(get_rag)):
    response = rag.upload_lesson_data(lesson_id)
    return {"answer": response}

@router.post("/update_file_data")
async def update_file_data(collection_name: str, url: str, rag: RAG = Depends(get_rag)):
    response = rag.upload_file_data(collection_name, url)
    return {"answer": response}

# Include the chat router
app.include_router(router)

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=9000, reload=True)
