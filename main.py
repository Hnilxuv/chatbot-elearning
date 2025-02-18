import uvicorn
from fastapi import FastAPI
from app.core.ingest import RAG
from app.api.routers import router

app = FastAPI(
    title="E-Learning Chatbot API",
    description="API hỗ trợ chatbot e-learning với RAG, ChromaDB và Ollama.",
    version="1.0",
    docs_url="/docs",  # Bật Swagger UI
    redoc_url="/redoc",  # Bật ReDocs
    openapi_url="/openapi.json"  # Bật OpenAPI JSON
)

app.include_router(router)

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=9000, reload=True)
