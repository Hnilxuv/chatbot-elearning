from fastapi import APIRouter, Depends
from app.core.ingest import RAG

rag = RAG()
def get_rag() -> RAG:
    return rag
router = APIRouter()

@router.post("/chat")
async def ask_chatbot(question: str, rag: RAG = Depends(get_rag)):
    response = rag.chat(question)
    return {"answer": response}

@router.post("/update_file_data")
async def update_file_data(file_url: str, file_type: str, rag: RAG = Depends(get_rag)):
    response = rag.upload_file_data(file_url, file_type)
    return {"answer": response}