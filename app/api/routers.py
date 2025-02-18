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
async def update_file_data(collection_name: str, url: str, rag: RAG = Depends(get_rag)):
    response = rag.upload_file_data(collection_name, url)
    return {"answer": response}