from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from app.core.ingest import RAG

rag = RAG()

async def get_rag() -> RAG:
    return rag

router = APIRouter()

# Định nghĩa Pydantic models cho request body
class ChatRequest(BaseModel):
    question: str

class FileUploadRequest(BaseModel):
    file_url: str
    file_type: str

@router.post("/chat")
async def ask_chatbot(request: ChatRequest, rag: RAG = Depends(get_rag)):
    response = await rag.chat(request.question)  # Gọi async
    return {"answer": response}

@router.post("/update_file_data")
async def update_file_data(request: FileUploadRequest, rag: RAG = Depends(get_rag)):
    response = await rag.upload_file_data(request.file_url, request.file_type)  # Gọi async
    return {"answer": response}
