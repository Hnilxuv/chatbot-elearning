import logging
import os
import asyncio

from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_chroma import Chroma
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain_community.vectorstores.utils import filter_complex_metadata
from app.utils.extract import extract_text_from_pdf

CHROMA_DB_PATH = "./chroma_db"
OLLAMA_MODEL = "llama3"
OLLAMA_EMBEDDING = "nomic-embed-text"

# Cấu hình logging để debug
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RAG:
    """A class for handling PDF ingestion and question answering using RAG asynchronously."""

    def __init__(self):
        try:
            self.model = ChatOllama(model=OLLAMA_MODEL)
            self.embeddings = OllamaEmbeddings(model=OLLAMA_EMBEDDING)
            self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=100)
            # self.prompt = ChatPromptTemplate.from_template(
            #     """
            #     Bạn là một trợ lý thông minh, được thiết kế để trả lời câu hỏi dựa trên tài liệu đã được tải lên.
            #     Nếu câu hỏi không liên quan đến tài liệu, hãy lịch sự nhắc nhở người dùng và gợi ý họ đặt câu hỏi liên quan.
            #
            #     **Ngữ cảnh:**
            #     {context}
            #
            #     **Câu hỏi:**
            #     {question}
            #
            #     Hãy cung cấp câu trả lời rõ ràng, chính xác và súc tích với dữ liệu trong tài liệu, không thêm thông tin không cần thiết.
            #     Và hãy trích dẫn nguồn trong tài liệu từ ngữ cảnh và vị trí của thông tin (ví dụ: số trang, tiêu đề phần hoặc chỉ mục).
            #
            #     - Nếu câu hỏi bằng tiếng Việt, hãy trả lời bằng tiếng Việt.
            #     - Nếu câu hỏi bằng tiếng Anh, hãy trả lời bằng tiếng Anh.
            #
            #     """
            # )
            self.prompt = ChatPromptTemplate.from_template(
                """
                You are an intelligent assistant designed to answer questions based on the uploaded document.  
                If the question is unrelated to the document, politely remind the user and suggest they ask a relevant question.  

                **Context:**  
                {context}  

                **Question:**  
                {question}  

                Provide a clear, accurate, and concise answer using the data from the document, without adding unnecessary information.  
                Cite the source from the document, including the context and location of the information (e.g., page number, section title, or index).  

                - If the question is in Vietnamese, respond in Vietnamese.  
                - If the question is in English, respond in English.  
                """
            )

            self.memory = ConversationBufferMemory(memory_key="history", return_messages=True)
            asyncio.run(self._load_vector_store())  # Chạy bất đồng bộ trong init
        except Exception as e:
            logger.error(f"Failed to initialize RAG class: {e}")
            raise

    async def _load_vector_store(self):
        """Tải hoặc khởi tạo ChromaDB từ thư mục lưu trữ."""
        if os.path.exists(CHROMA_DB_PATH):
            logger.info("Loading existing ChromaDB asynchronously...")
            self.vector_store = Chroma(
                collection_name="elearning_collection",
                embedding_function=self.embeddings,
                persist_directory=CHROMA_DB_PATH
            )
        else:
            logger.info("Creating new ChromaDB instance asynchronously...")
            self.vector_store = Chroma(
                collection_name="elearning_collection",
                persist_directory=CHROMA_DB_PATH
            )
        self.retriever = self.vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"k": 5, "score_threshold": 0.5},
        )

    async def _process_and_store_text(self, documents):
        """Processes and stores extracted text into ChromaDB asynchronously."""
        try:
            chunks = await asyncio.to_thread(self.text_splitter.split_documents, documents)
            chunks = await asyncio.to_thread(filter_complex_metadata, chunks)

            if chunks:
                logger.info("Adding new documents to ChromaDB asynchronously...")
                await asyncio.to_thread(self.vector_store.add_documents, documents=chunks, embeddings=self.embeddings)
                logger.info("Documents successfully added to ChromaDB.")
                return "Dữ liệu đã được xử lý và lưu vào ChromaDB!"
            else:
                return "Không có dữ liệu mới để lưu vào ChromaDB."
        except Exception as e:
            logger.error(f"Error processing and storing text: {e}")
            return f"Error: {e}"

    async def chat(self, query):
        """Answers a query using the RAG pipeline asynchronously."""
        try:
            logger.info(f"Retrieving context for query: {query}")
            retrieved_docs = await asyncio.to_thread(self.retriever.invoke, query)

            if not retrieved_docs:
                return "No relevant context found in the document to answer your question."

            history = self.memory.load_memory_variables({})["history"]
            formatted_input = {
                "history": history,
                "context": "\n\n".join(doc.page_content for doc in retrieved_docs),
                "question": query,
            }

            chain = RunnablePassthrough() | self.prompt | self.model | StrOutputParser()
            logger.info("Generating response using the LLM asynchronously.")
            response = await asyncio.to_thread(chain.invoke, formatted_input)
            return response
        except Exception as e:
            logger.error(f"Error during chat processing: {e}")
            return f"Error: {e}"

    async def upload_file_data(self, file_url, file_type):
        """Uploads data from a file to a specific collection in ChromaDB asynchronously."""
        try:
            if file_type != "file":
                return "Dữ liệu không phải file data"
            text = await asyncio.to_thread(extract_text_from_pdf, file_url)
            return await self._process_and_store_text(text)
        except Exception as e:
            logger.error(f"Error during upload data processing: {e}")
            return f"Error: {e}"
