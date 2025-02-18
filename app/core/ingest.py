import logging

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
    """A class for handling PDF ingestion and question answering using RAG."""

    def __init__(self):
        self.model = ChatOllama(model=OLLAMA_MODEL)
        self.embeddings = OllamaEmbeddings(model=OLLAMA_EMBEDDING)
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=100)
        self.prompt = ChatPromptTemplate.from_template(
            """
            You are a helpful assistant developed to answer questions based on the uploaded documents.
            If the question is not related to the document, tell the user that the question is out of scope.

            Context:
            {context}

            Question:
            {question}

            You only give precise and detailed answer to the question, no unnecessary information.
            If the question is unrelated to the course, respond with:
            "Your question is not related to the course content. Please ask a relevant question."
            If the question is in Vietnamese, answer in Vietnamese. If the question is in English, answer in English.
            """
        )
        self.memory = ConversationBufferMemory(memory_key="history", return_messages=True)
        self.vector_store = None
        self.retriever = None

    def _process_and_store_text(self, collection_name, documents):
        """Processes and stores extracted text into ChromaDB."""
        chunks = self.text_splitter.split_documents(documents)
        chunks = filter_complex_metadata(chunks)
        if not self.vector_store:
            self.vector_store = Chroma.from_documents(
                collection_name=collection_name,
                documents=chunks,
                embedding=self.embeddings,
                persist_directory=CHROMA_DB_PATH,
                collection_metadata={"hnsw:space": "cosine"}
            )
        else:
            # Nếu vector store đã có sẵn, chỉ cần thêm các chunk mới vào
            self.vector_store.add_documents(documents=chunks, embeddings=self.embeddings)
        logging.info(f"Dữ liệu đã được xử lý và lưu vào ChromaDB: {collection_name}")
        return "Dữ liệu đã được xử lý và lưu vào ChromaDB!"


    def chat(self, query):
        """Answers a query using the RAG pipeline."""
        if not self.retriever:
            self.retriever = self.vector_store.as_retriever(
                search_type="similarity_score_threshold",
                search_kwargs={"k": 5, "score_threshold": 0.2},
            )
        # collection_name = f"course_{course_id}"
        logger.info(f"Retrieving context for query: {query}")
        # docs = self.vector_store.similarity_search(query)

        retrieved_docs = self.retriever.invoke(query)
        #
        if not retrieved_docs:
            return "No relevant context found in the document to answer your question."
        history = self.memory.load_memory_variables({})["history"]
        formatted_input = {
            "history": history,
            "context": "\n\n".join(doc.page_content for doc in retrieved_docs),
            "question": query,
        }
        chain = RunnablePassthrough() | self.prompt | self.model | StrOutputParser()
        logger.info("Generating response using the LLM.")
        return chain.invoke(formatted_input)


    def upload_file_data(self,collection_name, video_url):
        """Uploads data from a file to a specific collection in ChromaDB."""
        text = extract_text_from_pdf(video_url)
        return self._process_and_store_text(collection_name, text)
