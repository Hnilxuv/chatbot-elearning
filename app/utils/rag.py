# rag.py
from langchain_core.globals import set_verbose, set_debug
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.schema.output_parser import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.runnable import RunnablePassthrough
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_core.prompts import ChatPromptTemplate
from langchain_chroma import Chroma
import logging
import os
import shutil
import streamlit as st
import requests
import subprocess
import re

set_debug(True)
set_verbose(True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
ollama_endpoint = "http://localhost:11434"


class ChatPDF:
    """A class for handling PDF ingestion and question answering using RAG."""

    def __init__(self, is_local: bool = True, api_key: str = "", llm_model: str = "", embedding_model: str = "", dimensions: int = 1024):
        """
        Initialize the ChatPDF instance with an LLM and embedding model.
        """
        self.llm_model = llm_model
        self.embedding_model = embedding_model
        self.is_local = is_local

        if self.is_local:
            self.pull_model()
            self.model = ChatOllama(model=llm_model)
            self.embeddings = OllamaEmbeddings(model=embedding_model) 
        else:
            if "gpt" in llm_model:
                self.model = ChatOpenAI(model=llm_model)
                self.embeddings = OpenAIEmbeddings(model=embedding_model)
            else:
                self.model = ChatGoogleGenerativeAI(model=llm_model)
                self.embeddings = GoogleGenerativeAIEmbeddings(model="models/"+embedding_model)

        self.chunk_size = dimensions
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=self.chunk_size, chunk_overlap=100)

        self.prompt = ChatPromptTemplate.from_template(
            """
            You are a helpful assistant developed to answer questions only based on the uploaded document. If the question is not related to the document, tell the user that the question is out of scope.
            Context:
            {context}
            
            Question:
            {question}
            
            You only give precise and detailed answer to the question, no yapping. And when giving answer, please start with "Answer:".
            """
        )
        self.vector_store = None
        self.retriever = None

    def ingest(self, pdf_file_path: str):
        """
        Ingest a PDF file, split its contents, and store the embeddings in the vector store.
        """

        logger.info(f"Starting ingestion for file: {pdf_file_path}")
        docs = PyPDFLoader(file_path=pdf_file_path).load()
        chunks = self.text_splitter.split_documents(docs)
        

        chunks = filter_complex_metadata(chunks)

        self.vector_store = Chroma.from_documents(
            collection_name = f"pdf_collection_{re.sub(r'[^a-zA-Z0-9._-]', '', self.llm_model)}",
            documents=chunks,
            embedding=self.embeddings,
            persist_directory = f"chroma_db_{self.llm_model}",
            collection_metadata={"hnsw:space": "cosine"}
        )
        
        logger.info("Ingestion completed. Document embeddings stored successfully.")

    def ask(self, query: str, k: int = 5, score_threshold: float = 0.2):
        """
        Answer a query using the RAG pipeline.
        """
        if not self.vector_store:
            raise ValueError("Please upload a document before asking a question.")

        if not self.retriever:
            self.retriever = self.vector_store.as_retriever(
                search_type="similarity_score_threshold",
                search_kwargs={"k": k, "score_threshold": score_threshold},
            )

        logger.info(f"Retrieving context for query: {query}")
        retrieved_docs = self.retriever.invoke(query)
        print(f"retrieved_docs{retrieved_docs}")
        if not retrieved_docs:
            return "No relevant context found in the document to answer your question."

        formatted_input = {
            "context": "\n\n".join(doc.page_content for doc in retrieved_docs),
            "question": query,
        }

        # Build the RAG chain
        chain = (
            RunnablePassthrough()  # Passes the input as-is
            | self.prompt           # Formats the input for the LLM
            | self.model            # Queries the LLM
            | StrOutputParser()     # Parses the LLM's output
        )

        logger.info("Generating response using the LLM.")
        # return chain.invoke(formatted_input)
        return chain.stream(formatted_input)


    def clear(self):
        """
        Reset the vector store and retriever.
        """
        logger.info("Clearing vector store and retriever.")
        self.vector_store = None
        self.retriever = None

    def pull_model(self):
        """Pull the specified model from the Ollama server."""

        with st.spinner(f"Pulling model {self.llm_model}..."):
            response_llm = requests.post(
                f"{ollama_endpoint}/api/pull",
                json={"model": self.llm_model}
            )
        
        if response_llm.status_code != 200:
            st.error(f"Failed to pull model {self.llm_model}: {response_llm.text}")
            raise Exception(f"Model pull failed: {response_llm.text}")
        else:
            st.sidebar.success(f"Model {self.llm_model} pulled successfully.")

        with st.spinner(f"Pulling embedding model {self.embedding_model}..."):
            os.system(f"ollama pull {self.embedding_model}")

            result = subprocess.run(
                ["ollama", "pull", self.embedding_model],
                capture_output=True,
                text=True
            )

        if result.returncode == 0:
            st.sidebar.success(f"Embedding model {self.embedding_model} pulled successfully.")
        else:
            st.error(f"Failed to pull embedding model {self.embedding_model}: {result.stderr}")
            raise Exception(f"Embedding model pull failed: {result.stderr}")
        