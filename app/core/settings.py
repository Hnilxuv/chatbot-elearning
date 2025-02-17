import os
from dotenv import load_dotenv

# Load biến môi trường từ file .env
load_dotenv()

DATABASE_CONFIG = {
    "host": os.getenv("MYSQL_HOST"),
    "user": os.getenv("MYSQL_USER"),
    "password": os.getenv("MYSQL_PASSWORD"),
    "database": os.getenv("MYSQL_DB"),
}

CHROMA_DB_PATH = "./chroma_db"

OLLAMA_MODEL = "llama3"
OLLAMA_EMBEDDING = "nomic-embed-text"
