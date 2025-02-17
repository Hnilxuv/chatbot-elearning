import requests
import tempfile
from langchain_community.document_loaders import PyPDFLoader

def extract_text_from_pdf(pdf_url):
    response = requests.get(pdf_url)
    with tempfile.NamedTemporaryFile(delete=True, suffix=".pdf") as temp_pdf:
        temp_pdf.write(response.content)
        temp_pdf.flush()

        loader = PyPDFLoader(temp_pdf.name)
        documents = loader.load()

    return documents