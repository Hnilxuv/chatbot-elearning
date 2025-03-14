
# RAG Pipeline with LangChain, ChromaDB, and Ollama

An efficient Retrieval-Augmented Generation (RAG) pipeline leveraging LangChain, ChromaDB, and Ollama for building state-of-the-art natural language understanding applications. This repository provides a complete workflow for retrieving and generating contextually relevant responses using modern AI technologies.

## Overview

This project implements a Retrieval-Augmented Generation (RAG) pipeline combining:

- **LangChain**: Orchestrates Large Language Model (LLM) workflows.
- **ChromaDB**: Serves as a fast and efficient vector database for storing and retrieving embeddings.
- **Ollama**: Generates responses using fine-tuned LLMs.

The pipeline is designed for tasks such as question answering, summarization, and other NLP applications where leveraging external knowledge sources is crucial.

## Features

- **Modular Design**: Easily adapt and extend the pipeline for various use cases.
- **Embeddings Storage**: Store and retrieve embeddings using ChromaDB.
- **Contextual Generations**: Retrieve relevant context and pass it to LLMs using LangChain.
- **Seamless Integration**: Pre-integrated with Ollama for generating fine-tuned outputs.
- **Documentation**: Comprehensive setup and usage guide included.

## Prerequisites

- **Python**: Version 3.8 or later.
- **pip**: For package installation.
- **Ollama**: Ensure Ollama is installed and running on your local machine. You can download it from [Ollama's official website](https://ollama.com/download).

## Installation

### Local Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/jamalissa/rag-pipeline-langchain-chromadb-ollama.git
   cd rag-pipeline-langchain-chromadb-ollama
   ```

2. **Set up a virtual environment** (recommended):

   For both the front and back end, simply run the provided `setupvenv.cmd` script:

   ```bash
   ./setupvenv.cmd
   ```

   This script will automatically configure the virtual environment and install necessary dependencies.

3. **Verify the installation**:

   Check if all dependencies are installed and no errors are present by running:

   ```bash
   python -m pip check
   ```

4. **Ensure Ollama is running**:

   Start the Ollama server by running:

   ```bash
   ollama serve
   ```

   By default, Ollama listens on `http://localhost:11434`.

### Docker Installation (using Docker Compose)

1. **Clone the repository**:

   ```bash
   git clone https://github.com/jamalissa/rag-pipeline-langchain-chromadb-ollama.git
   cd rag-pipeline-langchain-chromadb-ollama
   ```

2. **Run the application using Docker Compose**:

   Ensure `docker-compose.yml` is present in the project directory, then execute:

   ```bash
   docker-compose up -d
   ```

   This will build and start all necessary containers.

3. **Verify the containers are running**:

   Ensure the Docker containers are running and accessible by visiting `http://localhost:11434` or using Docker commands:

   ```bash
   docker ps
   ```

4. **Stop the containers when done**:

   To stop the running containers, execute:

   ```bash
   docker-compose down
   ```

## Usage

1. **Ingest your data**:

   Prepare your data in a JSON format and place it in the `data.json` file. Then, run the ingestion script to process and store embeddings in ChromaDB.

   ```bash
   python ingest_data.py
   ```

2. **Run the RAG pipeline**:

   Execute the main script to start the RAG pipeline.

   ```bash
   python rag_pipeline.py
   ```

3. **Interact with the system**:

   Once the pipeline is running, you can input queries, and the system will provide contextually relevant responses based on the ingested data.

## Configuration

- **Ollama Model**: The default model used is `llama3`. You can change this by modifying the model parameter in the `rag_pipeline.py` script.

- **ChromaDB Settings**: Configuration for ChromaDB can be adjusted in the `ingest_data.py` and `rag_pipeline.py` scripts as needed.

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request with your changes. Ensure that your code adheres to the project's coding standards and includes appropriate tests.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [LangChain](https://github.com/hwchase17/langchain)
- [ChromaDB](https://www.trychroma.com/)
- [Ollama](https://ollama.com/)