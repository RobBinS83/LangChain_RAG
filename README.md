# DocAssist

A simple doc-assistant prototype using LangChain, Tavily, OpenAI embeddings, and Pinecone.

This project crawls documentation from `https://python.langchain.com`, splits the text into chunks, indexes them into Pinecone, and provides a retrieval-augmented generation (RAG) backend for answering questions based on the retrieved documentation.

---

## Project Structure

- `pyproject.toml` — package metadata and dependencies
- `ingestion.py` — documentation crawling, chunking, and Pinecone indexing pipeline
- `backend/core.py` — retrieval + agent answer pipeline using LangChain tools
- `logger.py` — simple colored console logging helpers
- `main.py` — basic entrypoint stub
- `backend/__init__.py` — empty package initializer
- `notebooks/` — example/demo notebooks

---

## Requirements

- Python `>=3.13`
- `pip` or Poetry compatible with `pyproject.toml`
- Pinecone account + index
- OpenAI API access

---

## Installation

```bash
uv sync
```

---

## Environment Variables

Create a `.env` file with values like:

```ini
OPENAI_API_KEY=your_openai_api_key
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_ENV=your_pinecone_environment
```

Depending on your Pinecone setup, you may also need:

```ini
PINECONE_PROJECT_NAME=...
```

---

## Usage

### 1. Run ingestion pipeline

```bash
python ingestion.py
```

This script:
- crawls LangChain docs using `TavilyCrawl`
- extracts and converts pages to `Document`
- splits documents into chunks with `RecursiveCharacterTextSplitter`
- indexes chunks into the Pinecone index `langchain-doc-assist`

### 2. Run the backend demo

```bash
python backend/core.py
```

This module:
- initializes `OpenAIEmbeddings`
- initializes `PineconeVectorStore`
- defines `retrieve_context` as a LangChain tool
- creates an agent using `gpt-5.5`
- runs a sample query (`what are deep agents?`)

### 3. Example entrypoint

```bash
python main.py
```

Currently prints:

```text
Hello from docassist!
```

---

## Notes

- The current ingestion pipeline targets `https://python.langchain.com`
- The vector store index name is hard-coded as `langchain-doc-assist`
- The backend uses `gpt-5.5` as the chat model and expects Pinecone to be reachable via env vars
- `logger.py` provides colored terminal output for pipeline progress

---

## Future Improvements

- Add CLI arguments for crawl URL, Pinecone index name, and batch size
- Add a retrieval query interface for interactive chat
- Persist and reuse a local embedding/vector store
- Add tests and notebook documentation for usage examples
