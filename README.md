# RAG Assist Agent

A Retrieval-Augmented Generation (RAG) backend API designed to seamlessly integrate with **OpenWebUI**. This system utilizes **Gemini Embeddings**, **Qdrant Vector Database**, and **LLM Reranking (OpenRouter/Qwen)** to provide accurate, context-aware answers based on your documents, including rich support for parsing charts, tables, and correctly resolving images from PDFs.

## System Architecture

1. **Document Parsing (Docling + RapidOCR):** Generates Markdown & JSON with spatial preservation of tables and high-resolution images.
2. **Semantic Chunking:** Context-aware slicing of documents up to 512 tokens using parent-child relationship tracking.
3. **Embedding:** `gemini-embedding-001` via `google-genai` with MRL truncation.
4. **Vector Retrieval:** In-Memory or Cloud Qdrant indexing & similarity search.
5. **Reranking:** LLM-based reranking for semantic refinement (using OpenRouter).
6. **API Layer:** A drop-in FastAPI proxy that mocks OpenAI's `/v1/chat/completions` endpoint for 100% compatibility with OpenWebUI.

## Quick Start

### 1. Requirements
- Python 3.10+
- `pip install -r requirements.txt`

### 2. Environment Configuration
Copy the `.env.example` file and fill in your keys:
```bash
cp .env.example .env
```
_Note: If using OpenWebUI via a mobile device or external IP, set `BASE_IMAGE_URL` (e.g. `http://YOUR_IP:8001`) in your `.env` to correctly resolve local images._

### 3. Run the API Server
Start the Uvicorn FastAPI server:
```bash
cd src
uvicorn api:app --host 0.0.0.0 --port 8001
```

### 4. Connect OpenWebUI
1. Go to **Settings > Connections** in OpenWebUI.
2. Add a new OpenAI connection with the URL: `http://host.docker.internal:8001/v1` (or your machine's IP if bridging fails, e.g. `http://192.168.1.x:8001/v1`).
3. Set any dummy API key (e.g., `sk-12345`).
4. Select the `rag-assist-agent` model from the workspace and start chatting!

## Documentation
- Refer to the `docs/` folder for deeper implementation details on chunking, PDF layout analysis, and cloud configuration.
