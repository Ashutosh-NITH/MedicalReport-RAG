# RAG Model — Retrieval Augmented Generation API

A FastAPI-based RAG (Retrieval Augmented Generation) system that accepts PDF or image uploads, retrieves relevant context from a ChromaDB vector store, and streams AI-generated summaries and suggestions using Groq's LLaMA 3.3 70B model.

---

## Tech Stack

| Layer | Technology |
|---|---|
| API Framework | FastAPI |
| LLM | Groq (`llama-3.3-70b-versatile`) |
| Vision (Image OCR) | Google Gemini (`gemini-2.5-flash`) |
| Vector Store | ChromaDB (persistent) |
| Embeddings | `all-MiniLM-L6-v2` (SentenceTransformers) |
| PDF Parsing | PyMuPDF (fitz) |

---

## Project Structure

```
RAG-MODEL/
├── chroma_db/                   # Persistent ChromaDB vector store
├── dataset/
│   └── data.pdf                 # Source document for ingestion
├── pydantic_models/
│   └── chat_body.py             # Pydantic request models
├── services/
│   ├── file_extraction_service.py   # PDF & image text extraction
│   ├── ingest_service.py            # PDF chunking & embedding pipeline
│   ├── llm_service.py               # Groq LLM streaming response
│   ├── sort_source_service.py       # Cosine similarity re-ranking
│   └── vector_store_service.py      # ChromaDB query interface
├── .env                         # API keys (never commit)
├── config.py                    # Settings via pydantic-settings
├── ingest.py                    # CLI ingestion script
├── main.py                      # FastAPI app entry point
└── requirements.txt
```

---

## Complete Flow

### Step 1 — Ingestion (One-time setup)

Before the API can answer questions, your source PDF must be chunked, embedded, and stored in ChromaDB.

```
data.pdf
   │
   ▼
IngestService.extract_text_from_pdf_path()   ← PyMuPDF reads page by page
   │
   ▼
IngestService.chunk_text()                   ← splits into 500-word chunks (50-word overlap)
   │
   ▼
ChromaDB.upsert()                            ← embeds with all-MiniLM-L6-v2 & stores
   │
   ▼
chroma_db/ (persisted on disk)
```

Run once:
```bash
python ingest.py ./dataset/data.pdf
```

---

### Step 2 — File Upload & Text Extraction

User uploads a file via `/chat` (HTTP) or `/ws/chat` (WebSocket).

```
User uploads file (PDF / JPEG / PNG / WEBP / GIF)
   │
   ├── application/pdf  →  FileExtractionService.extract_from_pdf()
   │                        PyMuPDF extracts raw text (no API call)
   │
   └── image/*          →  FileExtractionService.extract_from_image()
                            Gemini Vision (gemini-2.5-flash) reads & describes image content
```

---

### Step 3 — Vector Store Query

The extracted text (first 1000 characters) is used to search ChromaDB for the most relevant chunks from the ingested document.

```
extracted_text[:1000]
   │
   ▼
VectorStoreService.query()
   │
   ▼
ChromaDB searches with all-MiniLM-L6-v2 embeddings
   │
   ▼
Returns top 5 chunks with content, page number, source, relevance score
```

---

### Step 4 — LLM Streaming Response

The extracted text and retrieved chunks are combined into a structured prompt and sent to Groq for streaming generation.

```
extracted_text + top-K ChromaDB results
   │
   ▼
LLMService.generate_response()
   │
   ▼
Groq API (llama-3.3-70b-versatile) — streamed chunk by chunk
   │
   ▼
Response format:

## Summary
  Concise summary of relevant content with page citations

## Suggestions
  Actionable insights and recommendations
```

---

## API Endpoints

### `POST /chat`
Upload a file and receive a streamed plain-text response.

- **Content-Type:** `multipart/form-data`
- **Field:** `file` (PDF, JPEG, PNG, WEBP, or GIF)
- **Response:** `StreamingResponse` (text/plain)

**Example (curl):**
```bash
curl -X POST http://localhost:8000/chat \
  -F "file=@./dataset/data.pdf"
```

---

### `WS /ws/chat`
Upload a file over WebSocket and receive chunked JSON messages.

**Send:**
```json
{
  "file": "<base64_encoded_file_bytes>",
  "mime_type": "application/pdf"
}
```

**Receive (sequence of messages):**
```json
{ "type": "search_result", "data": [ ...top-K chunks... ] }
{ "type": "content", "data": "## Summary\n..." }
{ "type": "content", "data": "...next chunk..." }
{ "type": "error", "data": "error message" }
```

---

## Supported File Types

| Format | MIME Type | Extraction Method |
|---|---|---|
| PDF | `application/pdf` | PyMuPDF (local) |
| JPEG | `image/jpeg` | Gemini Vision API |
| PNG | `image/png` | Gemini Vision API |
| WEBP | `image/webp` | Gemini Vision API |
| GIF | `image/gif` | Gemini Vision API |

---

## Setup & Installation

### 1. Clone the repository
```bash
git clone <your-repo-url>
cd rag-model
```

### 2. Create and activate virtual environment
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Mac/Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file in the root directory:
```env
GEMINI_API_KEY=your_gemini_api_key_here
GROQ_API_KEY=your_groq_api_key_here
CHROMA_DB_PATH=./chroma_db
COLLECTION_NAME=rag_documents
TOP_K_RESULTS=5
```

- Get your Groq API key: https://console.groq.com
- Get your Gemini API key: https://aistudio.google.com

### 5. Ingest your document (one-time)
```bash
python ingest.py ./dataset/data.pdf
```

### 6. Run the server
```bash
uvicorn main:app --reload
```

API is live at: `http://localhost:8000`

---

## Deployment (Render)

### Environment Variables
Set these in Render dashboard → Environment:
```
GEMINI_API_KEY=your_gemini_key
GROQ_API_KEY=your_groq_key
CHROMA_DB_PATH=./chroma_db
COLLECTION_NAME=rag_documents
TOP_K_RESULTS=5
```

### Build Command
```bash
pip install -r requirements.txt && python ingest.py ./dataset/data.pdf
```

### Start Command
```bash
uvicorn main:app --host 0.0.0.0 --port 10000
```

> **Note:** Render's free tier does not provide persistent disk storage. The build command re-ingests `data.pdf` on every deploy to ensure ChromaDB is always populated.

---

## Requirements

```
fastapi
uvicorn
pydantic-settings
google-genai
chromadb
sentence-transformers
pymupdf
python-dotenv
groq
python-multipart
```
