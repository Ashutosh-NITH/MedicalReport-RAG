import fitz  # PyMuPDF
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from config import Settings

settings = Settings()


class IngestService:
    def __init__(self):
        self.client = chromadb.PersistentClient(path=settings.CHROMA_DB_PATH)
        self.embedding_fn = SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2" # 1 word 32 bytes -> 716 * 11562 * 1024 * 1024 * 16 
        )
        self.collection = self.client.get_or_create_collection(
            name=settings.COLLECTION_NAME,
            embedding_function=self.embedding_fn,
        )

    def extract_text_from_pdf_path(self, pdf_path: str) -> list[dict]:
        """Extract text page by page from a PDF file path."""
        doc = fitz.open(pdf_path)
        return self._extract_pages(doc)

    def extract_text_from_pdf_bytes(self, pdf_bytes: bytes) -> list[dict]:
        """Extract text page by page from PDF bytes (uploaded file)."""
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        return self._extract_pages(doc)

    def _extract_pages(self, doc) -> list[dict]:
        pages = []
        for page_num, page in enumerate(doc):
            text = page.get_text().strip()
            if text:
                pages.append({
                    "page": page_num + 1,
                    "content": text,
                })
        doc.close()
        return pages

    def chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
        """Split text into overlapping chunks."""
        words = text.split()
        chunks = []
        i = 0
        while i < len(words):
            chunk = " ".join(words[i:i + chunk_size])
            chunks.append(chunk)
            i += chunk_size - overlap
        return chunks

    def ingest_pdf(self, pdf_path: str):
        """Full pipeline for CLI ingestion: extract → chunk → embed → store."""
        print(f"📄 Extracting text from: {pdf_path}")
        pages = self.extract_text_from_pdf_path(pdf_path)
        self._store_pages(pages, source=pdf_path)

    def _store_pages(self, pages: list[dict], source: str):
        all_chunks, all_ids, all_metadata = [], [], []

        for page in pages:
            chunks = self.chunk_text(page["content"])
            for j, chunk in enumerate(chunks):
                chunk_id = f"page{page['page']}_chunk{j}"
                all_chunks.append(chunk)
                all_ids.append(chunk_id)
                all_metadata.append({
                    "page": page["page"],
                    "source": source,
                })

        print(f"✅ Total chunks created: {len(all_chunks)}")
        print("⚙️  Embedding and storing in ChromaDB...")

        batch_size = 100
        for i in range(0, len(all_chunks), batch_size):
            self.collection.upsert(
                documents=all_chunks[i:i + batch_size],
                ids=all_ids[i:i + batch_size],
                metadatas=all_metadata[i:i + batch_size],
            )

        print(f"🎉 Ingestion complete! {len(all_chunks)} chunks stored.")



        