import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from config import Settings

settings = Settings()


class VectorStoreService:
    def __init__(self):
        self.client = chromadb.PersistentClient(path=settings.CHROMA_DB_PATH)
        self.embedding_fn = SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        self.collection = self.client.get_or_create_collection(
            name=settings.COLLECTION_NAME,
            embedding_function=self.embedding_fn,
        )

    def query(self, query: str) -> list[dict]:
        results = self.collection.query(
            query_texts=[query],
            n_results=settings.TOP_K_RESULTS,
        )

        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        output = []
        for doc, meta, dist in zip(docs, metas, distances):
            output.append({
                "content": doc,
                "page": meta.get("page", "?"),
                "source": meta.get("source", ""),
                "relevance_score": round(1 - dist, 4),
            })

        return output