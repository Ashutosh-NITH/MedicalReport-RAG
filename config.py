from dotenv import load_dotenv
from pydantic_settings import BaseSettings

load_dotenv()


class Settings(BaseSettings):
    GEMINI_API_KEY: str = ""
    CHROMA_DB_PATH: str = "./chroma_db"
    COLLECTION_NAME: str = "rag_documents"
    TOP_K_RESULTS: int = 5

    class Config:
        env_file = ".env"