import os
from pathlib import Path
from typing import Optional

class Config:
    # ChromaDB
    CHROMA_HOST: str = os.getenv("CHROMA_HOST", "localhost")
    CHROMA_PORT: int = int(os.getenv("CHROMA_PORT", "8000"))
    CHROMA_HTTP_URL: str = os.getenv("CHROMA_HTTP_URL") or f"http://{CHROMA_HOST}:{CHROMA_PORT}"
    CHROMA_TENANT: str = os.getenv("CHROMA_TENANT", "default_tenant")
    CHROMA_DATABASE: str = os.getenv("CHROMA_DATABASE", "default_database")

    # Ollama Config
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "https://ollama-bim24.apps.rhos.th-wildau.de")
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "llama3.2")
    OLLAMA_EMBEDDING_MODEL: str = os.getenv("OLLAMA_EMBEDDING_MODEL", "granite-embedding:278m")
    OLLAMA_KEEP_ALIVE: str = os.getenv("OLLAMA_KEEP_ALIVE", "5m")
    
    # Embedding Model
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "oliverguhr/revosax-granite-embedding-278m-multilingual")
    
    # LLM Config
    ANTHROPIC_API_KEY: Optional[str] = os.getenv("ANTHROPIC_API_KEY")
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
    
    # Document Processing
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "200"))

    # Data Directories
    BASE_DATA_DIR: Path = Path(__file__).parent.parent.parent / "data"
    DOCUMENTS_DIR: Path = BASE_DATA_DIR / "documents"
    METADATA_DIR: Path = BASE_DATA_DIR / "metadata"

        # Collection Names
    DOCUMENTS_COLLECTION: str = os.getenv("DOCUMENTS_COLLECTION", "documents-collection")
    METADATA_COLLECTION: str = os.getenv("METADATA_COLLECTION", "metadata-collection")
    
    # Default Collection (für Abwärtskompatibilität)
    CHROMA_COLLECTION_NAME: str = os.getenv("CHROMA_COLLECTION_NAME", DOCUMENTS_COLLECTION)
    DATA_DIR: Path = DOCUMENTS_DIR  # Default für load_documents.py