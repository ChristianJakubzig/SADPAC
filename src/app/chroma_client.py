# app/chroma_client.py
import logging
from urllib.parse import urlparse
import chromadb
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from .config import Config

logger = logging.getLogger(__name__)

def get_chroma_vectorstore(embedding_model=None) -> Chroma:
    """
    Verbindet sich mit ChromaDB und nutzt Ollama Embeddings von TH Wildau
    """
    try:
        # Falls kein Embedding-Model übergeben, nutze Ollama
        if embedding_model is None:
            embedding_model = OllamaEmbeddings(
                base_url=Config.OLLAMA_BASE_URL,
                model=Config.OLLAMA_EMBEDDING_MODEL
            )
        
        u = urlparse(Config.CHROMA_HTTP_URL)
        ssl = (u.scheme == "https")
        port = u.port or (443 if ssl else 80)
        
        client = chromadb.HttpClient(
            host=u.hostname,
            port=port,
            ssl=ssl,
            tenant=getattr(Config, "CHROMA_TENANT", "default_tenant"),
            database=getattr(Config, "CHROMA_DATABASE", "default_database"),
        )
        
        vectorstore = Chroma(
            collection_name=Config.CHROMA_COLLECTION_NAME,
            embedding_function=embedding_model,
            client=client,
        )
        
        logger.info("✅ ChromaDB verbunden mit Ollama Embeddings: %s", 
                    Config.OLLAMA_EMBEDDING_MODEL)
        return vectorstore
        
    except Exception as e:
        logger.error("❌ Fehler beim Verbinden: %s", e)
        raise