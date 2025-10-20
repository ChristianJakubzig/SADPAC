import logging
from app.chroma_client import get_chroma_vectorstore
from langchain_ollama import OllamaEmbeddings  
from app.config import Config  

logging.basicConfig(level=logging.INFO)

def test_connection():
    """Testet die Verbindung zu ChromaDB"""
    try:
        # Embedding-Modell erstellen
        embedding_model = OllamaEmbeddings(
            base_url=Config.OLLAMA_BASE_URL,
            model=Config.OLLAMA_EMBEDDING_MODEL
        )
       
        # Vectorstore holen
        vectorstore = get_chroma_vectorstore(embedding_model)
       
        print("✅ Verbindung erfolgreich!")
       
        # Collection-Info abrufen
        collection = vectorstore._collection
        print(f"📦 Collection: {collection.name}")
        print(f"📊 Anzahl Dokumente: {collection.count()}")
       
        return True
    except Exception as e:
        print(f"❌ Fehler: {e}")
        return False

if __name__ == "__main__":
    test_connection()