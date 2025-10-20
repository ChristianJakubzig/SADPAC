#!/usr/bin/env python3
# scripts/load_documents.py
"""
Script zum initialen Laden von Dokumenten in ChromaDB.
Nutzt die gemeinsame Verarbeitungslogik aus app/document_processor.py
"""
import argparse
import logging
import sys
from pathlib import Path

# Füge Parent-Directory zum Path hinzu
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.document_processor import DocumentProcessor
from app.chroma_client import get_chroma_vectorstore
from app.config import Config
from langchain_community.embeddings import OllamaEmbeddings

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Lädt Dokumente in ChromaDB mit Ollama Embeddings (TH Wildau)"
    )
    parser.add_argument(
        "--folder",
        type=str,
        default=str(Config.DATA_DIR),  # Nutzt Config.DATA_DIR als Default
        help=f"Pfad zum Dokumenten-Ordner (default: {Config.DATA_DIR})"
    )
    parser.add_argument(
        "--file-types",
        nargs="+",
        default=[".pdf", ".txt", ".docx"],
        help="Zu ladende Dateitypen (default: .pdf .txt .docx)"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        help="Größe der Text-Chunks (default: 1000)"
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=200,
        help="Überlappung zwischen Chunks (default: 200)"
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Löscht existierende Collection vor dem Laden"
    )

    parser.add_argument(
        "--collection",
        type=str,
        default=None,
        help=f"Collection-Name (default: {Config.CHROMA_COLLECTION_NAME})"
    )
    
    args = parser.parse_args()
    
    # Validiere Ordner
    folder_path = Path(args.folder)
    if not folder_path.exists():
        logger.error(f"❌ Ordner nicht gefunden: {folder_path}")
        sys.exit(1)
    
    logger.info(f"🚀 Starte Dokumenten-Import aus: {folder_path}")
    
    # 1. Ollama Embedding-Modell (TH Wildau Server)
    logger.info(f"📦 Verbinde mit Ollama: {Config.OLLAMA_BASE_URL}")
    logger.info(f"📊 Embedding-Model: {Config.OLLAMA_EMBEDDING_MODEL}")
    
    embedding_model = OllamaEmbeddings(
        base_url=Config.OLLAMA_BASE_URL,
        model=Config.OLLAMA_EMBEDDING_MODEL
    )
    
    # 2. ChromaDB Verbindung
    logger.info("🔌 Verbinde mit ChromaDB...")
    vectorstore = get_chroma_vectorstore(embedding_model)
    
    # 3. Optional: Collection leeren
    if args.clear:
        logger.warning("⚠️  Lösche existierende Dokumente...")
        collection = vectorstore._collection
        collection.delete(where={})  # Löscht alle Dokumente
        logger.info("🗑️  Collection geleert")
    
    # 4. Dokumente laden und verarbeiten
    processor = DocumentProcessor(
        chunk_size=Config.CHUNK_SIZE,
        chunk_overlap=Config.CHUNK_OVERLAP
    )
    
    logger.info("📚 Lade und verarbeite Dokumente...")
    chunks = processor.load_and_process_folder(folder_path, args.file_types)
    
    if not chunks:
        logger.warning("⚠️  Keine Dokumente gefunden!")
        sys.exit(0)
    
    # 5. In ChromaDB speichern (mit Batching für große Dokumente)
    batch_size = 10  # Anzahl Chunks pro Batch (anpassbar)
    total_chunks = len(chunks)
    logger.info(f"💾 Speichere {total_chunks} Chunks in ChromaDB (Batch-Größe: {batch_size})...")
    
    # Verarbeite in Batches
    for i in range(0, total_chunks, batch_size):
        batch = chunks[i:i + batch_size]
        batch_num = (i // batch_size) + 1
        total_batches = (total_chunks + batch_size - 1) // batch_size
        
        logger.info(f"  📦 Batch {batch_num}/{total_batches}: {len(batch)} Chunks...")
        
        try:
            vectorstore.add_documents(batch)
        except Exception as e:
            logger.error(f"  ❌ Fehler bei Batch {batch_num}: {e}")
            logger.warning(f"  ⏭️  Überspringe Batch und fahre fort...")
            continue
    
    logger.info("✅ Alle Batches verarbeitet!")
    
    # 6. Statistiken
    total_docs = vectorstore._collection.count()
    logger.info(f"✅ Erfolgreich! ChromaDB enthält jetzt {total_docs} Dokumente")
    
    # Zeige Beispiel-Metadaten
    if chunks:
        logger.info("\n📊 Beispiel-Metadaten:")
        example = chunks[0]
        for key, value in example.metadata.items():
            logger.info(f"  {key}: {value}")


if __name__ == "__main__":
    main()