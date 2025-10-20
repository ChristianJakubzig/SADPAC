#!/usr/bin/env python3
# scripts/load_documents.py
"""
Script zum initialen Laden von Dokumenten in ChromaDB mit Collection-Support.
Nutzt die gemeinsame Verarbeitungslogik aus app/document_processor.py
"""
import argparse
import logging
import sys
import time
from pathlib import Path

# Füge Parent-Directory zum Path hinzu
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.document_processor import DocumentProcessor
from app.chroma_client import get_chroma_vectorstore
from app.config import Config
from langchain_ollama import OllamaEmbeddings

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
        default=str(Config.DATA_DIR),
        help=f"Pfad zum Dokumenten-Ordner (default: {Config.DATA_DIR})"
    )
    parser.add_argument(
        "--collection",
        type=str,
        default=None,
        help=f"Collection-Name (default: {Config.CHROMA_COLLECTION_NAME})"
    )
    parser.add_argument(
        "--file-types",
        nargs="+",
        default=[".pdf", ".txt", ".docx"],
        help="Zu ladende Dateitypen (default: .pdf .txt .docx)"
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Löscht existierende Collection vor dem Laden"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Anzahl Chunks pro Batch für Ollama (default: 10)"
    )
    
    args = parser.parse_args()
    
    # Validiere Ordner
    folder_path = Path(args.folder)
    if not folder_path.exists():
        logger.error(f"❌ Ordner nicht gefunden: {folder_path}")
        sys.exit(1)
    
    # Collection-Name bestimmen
    collection_name = args.collection or Config.CHROMA_COLLECTION_NAME
    
    logger.info(f"🚀 Starte Dokumenten-Import")
    logger.info(f"   📁 Ordner: {folder_path}")
    logger.info(f"   📦 Collection: {collection_name}")
    
    # 1. Ollama Embedding-Modell (TH Wildau Server)
    logger.info(f"📊 Verbinde mit Ollama: {Config.OLLAMA_BASE_URL}")
    logger.info(f"🔧 Embedding-Model: {Config.OLLAMA_EMBEDDING_MODEL}")
    
    embedding_model = OllamaEmbeddings(
        base_url=Config.OLLAMA_BASE_URL,
        model=Config.OLLAMA_EMBEDDING_MODEL
    )
    
    # 2. ChromaDB Verbindung mit gewählter Collection
    logger.info(f"🔌 Verbinde mit ChromaDB Collection: {collection_name}...")
    vectorstore = get_chroma_vectorstore(embedding_model, collection_name=collection_name)
    
    # 3. Optional: Collection leeren
    if args.clear:
        logger.warning(f"⚠️  Lösche existierende Dokumente aus Collection '{collection_name}'...")
        collection = vectorstore._collection
        collection.delete(where={})
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
    batch_size = args.batch_size
    total_chunks = len(chunks)
    logger.info(f"💾 Speichere {total_chunks} Chunks in ChromaDB (Batch-Größe: {batch_size})...")
    
    successful_batches = 0
    failed_batches = 0
    
    # Verarbeite in Batches
    for i in range(0, total_chunks, batch_size):
        batch = chunks[i:i + batch_size]
        batch_num = (i // batch_size) + 1
        total_batches = (total_chunks + batch_size - 1) // batch_size
        
        logger.info(f"  📦 Batch {batch_num}/{total_batches}: {len(batch)} Chunks...")
        
        # Retry-Logik
        max_retries = 3
        success = False
        
        for attempt in range(max_retries):
            try:
                vectorstore.add_documents(batch)
                successful_batches += 1
                success = True
                break  # Erfolg, gehe zum nächsten Batch
            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 5  # 5s, 10s, 15s
                    logger.warning(f"  ⚠️  Versuch {attempt + 1} fehlgeschlagen: {e}")
                    logger.info(f"  ⏳ Warte {wait_time}s vor erneutem Versuch...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"  ❌ Batch {batch_num} nach {max_retries} Versuchen fehlgeschlagen!")
                    failed_batches += 1
        
        if not success:
            logger.warning(f"  ⏭️  Überspringe Batch {batch_num} und fahre fort...")
    
    # 6. Statistiken
    total_docs = vectorstore._collection.count()
    logger.info("=" * 60)
    logger.info(f"✅ Import abgeschlossen!")
    logger.info(f"   📊 Collection '{collection_name}' enthält jetzt {total_docs} Dokumente")
    logger.info(f"   ✅ Erfolgreiche Batches: {successful_batches}/{total_batches}")
    if failed_batches > 0:
        logger.warning(f"   ⚠️  Fehlgeschlagene Batches: {failed_batches}/{total_batches}")
    logger.info("=" * 60)
    
    # Zeige Beispiel-Metadaten
    if chunks and successful_batches > 0:
        logger.info("\n📊 Beispiel-Metadaten:")
        example = chunks[0]
        for key, value in example.metadata.items():
            logger.info(f"  {key}: {value}")


if __name__ == "__main__":
    main()