# app/document_processor.py
import logging
from pathlib import Path
from typing import List, Optional
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredWordDocumentLoader,
)

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """Verarbeitet Dokumente und bereitet sie für ChromaDB vor"""
    
    def __init__(self, chunk_size: int = None, chunk_overlap: int = None):
        # Importiere Config hier um Circular Import zu vermeiden
        from .config import Config
        
        # Nutze Config-Werte als Default
        chunk_size = chunk_size or Config.CHUNK_SIZE
        chunk_overlap = chunk_overlap or Config.CHUNK_OVERLAP
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
    
    def load_document(self, file_path: Path) -> List[Document]:
        """Lädt ein Dokument basierend auf der Dateiendung"""
        suffix = file_path.suffix.lower()
        
        try:
            if suffix == ".pdf":
                loader = PyPDFLoader(str(file_path))
            elif suffix == ".txt":
                loader = TextLoader(str(file_path))
            elif suffix in [".doc", ".docx"]:
                loader = UnstructuredWordDocumentLoader(str(file_path))
            else:
                logger.warning(f"Unsupported file type: {suffix}")
                return []
            
            documents = loader.load()
            logger.info(f"✅ Geladen: {file_path.name} ({len(documents)} Seiten)")
            return documents
            
        except Exception as e:
            logger.error(f"❌ Fehler beim Laden von {file_path.name}: {e}")
            return []
    
    def process_documents(self, documents: List[Document]) -> List[Document]:
        """Teilt Dokumente in Chunks und fügt Metadaten hinzu"""
        chunks = self.text_splitter.split_documents(documents)
        
        # Erweitere Metadaten
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_id"] = i
            chunk.metadata["chunk_size"] = len(chunk.page_content)
        
        logger.info(f"📄 {len(documents)} Dokumente → {len(chunks)} Chunks")
        return chunks
    
    def load_and_process_file(self, file_path: Path) -> List[Document]:
        """Lädt und verarbeitet eine einzelne Datei"""
        documents = self.load_document(file_path)
        if not documents:
            return []
        
        chunks = self.process_documents(documents)
        
        # Füge Dateinamen zu Metadaten hinzu
        for chunk in chunks:
            chunk.metadata["filename"] = file_path.name
            chunk.metadata["source"] = str(file_path)
        
        return chunks
    
    def load_and_process_folder(
        self, 
        folder_path: Path, 
        file_types: Optional[List[str]] = None
    ) -> List[Document]:
        """Lädt und verarbeitet alle Dateien in einem Ordner"""
        if file_types is None:
            file_types = [".pdf", ".txt", ".doc", ".docx"]
        
        all_chunks = []
        
        for file_type in file_types:
            files = list(folder_path.glob(f"*{file_type}"))
            logger.info(f"🔍 Gefunden: {len(files)} {file_type}-Dateien")
            
            for file_path in files:
                chunks = self.load_and_process_file(file_path)
                all_chunks.extend(chunks)
        
        logger.info(f"✅ Gesamt: {len(all_chunks)} Chunks aus {folder_path}")
        return all_chunks