# pages/1_📚_Dokumente.py
import streamlit as st
import logging
from pathlib import Path
import tempfile
import sys

# Füge Parent-Directory zum Path hinzu
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.document_processor import DocumentProcessor
from app.chroma_client import get_chroma_vectorstore
from app.config import Config
from langchain_ollama import OllamaEmbeddings

# TODO: der Server antwortet nicht in time 
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="Dokumentenverwaltung",
    page_icon="📚",
    layout="wide"
)

@st.cache_resource
def get_vectorstore():
    """Erstellt Vectorstore-Verbindung"""
    embedding_model = OllamaEmbeddings(
        base_url=Config.OLLAMA_BASE_URL,
        model=Config.OLLAMA_EMBEDDING_MODEL
    )
    return get_chroma_vectorstore(embedding_model)

st.title("📚 Dokumentenverwaltung")

# Tabs für verschiedene Aktionen
tab1, tab2, tab3 = st.tabs(["📤 Upload", "📊 Übersicht", "🗑️ Verwaltung"])

# Tab 1: Upload
with tab1:
    st.header("Dokument hochladen")
    
    uploaded_file = st.file_uploader(
        "Wähle eine Datei",
        type=['pdf', 'txt', 'docx'],
        help="Unterstützte Formate: PDF, TXT, DOCX"
    )
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        if uploaded_file:
            st.info(f"📄 **{uploaded_file.name}** ({uploaded_file.size / 1024:.1f} KB)")
    
    with col2:
        upload_button = st.button("📤 Hochladen", disabled=uploaded_file is None, type="primary")
    
    if upload_button and uploaded_file:
        with st.spinner("Verarbeite Dokument..."):
            try:
                # Temporäre Datei erstellen
                with tempfile.NamedTemporaryFile(
                    delete=False, 
                    suffix=Path(uploaded_file.name).suffix
                ) as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_path = Path(tmp_file.name)
                
                # Progress anzeigen
                progress_bar = st.progress(0, "Lade Dokument...")
                
                # Dokument verarbeiten
                processor = DocumentProcessor()
                progress_bar.progress(30, "Verarbeite Text...")
                
                chunks = processor.load_and_process_file(tmp_path)
                progress_bar.progress(60, "Erstelle Embeddings...")
                
                # In ChromaDB speichern
                vectorstore = get_vectorstore()
                vectorstore.add_documents(chunks)
                progress_bar.progress(100, "Fertig!")
                
                # Temporäre Datei löschen
                tmp_path.unlink()
                
                st.success(f"✅ **{uploaded_file.name}** erfolgreich hochgeladen!")
                st.info(f"📊 {len(chunks)} Text-Chunks erstellt")
                
                # Zeige Beispiel-Chunk
                with st.expander("👀 Vorschau erstes Chunk"):
                    st.text(chunks[0].page_content[:500] + "...")
                
                st.balloons()
                
            except Exception as e:
                st.error(f"❌ Fehler beim Upload: {e}")
                logger.error(f"Upload-Fehler: {e}", exc_info=True)

# Tab 2: Übersicht
with tab2:
    st.header("Dokument-Übersicht")
    
    try:
        vectorstore = get_vectorstore()
        collection = vectorstore._collection
        
        doc_count = collection.count()
        
        st.metric("📊 Gesamtzahl Chunks", doc_count)
        
        if doc_count > 0:
            # Hole Beispiel-Dokumente
            results = collection.get(limit=10, include=['metadatas'])
            
            if results and results['metadatas']:
                # Extrahiere eindeutige Dateinamen
                filenames = set()
                for metadata in results['metadatas']:
                    if 'filename' in metadata:
                        filenames.add(metadata['filename'])
                
                st.subheader(f"📄 Hochgeladene Dokumente ({len(filenames)})")
                
                for filename in sorted(filenames):
                    # Zähle Chunks pro Datei
                    chunks_count = sum(1 for m in results['metadatas'] if m.get('filename') == filename)
                    st.write(f"- **{filename}** ({chunks_count}+ Chunks)")
        else:
            st.info("Noch keine Dokumente vorhanden.")
            
    except Exception as e:
        st.error(f"Fehler beim Laden der Übersicht: {e}")

# Tab 3: Verwaltung
with tab3:
    st.header("Datenbank-Verwaltung")
    
    try:
        vectorstore = get_vectorstore()
        doc_count = vectorstore._collection.count()
        
        st.warning("⚠️ **Achtung:** Diese Aktionen sind nicht rückgängig zu machen!")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🗑️ Datenbank leeren")
            st.write(f"Aktuell: **{doc_count}** Dokumente")
            
            if st.button("🗑️ Alle Dokumente löschen", type="secondary"):
                if doc_count > 0:
                    with st.spinner("Lösche Dokumente..."):
                        vectorstore._collection.delete(where={})
                    st.success("✅ Datenbank geleert!")
                    st.rerun()
                else:
                    st.info("Datenbank ist bereits leer")
        
        with col2:
            st.subheader("ℹ️ Konfiguration")
            st.code(f"""
Embedding Model: {Config.OLLAMA_EMBEDDING_MODEL}
Chunk Size: {Config.CHUNK_SIZE}
Chunk Overlap: {Config.CHUNK_OVERLAP}
Collection: {Config.CHROMA_COLLECTION_NAME}
            """, language="text")
            
    except Exception as e:
        st.error(f"Fehler: {e}")

# Sidebar Info
with st.sidebar:
    st.header("📊 Status")
    try:
        vectorstore = get_vectorstore()
        doc_count = vectorstore._collection.count()
        st.metric("Dokumente", doc_count)
        st.success("✅ Verbunden")
    except:
        st.error("❌ Nicht verbunden")
    
    st.divider()
    st.info("💡 **Tipp:** Nach dem Upload zur Chat-Seite wechseln!")