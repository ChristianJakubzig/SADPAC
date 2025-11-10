# pages/1_📚_Dokumente.py
import streamlit as st
import logging
from pathlib import Path
import tempfile
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.document_processor import DocumentProcessor
from app.chroma_client import get_chroma_vectorstore
from app.config import Config
from langchain_ollama import OllamaEmbeddings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="Dokumentenverwaltung",
    page_icon="📚",
    layout="wide"
)

# Session State
if "selected_collection" not in st.session_state:
    st.session_state.selected_collection = Config.DOCUMENTS_COLLECTION

@st.cache_resource
def get_embedding_model():
    """Erstellt Ollama Embedding Model"""
    return OllamaEmbeddings(
        base_url=Config.OLLAMA_BASE_URL,
        model=Config.OLLAMA_EMBEDDING_MODEL
    )

def get_vectorstore_for_collection(collection_name: str):
    """Erstellt Vectorstore für spezifische Collection"""
    embedding_model = get_embedding_model()
    return get_chroma_vectorstore(embedding_model, collection_name=collection_name)

st.title("📚 Dokumentenverwaltung")

# Sidebar: Collection-Auswahl
with st.sidebar:
    st.header("📦 Collection")
    
    collections = [Config.DOCUMENTS_COLLECTION, Config.METADATA_COLLECTION]
    
    selected_collection = st.selectbox(
        "Aktive Collection",
        collections,
        index=collections.index(st.session_state.selected_collection),
        help="Wähle zwischen Dokumenten und Metadaten"
    )
    
    if selected_collection != st.session_state.selected_collection:
        st.session_state.selected_collection = selected_collection
    
    st.divider()
    
    # Zeige Status beider Collections
    @st.cache_data(ttl=60)
    def get_all_collection_stats():
        stats = {}
        for coll in collections:
            try:
                vs = get_vectorstore_for_collection(coll)
                stats[coll] = vs._collection.count()
            except Exception as e:
                stats[coll] = 0
        return stats

    st.subheader("📊 Status")

    stats = get_all_collection_stats()

    for coll in collections:
        count = stats.get(coll, 0)
        icon = "✅" if count > 0 else "⚪"
        st.metric(
            f"{icon} {coll.replace('-collection', '')}",
            f"{count} Docs")

# Tabs für verschiedene Aktionen
tab1, tab2, tab3 = st.tabs(["📤 Upload", "📊 Übersicht", "🗑️ Verwaltung"])

# Tab 1: Upload
with tab1:
    st.header(f"Dokument hochladen → {selected_collection}")
    
    st.info(f"📌 Uploads werden in **{selected_collection}** gespeichert")
    
    # Erweiterte Einstellungen
    with st.expander("⚙️ Upload-Einstellungen"):
        batch_size = st.slider(
            "Batch-Größe für Embeddings",
            min_value=5,
            max_value=50,
            value=10,
            step=5,
            help="Kleinere Batches = langsamer aber sicherer. Bei Timeouts verringern!"
        )
        st.caption(f"💡 Empfohlen für TH Wildau Server: 5-10 Chunks pro Batch")
    
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
                
                # In ChromaDB speichern mit Batching (wichtig für große Dokumente!)
                vectorstore = get_vectorstore_for_collection(selected_collection)
                
                total_chunks = len(chunks)
                
                logger.info(f"Speichere {total_chunks} Chunks in Batches von {batch_size}...")
                
                # Verarbeite in Batches
                for i in range(0, total_chunks, batch_size):
                    batch = chunks[i:i + batch_size]
                    batch_num = (i // batch_size) + 1
                    total_batches = (total_chunks + batch_size - 1) // batch_size
                    
                    # Progress aktualisieren
                    progress_pct = 60 + int((i / total_chunks) * 40)
                    progress_bar.progress(
                        progress_pct, 
                        f"Speichere Batch {batch_num}/{total_batches}..."
                    )
                    
                    try:
                        vectorstore.add_documents(batch)
                    except Exception as batch_error:
                        st.warning(f"⚠️ Batch {batch_num} fehlgeschlagen: {batch_error}")
                        logger.error(f"Batch {batch_num} error: {batch_error}")
                        # Fahre mit nächstem Batch fort
                        continue
                
                progress_bar.progress(100, "Fertig!")
                
                # Temporäre Datei löschen
                tmp_path.unlink()
                
                st.success(f"✅ **{uploaded_file.name}** erfolgreich in '{selected_collection}' hochgeladen!")
                st.info(f"📊 {len(chunks)} Text-Chunks erstellt")
                
                # Zeige Beispiel-Chunk
                with st.expander("👀 Vorschau erstes Chunk"):
                    st.text(chunks[0].page_content[:500] + "...")
                
                st.balloons()
                
                # Cache clearen für Refresh
                st.cache_resource.clear()
                
            except Exception as e:
                st.error(f"❌ Fehler beim Upload: {e}")
                logger.error(f"Upload-Fehler: {e}", exc_info=True)

# Tab 2: Übersicht
with tab2:
    st.header(f"Übersicht: {selected_collection}")
    
    try:
        vectorstore = get_vectorstore_for_collection(selected_collection)
        collection = vectorstore._collection
        
        doc_count = collection.count()
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("📊 Text-Chunks", doc_count)
        with col2:
            st.metric("📦 Collection", selected_collection)
        
        if doc_count > 0:
            st.divider()
            
            # Hole Sample-Dokumente
            results = collection.get(limit=50, include=['metadatas'])
            
            if results and results['metadatas']:
                # Extrahiere eindeutige Dateinamen
                filenames = {}
                for metadata in results['metadatas']:
                    if 'filename' in metadata:
                        fname = metadata['filename']
                        filenames[fname] = filenames.get(fname, 0) + 1
                
                st.subheader(f"📄 Hochgeladene Dokumente ({len(filenames)})")
                
                # Zeige als Tabelle
                import pandas as pd
                df = pd.DataFrame([
                    {"Dateiname": fname, "Chunks": count}
                    for fname, count in sorted(filenames.items())
                ])
                st.dataframe(df, use_container_width=True, hide_index=True)
                
                # Download-Option
                csv = df.to_csv(index=False)
                st.download_button(
                    label="📥 Liste als CSV",
                    data=csv,
                    file_name=f"{selected_collection}_dokumente.csv",
                    mime="text/csv"
                )
        else:
            st.info("Noch keine Dokumente vorhanden.")
            
            with st.expander("💡 Dokumente hinzufügen"):
                st.markdown("""
                **Option 1: Via UI**
                - Nutze den Upload-Tab oben
                
                **Option 2: Via Script (für viele Dokumente)**
                """)
                
                if selected_collection == Config.DOCUMENTS_COLLECTION:
                    folder = "data/documents"
                else:
                    folder = "data/metadata"
                
                st.code(f"""
# Dokumente laden:
python src/scripts/load_documents.py \\
  --folder {folder} \\
  --collection {selected_collection} \\
  --batch-size 5
                """, language="bash")
            
    except Exception as e:
        st.error(f"Fehler beim Laden der Übersicht: {e}")

# Tab 3: Verwaltung
with tab3:
    st.header("Datenbank-Verwaltung")
    
    col1, col2 = st.columns(2)
    
    # Collection-spezifische Verwaltung
    with col1:
        st.subheader(f"🗑️ {selected_collection}")
        
        try:
            vectorstore = get_vectorstore_for_collection(selected_collection)
            doc_count = vectorstore._collection.count()
            
            st.write(f"Aktuell: **{doc_count}** Dokumente")
            
            st.warning("⚠️ Diese Aktion ist nicht rückgängig zu machen!")
            
            if st.button(f"🗑️ {selected_collection} leeren", type="secondary"):
                if doc_count > 0:
                    with st.spinner("Lösche Dokumente..."):
                        vectorstore._collection.delete(where={})
                    st.success(f"✅ {selected_collection} geleert!")
                    st.cache_resource.clear()
                    st.rerun()
                else:
                    st.info("Collection ist bereits leer")
        except Exception as e:
            st.error(f"Fehler: {e}")
    
    # Beide Collections verwalten
    with col2:
        st.subheader("🗑️ Alle Collections")
        
        st.warning("⚠️⚠️ Löscht ALLE Daten in BEIDEN Collections!")
        
        if st.button("🗑️ ALLES löschen", type="secondary", help="Vorsicht!"):
            try:
                for coll in collections:
                    vs = get_vectorstore_for_collection(coll)
                    vs._collection.delete(where={})
                st.success("✅ Alle Collections geleert!")
                st.cache_resource.clear()
                st.rerun()
            except Exception as e:
                st.error(f"Fehler: {e}")
    
    st.divider()
    
    # Konfiguration
    st.subheader("ℹ️ Konfiguration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.code(f"""
📊 Collection Settings:
- Dokumente: {Config.DOCUMENTS_COLLECTION}
- Metadaten: {Config.METADATA_COLLECTION}
        """, language="text")
    
    with col2:
        st.code(f"""
🔧 Processing Settings:
- Chunk Size: {Config.CHUNK_SIZE}
- Chunk Overlap: {Config.CHUNK_OVERLAP}
- Embedding: {Config.OLLAMA_EMBEDDING_MODEL}
        """, language="text")