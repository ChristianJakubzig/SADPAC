# pages/2_💬_Chat.py
import streamlit as st
import logging
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.chroma_client import get_chroma_vectorstore
from app.rag_pipeline import RAGPipeline
from app.config import Config
from langchain_ollama import OllamaEmbeddings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="Chat",
    page_icon="💬",
    layout="wide"
)

# Session State initialisieren
if "selected_collection" not in st.session_state:
    st.session_state.selected_collection = Config.DOCUMENTS_COLLECTION
if "messages" not in st.session_state:
    st.session_state.messages = {}

@st.cache_resource
def get_embedding_model():
    """Erstellt Ollama Embedding Model (gecached)"""
    return OllamaEmbeddings(
        base_url=Config.OLLAMA_BASE_URL,
        model=Config.OLLAMA_EMBEDDING_MODEL
    )

def get_vectorstore_for_collection(collection_name: str):
    """Erstellt Vectorstore für spezifische Collection"""
    embedding_model = get_embedding_model()
    return get_chroma_vectorstore(embedding_model, collection_name=collection_name)

def get_rag_pipeline(collection_name: str):
    """Erstellt RAG Pipeline für Collection"""
    vectorstore = get_vectorstore_for_collection(collection_name)
    return RAGPipeline(vectorstore, collection_name=collection_name)

st.title("💬 RAG Chat")

# Sidebar: Collection-Auswahl und Einstellungen
with st.sidebar:
    st.header("📚 Collection")
    
    collections = [Config.DOCUMENTS_COLLECTION, Config.METADATA_COLLECTION]
    
    selected_collection = st.selectbox(
        "Wähle Collection",
        collections,
        index=collections.index(st.session_state.selected_collection),
        help="Wähle zwischen Dokumenten und Metadaten"
    )
    
    if selected_collection != st.session_state.selected_collection:
        st.session_state.selected_collection = selected_collection
        st.rerun()
    
    try:
        vectorstore = get_vectorstore_for_collection(selected_collection)
        doc_count = vectorstore._collection.count()
        
        st.metric("📊 Chunks", doc_count)
        st.success("✅ Verbunden")
        
    except Exception as e:
        st.error(f"❌ Verbindungsfehler: {e}")
        doc_count = 0
    
    st.divider()
    
    # Such-Einstellungen
    st.header("⚙️ Einstellungen")
    
    k_results = st.slider(
        "Anzahl Kontext-Quellen", 
        1, 10, 3,
        help="Wie viele relevante Text-Chunks sollen dem LLM als Kontext gegeben werden?"
    )
    
    temperature = st.slider(
        "Kreativität (Temperature)",
        0.0, 1.0, 0.7, 0.1,
        help="0 = faktisch/präzise, 1 = kreativ/variabel"
    )
    
    show_sources = st.checkbox("Quellen anzeigen", value=True)
    show_scores = st.checkbox("Relevanz-Scores anzeigen", value=False)
    
    st.divider()
    
    # Collection-Info
    st.subheader("ℹ️ Info")
    st.caption(f"**Collection:** {selected_collection}")
    st.caption(f"**Chunks:** {doc_count}")
    st.caption(f"**LLM:** {Config.OLLAMA_MODEL}")
    
    st.divider()
    
    if st.button("🗑️ Chat löschen"):
        if selected_collection in st.session_state.messages:
            st.session_state.messages[selected_collection] = []
        st.rerun()

# Warnung wenn keine Dokumente
if doc_count == 0:
    st.warning(f"⚠️ Keine Chunks in Collection '{selected_collection}'!")
    st.info("👉 Gehe zur **Dokumente**-Seite um Dateien hochzuladen oder nutze das Load-Script.")
    
    with st.expander("💡 Dateien laden"):
        st.code(f"""
# Via Script:
python src/scripts/load_documents.py \\
  --folder data/documents \\
  --collection {selected_collection} \\
  --batch-size 5

# Via Makefile:
make load-docs
        """, language="bash")
    st.stop()

# Chat-Historie für aktuelle Collection initialisieren
if selected_collection not in st.session_state.messages:
    st.session_state.messages[selected_collection] = []

# Chat-Historie anzeigen
for message in st.session_state.messages[selected_collection]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # Zeige Quellen falls vorhanden
        if message["role"] == "assistant" and "sources" in message and show_sources and message["sources"]:
            with st.expander(f"📚 Quellen ({len(message['sources'])})"):
                for i, source in enumerate(message["sources"], 1):
                    st.markdown(f"**📄 Quelle {i}**")
                    
                    # Text in scrollbarem Container
                    st.text_area(
                        f"quelle_{i}",
                        source['content'][:500] + ("..." if len(source['content']) > 500 else ""),
                        height=100,
                        key=f"history_{message.get('timestamp', i)}_{i}",
                        label_visibility="collapsed"
                    )
                    
                    # Metadaten
                    col1, col2 = st.columns(2)
                    with col1:
                        if show_scores:
                            st.caption(f"⭐ Relevanz: {source.get('score', 0):.3f}")
                    with col2:
                        st.caption(f"📄 {source['metadata'].get('filename', 'Unbekannt')}")
                    
                    if i < len(message['sources']):
                        st.divider()

# Chat-Input
if prompt := st.chat_input(f"Stelle eine Frage zu '{selected_collection}'..."):
    # User-Nachricht anzeigen und speichern
    st.session_state.messages[selected_collection].append({
        "role": "user", 
        "content": prompt
    })
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Bot-Antwort generieren mit RAG + Streaming
    with st.chat_message("assistant"):
        try:
            # RAG Pipeline erstellen
            rag = get_rag_pipeline(selected_collection)
            
            # Container für gestreamte Antwort
            response_placeholder = st.empty()
            sources_placeholder = st.empty()
            
            full_response = ""
            sources = []
            
            # Streame die Antwort
            with st.spinner("🤔 Denke nach..."):
                for chunk in rag.query_stream(prompt, k=k_results):
                    if chunk["type"] == "sources":
                        # Speichere Quellen
                        sources = chunk["sources"]
                        
                    elif chunk["type"] == "token":
                        # Füge Token zur Antwort hinzu
                        full_response += chunk["token"]
                        response_placeholder.markdown(full_response + "▌")
                        
                    elif chunk["type"] == "error":
                        st.error(f"❌ Fehler: {chunk['error']}")
                        break
            
            # Finale Antwort ohne Cursor
            response_placeholder.markdown(full_response)
            
            # Zeige Quellen
            if sources and show_sources:
                with st.expander(f"📚 Verwendete Quellen ({len(sources)})", expanded=False):
                    for i, source in enumerate(sources, 1):
                        st.markdown(f"**📄 Quelle {i}**")
                        
                        st.text_area(
                            f"source_{i}",
                            source['content'][:500] + ("..." if len(source['content']) > 500 else ""),
                            height=100,
                            key=f"source_{i}_{len(st.session_state.messages[selected_collection])}",
                            label_visibility="collapsed"
                        )
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            if show_scores:
                                st.caption(f"⭐ Relevanz: {source.get('score', 0):.3f}")
                        with col2:
                            st.caption(f"📄 {source['metadata'].get('filename', 'Unbekannt')}")
                        
                        if i < len(sources):
                            st.divider()
            
            # Speichere Assistant-Antwort
            st.session_state.messages[selected_collection].append({
                "role": "assistant",
                "content": full_response,
                "sources": sources,
                "timestamp": len(st.session_state.messages[selected_collection])
            })
            
        except Exception as e:
            error_msg = f"❌ Fehler bei der Anfrage: {e}"
            st.error(error_msg)
            logger.error(f"RAG error: {e}", exc_info=True)
            
            st.session_state.messages[selected_collection].append({
                "role": "assistant",
                "content": error_msg,
                "sources": []
            })

# Footer
st.divider()
col1, col2, col3 = st.columns(3)
with col1:
    st.caption(f"📚 {selected_collection}")
with col2:
    st.caption(f"📊 {doc_count} Chunks")
with col3:
    st.caption(f"🤖 {Config.OLLAMA_MODEL}")