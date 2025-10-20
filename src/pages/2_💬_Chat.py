# pages/2_💬_Chat.py
import streamlit as st
import logging
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.chroma_client import get_chroma_vectorstore
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
    st.session_state.messages = {}  # Dictionary: collection_name -> messages

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

st.title("💬 RAG Chat")

# Sidebar: Collection-Auswahl und Einstellungen
with st.sidebar:
    st.header("📚 Collection")
    
    # Collection-Auswahl
    collections = [Config.DOCUMENTS_COLLECTION, Config.METADATA_COLLECTION]
    
    selected_collection = st.selectbox(
        "Wähle Collection",
        collections,
        index=collections.index(st.session_state.selected_collection),
        help="Wähle zwischen Dokumenten und Metadaten"
    )
    
    # Wenn Collection geändert wurde
    if selected_collection != st.session_state.selected_collection:
        st.session_state.selected_collection = selected_collection
        st.rerun()
    
    # Hole Vectorstore für gewählte Collection
    try:
        vectorstore = get_vectorstore_for_collection(selected_collection)
        doc_count = vectorstore._collection.count()
        
        st.metric("📄 Dokumente", doc_count)
        st.success("✅ Verbunden")
        
    except Exception as e:
        st.error(f"❌ Verbindungsfehler: {e}")
        doc_count = 0
    
    st.divider()
    
    # Such-Einstellungen
    st.header("⚙️ Einstellungen")
    
    k_results = st.slider("Anzahl Ergebnisse", 1, 10, 3)
    
    show_sources = st.checkbox("Quellen anzeigen", value=True)
    show_scores = st.checkbox("Relevanz-Scores anzeigen", value=False)
    
    st.divider()
    
    # Collection-Info
    st.subheader("ℹ️ Collection Info")
    st.caption(f"**Name:** {selected_collection}")
    st.caption(f"**Dokumente:** {doc_count}")
    
    st.divider()
    
    if st.button("🗑️ Chat löschen"):
        if selected_collection in st.session_state.messages:
            st.session_state.messages[selected_collection] = []
        st.rerun()

# Warnung wenn keine Dokumente
if doc_count == 0:
    st.warning(f"⚠️ Keine Dokumente in Collection '{selected_collection}'!")
    st.info("👉 Gehe zur **Dokumente**-Seite um PDFs hochzuladen oder nutze das Load-Script.")
    
    with st.expander("💡 Dokumente laden"):
        st.code(f"""
# Via Script:
python src/scripts/load_documents.py \\
  --folder data/documents \\
  --collection {selected_collection} \\
  --batch-size 5

# Via Makefile:
make load-docs      # Für Dokumente
make load-metadata  # Für Metadaten
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
        if "sources" in message and show_sources and message["sources"]:
            with st.expander("📚 Quellen"):
                for i, source in enumerate(message["sources"], 1):
                    st.markdown(f"**{i}.** {source['text'][:200]}...")
                    if show_scores:
                        st.caption(f"⭐ Relevanz: {source['score']:.3f} | 📄 {source['filename']}")
                    else:
                        st.caption(f"📄 {source['filename']}")

# Chat-Input
if prompt := st.chat_input(f"Stelle eine Frage zu '{selected_collection}'..."):
    # User-Nachricht anzeigen und speichern
    st.session_state.messages[selected_collection].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Bot-Antwort generieren
    with st.chat_message("assistant"):
        with st.spinner("🔍 Durchsuche Dokumente..."):
            try:
                # Similarity Search
                results_with_score = vectorstore.similarity_search_with_score(prompt, k=k_results)
                
                if not results_with_score:
                    response = f"❌ Keine relevanten Dokumente in '{selected_collection}' gefunden."
                    sources = []
                else:
                    # Erstelle Antwort
                    response = f"📊 Ich habe **{len(results_with_score)}** relevante Textabschnitte in '{selected_collection}' gefunden:\n\n"
                    
                    sources = []
                    for i, (doc, score) in enumerate(results_with_score, 1):
                        # Sammle Quellen
                        sources.append({
                            "text": doc.page_content,
                            "score": score,
                            "filename": doc.metadata.get('filename', 'Unbekannt')
                        })
                    
                    response += "\n\n💡 **Hinweis:** Die vollständige LLM-Integration kommt im nächsten Schritt. "
                    response += "Aktuell zeige ich dir die relevantesten Textabschnitte."
                
                st.markdown(response)
                
                # Zeige Quellen inline
                if sources and show_sources:
                    with st.expander("📚 Gefundene Quellen", expanded=True):
                        for i, source in enumerate(sources, 1):
                            st.markdown(f"### 📄 Quelle {i}")
                            
                            # Zeige Text in scrollbarem Container
                            st.text_area(
                                f"Text {i}",
                                source['text'],
                                height=150,
                                key=f"source_{i}",
                                label_visibility="collapsed"
                            )
                            
                            # Metadaten
                            col1, col2 = st.columns(2)
                            with col1:
                                if show_scores:
                                    st.caption(f"⭐ Relevanz: {source['score']:.3f}")
                            with col2:
                                st.caption(f"📄 {source['filename']}")
                            
                            if i < len(sources):
                                st.divider()
                
                # Speichere Antwort
                st.session_state.messages[selected_collection].append({
                    "role": "assistant",
                    "content": response,
                    "sources": sources
                })
                
            except Exception as e:
                error_msg = f"❌ Fehler bei der Suche: {e}"
                st.error(error_msg)
                logger.error(f"Search error: {e}", exc_info=True)
                st.session_state.messages[selected_collection].append({
                    "role": "assistant",
                    "content": error_msg
                })

# Footer
st.divider()
col1, col2, col3 = st.columns(3)
with col1:
    st.caption(f"📚 Collection: {selected_collection}")
with col2:
    st.caption(f"📊 {doc_count} Dokumente")
with col3:
    st.caption(f"🤖 {Config.OLLAMA_EMBEDDING_MODEL}")