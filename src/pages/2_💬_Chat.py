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

@st.cache_resource
def get_vectorstore():
    """Erstellt Vectorstore-Verbindung"""
    embedding_model = OllamaEmbeddings(
        base_url=Config.OLLAMA_BASE_URL,
        model=Config.OLLAMA_EMBEDDING_MODEL
    )
    return get_chroma_vectorstore(embedding_model)

st.title("💬 RAG Chat")

# Prüfe ob Dokumente vorhanden
try:
    vectorstore = get_vectorstore()
    doc_count = vectorstore._collection.count()
except Exception as e:
    st.error(f"❌ Verbindungsfehler: {e}")
    st.stop()

# Warnung wenn keine Dokumente
if doc_count == 0:
    st.warning("⚠️ Keine Dokumente vorhanden!")
    st.info("👉 Gehe zur **Dokumente**-Seite um PDFs hochzuladen.")
    st.stop()

# Sidebar: Einstellungen
with st.sidebar:
    st.header("⚙️ Einstellungen")
    
    st.metric("📄 Dokumente", doc_count)
    
    st.divider()
    
    st.subheader("🔍 Suche")
    k_results = st.slider("Anzahl Ergebnisse", 1, 10, 3)
    
    show_sources = st.checkbox("Quellen anzeigen", value=True)
    show_scores = st.checkbox("Relevanz-Scores anzeigen", value=False)
    
    st.divider()
    
    if st.button("🗑️ Chat löschen"):
        st.session_state.messages = []
        st.rerun()

# Chat-Historie initialisieren
if "messages" not in st.session_state:
    st.session_state.messages = []

# Chat-Historie anzeigen
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # Zeige Quellen falls vorhanden
        if "sources" in message and show_sources:
            with st.expander("📚 Quellen"):
                for i, source in enumerate(message["sources"], 1):
                    st.markdown(f"**{i}.** {source['text'][:200]}...")
                    if show_scores:
                        st.caption(f"Relevanz: {source['score']:.3f} | Quelle: {source['filename']}")
                    else:
                        st.caption(f"Quelle: {source['filename']}")

# Chat-Input
if prompt := st.chat_input("Stelle eine Frage zu deinen Dokumenten..."):
    # User-Nachricht anzeigen und speichern
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Bot-Antwort generieren
    with st.chat_message("assistant"):
        with st.spinner("🔍 Durchsuche Dokumente..."):
            try:
                # Similarity Search
                results_with_score = vectorstore.similarity_search_with_score(prompt, k=k_results)
                
                if not results_with_score:
                    response = "❌ Keine relevanten Dokumente gefunden."
                    sources = []
                else:
                    # Erstelle Antwort
                    response = f"📊 Ich habe **{len(results_with_score)}** relevante Textabschnitte gefunden:\n\n"
                    
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
                            st.markdown(f"### Quelle {i}")
                            st.text(source['text'][:500] + ("..." if len(source['text']) > 500 else ""))
                            
                            if show_scores:
                                st.caption(f"⭐ Relevanz: {source['score']:.3f} | 📄 {source['filename']}")
                            else:
                                st.caption(f"📄 {source['filename']}")
                            
                            st.divider()
                
                # Speichere Antwort
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response,
                    "sources": sources
                })
                
            except Exception as e:
                error_msg = f"❌ Fehler bei der Suche: {e}"
                st.error(error_msg)
                logger.error(f"Search error: {e}", exc_info=True)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg
                })

# Footer
st.divider()
st.caption(f"🤖 Powered by Ollama ({Config.OLLAMA_MODEL}) | 📊 {doc_count} Dokumente durchsuchbar")
