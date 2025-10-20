# Home.py (vorher main.py)
import streamlit as st
from app.chroma_client import get_chroma_vectorstore
from langchain_ollama import OllamaEmbeddings
from app.config import Config

st.set_page_config(
    page_title="RAG Chatbot - TH Wildau",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 RAG Chatbot")
st.markdown("## Willkommen!")

st.markdown("""
Dieser Chatbot hilft dir, Informationen aus hochgeladenen Dokumenten zu finden.

### 📚 Erste Schritte:
1. Gehe zu **Dokumente** (Sidebar links) um PDFs hochzuladen
2. Wechsle zu **Chat** um Fragen zu stellen

### 📊 Aktueller Status:
""")

# Zeige Statistiken
try:
    @st.cache_resource
    def get_vectorstore():
        embedding_model = OllamaEmbeddings(
            base_url=Config.OLLAMA_BASE_URL,
            model=Config.OLLAMA_EMBEDDING_MODEL
        )
        return get_chroma_vectorstore(embedding_model)
    
    vectorstore = get_vectorstore()
    doc_count = vectorstore._collection.count()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("📄 Dokumente", doc_count)
    
    with col2:
        st.metric("🗄️ ChromaDB", "✅ Verbunden")
    
    with col3:
        st.metric("🤖 Ollama", "✅ Bereit")
    
    if doc_count == 0:
        st.info("👉 Noch keine Dokumente vorhanden. Lade welche auf der **Dokumente**-Seite hoch!")
    else:
        st.success(f"✅ {doc_count} Dokumente bereit für Abfragen!")

except Exception as e:
    st.error(f"❌ Verbindungsfehler: {e}")
    st.info("Stelle sicher, dass ChromaDB läuft: `docker-compose up -d`")

st.divider()

st.markdown("""
### ℹ️ Info
- **Embedding Model**: {model}
- **Ollama Server**: {server}
- **Collection**: {collection}
""".format(
    model=Config.OLLAMA_EMBEDDING_MODEL,
    server=Config.OLLAMA_BASE_URL,
    collection=Config.CHROMA_COLLECTION_NAME
))