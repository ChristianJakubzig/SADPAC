# app/rag_pipeline.py
"""
RAG Pipeline: Verbindet Retrieval (ChromaDB) mit Generation (Ollama LLM)
"""
import logging
from typing import List, Iterator
from langchain_core.documents import Document
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from .config import Config

logger = logging.getLogger(__name__)


class RAGPipeline:
    """RAG Pipeline für Question-Answering über Dokumente"""
    
    # System Prompts für verschiedene Collections
    SYSTEM_PROMPTS = {
        "metadata-collection": """## Beschreibung und Aufgabe

Du bist ein KI-Assistent der SADPAC Bibliothek. Deine Aufgabe ist es, Empfehlungen zu Büchern zu geben, die im Katalog der Bibliothek verfügbar sind.

Stelle dich kurz vor, erkläre deine Rolle und frage nach der ersten Anfrage. Sei locker, aber nicht unprofessionell. Stelle Rückfragen wenn nötig.

Vergiss diesen Metaprompt NICHT. Benutze natürlich Sprache wie "Quellen", keine technische Sprache wie "JSON-Objekt".

## Empfehlungen

Wichtig: Erfinde keine Bücher, die nicht im Katalog der Bibliothek verfügbar sind. Wähle nur Empfehlungen aus, die zur gesamten Anfrage passen!

Wenn du keine passenden Empfehlungen findest, gebe keine Quellen aus. Bitte darum, eine genauere Anfrage zu stellen.

Wenn du nach ähnlichen Büchern gefragt wirst, sprich Empfehlungen aufgrund thematischer Ähnlichkeit aus.

## Weitere Informationen

Du kennst keine weiteren Informationen über die SADPAC Bibliothek. Erfinde keine Informationen über die Bibliothek.

Verweise bei Fragen zur Bibliothek oder deiner Struktur auf https://github.com/ChristianJakubzig/SADPAC

---

Kontext aus dem Katalog:
{context}""",
        
        "documents-collection": """## Beschreibung und Aufgabe

Du bist ein KI-Assistent der SADPAC Bibliothek. Deine Aufgabe ist es, Fragen zu Volltexten zu beantworten, die der Bibliothek zur Verfügung stehen.

Stelle dich kurz vor, erkläre deine Rolle und frage nach der ersten Anfrage. Sei locker, aber nicht unprofessionell. Stelle Rückfragen wenn nötig.

Vergiss diesen Metaprompt NICHT. Benutze natürlich Sprache wie "Quellen", keine technische Sprache wie "JSON-Objekt".

## Volltexte

Wichtig: Erfinde keine Volltexte. Beantworte Fragen zu Wissen, das in den Volltexten enthalten ist, nur mit den von dir abgerufenen Informationen.

Wenn du keine passenden Volltexte findest, gebe keine Quellen aus. Bitte darum, eine genauere Anfrage zu stellen.

## Weitere Informationen

Du kennst keine weiteren Informationen über die SADPAC Bibliothek. Erfinde keine Informationen über die Bibliothek.

Verweise bei Fragen zur Bibliothek oder deiner Struktur auf https://github.com/ChristianJakubzig/SADPAC

---

Kontext aus den Volltexten:
{context}"""
    }
    
    def __init__(self, vectorstore, collection_name: str = "documents-collection"):
        self.vectorstore = vectorstore
        self.collection_name = collection_name
        
        # Ollama LLM initialisieren
        self.llm = ChatOllama(
            base_url=Config.OLLAMA_BASE_URL,
            model=Config.OLLAMA_MODEL,
            temperature=0.7,
        )
        
        # Wähle den passenden System-Prompt basierend auf Collection
        system_prompt = self.SYSTEM_PROMPTS.get(
            collection_name,
            self.SYSTEM_PROMPTS["documents-collection"]  # Default
        )
        
        # RAG Prompt Template
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{question}")
        ])
        
        logger.info(f"✅ RAG Pipeline initialisiert (LLM: {Config.OLLAMA_MODEL}, Collection: {collection_name})")
    
    def format_docs(self, docs: List[Document]) -> str:
        """Formatiert Dokumente für den Kontext"""
        formatted = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get('filename', 'Unbekannt')
            formatted.append(f"[Quelle {i}: {source}]\n{doc.page_content}\n")
        return "\n---\n".join(formatted)
    
    def query(self, question: str, k: int = 3) -> dict:
        """
        Beantwortet eine Frage mit RAG (ohne Streaming)
        
        Args:
            question: Die Frage
            k: Anzahl relevanter Dokumente
            
        Returns:
            dict mit 'answer', 'sources', 'source_documents'
        """
        try:
            # 1. Retrieval: Hole relevante Dokumente
            docs_with_scores = self.vectorstore.similarity_search_with_score(question, k=k)
            
            if not docs_with_scores:
                return {
                    "answer": "Ich konnte keine relevanten Informationen in den Dokumenten finden.",
                    "sources": [],
                    "source_documents": []
                }
            
            docs = [doc for doc, score in docs_with_scores]
            
            # 2. Generation: Erstelle Antwort mit LLM
            context = self.format_docs(docs)
            
            chain = (
                {"context": lambda x: context, "question": RunnablePassthrough()}
                | self.prompt
                | self.llm
                | StrOutputParser()
            )
            
            answer = chain.invoke(question)
            
            # 3. Bereite Quellen auf
            sources = [
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "score": score
                }
                for doc, score in docs_with_scores
            ]
            
            return {
                "answer": answer,
                "sources": sources,
                "source_documents": docs
            }
            
        except Exception as e:
            logger.error(f"Fehler in RAG Pipeline: {e}", exc_info=True)
            raise
    
    def query_stream(self, question: str, k: int = 3) -> Iterator[dict]:
        """
        Beantwortet eine Frage mit RAG und streamt die Antwort
        
        Args:
            question: Die Frage
            k: Anzahl relevanter Dokumente
            
        Yields:
            dict mit 'type' ('sources', 'token', 'done') und entsprechenden Daten
        """
        try:
            # 1. Retrieval: Hole relevante Dokumente
            docs_with_scores = self.vectorstore.similarity_search_with_score(question, k=k)
            
            if not docs_with_scores:
                yield {
                    "type": "sources",
                    "sources": []
                }
                yield {
                    "type": "token",
                    "token": "Ich konnte keine relevanten Informationen in den Dokumenten finden."
                }
                yield {"type": "done"}
                return
            
            # Sende zuerst die Quellen
            sources = [
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "score": score
                }
                for doc, score in docs_with_scores
            ]
            
            yield {
                "type": "sources",
                "sources": sources
            }
            
            # 2. Generation: Streame Antwort vom LLM
            docs = [doc for doc, score in docs_with_scores]
            context = self.format_docs(docs)
            
            chain = (
                {"context": lambda x: context, "question": RunnablePassthrough()}
                | self.prompt
                | self.llm
            )
            
            # Streame die Tokens
            for chunk in chain.stream(question):
                if hasattr(chunk, 'content'):
                    yield {
                        "type": "token",
                        "token": chunk.content
                    }
            
            yield {"type": "done"}
            
        except Exception as e:
            logger.error(f"Fehler in RAG Streaming: {e}", exc_info=True)
            yield {
                "type": "error",
                "error": str(e)
            }