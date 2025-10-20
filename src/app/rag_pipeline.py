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
    
    def __init__(self, vectorstore, collection_name: str = "documents"):
        self.vectorstore = vectorstore
        self.collection_name = collection_name
        
        # Ollama LLM initialisieren
        self.llm = ChatOllama(
            base_url=Config.OLLAMA_BASE_URL,
            model=Config.OLLAMA_MODEL,
            temperature=0.7,
        )
        
        # RAG Prompt Template
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """Du bist ein hilfreicher Assistent, der Fragen basierend auf bereitgestellten Dokumenten beantwortet.

Kontext aus den Dokumenten:
{context}

Wichtige Regeln:
1. Beantworte die Frage NUR auf Basis des bereitgestellten Kontexts
2. Wenn die Antwort nicht im Kontext zu finden ist, sage das ehrlich
3. Zitiere relevante Stellen aus dem Kontext
4. Sei präzise und konkret
5. Antworte auf Deutsch"""),
            ("human", "{question}")
        ])
        
        logger.info(f"✅ RAG Pipeline initialisiert (LLM: {Config.OLLAMA_MODEL})")
    
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