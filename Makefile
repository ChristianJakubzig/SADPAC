# Makefile für RAG Chatbot Projekt

.PHONY: help install docker-up docker-down docker-logs load-docs run test clean

# Standard-Target (wird bei 'make' ohne Argumente ausgeführt)
help:
	@echo "🤖 RAG Chatbot - Verfügbare Befehle:"
	@echo ""
	@echo "  make install       - Installiert Dependencies"
	@echo "  make docker-up     - Startet ChromaDB"
	@echo "  make docker-down   - Stoppt ChromaDB"
	@echo "  make docker-logs   - Zeigt ChromaDB Logs"
	@echo "  make load-docs     - Lädt Dokumente in DB"
	@echo "  make run           - Startet Streamlit App"
	@echo "  make test          - Führt Tests aus"
	@echo "  make clean         - Löscht temp. Dateien"
	@echo ""

# Dependencies installieren
install:
	@echo "📦 Installiere Dependencies..."
	uv sync

# Docker
docker-up:
	@echo "🐳 Starte ChromaDB..."
	docker-compose up -d
	@echo "✅ ChromaDB läuft auf http://localhost:8000"

docker-down:
	@echo "🛑 Stoppe ChromaDB..."
	docker-compose down

docker-logs:
	@echo "📋 ChromaDB Logs:"
	docker-compose logs -f chromadb

# Dokumente laden
load-docs:
	@echo "📚 Lade Dokumente..."
	python src/scripts/load_documents.py

load-docs-clear:
	@echo "🗑️  Lösche DB und lade Dokumente neu..."
	python src/scripts/load_documents.py --clear

# App starten
run:
	@echo "🚀 Starte Streamlit App..."
	streamlit run src/Home.py

# Tests
test:
	@echo "🧪 Führe Tests aus..."
	python src/tests/test_chroma_client.py

# Cleanup
clean:
	@echo "🧹 Lösche temporäre Dateien..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@echo "✅ Cleanup fertig"