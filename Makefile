# Makefile für RAG Chatbot Projekt

.PHONY: help install dev prod docker-build docker-up docker-down docker-restart docker-logs docker-logs-app docker-logs-chroma docker-ps load-docs load-metadata load-all run test clean clean-all

# Standard-Target
help:
	@echo "🤖 RAG Chatbot - Verfügbare Befehle:"
	@echo ""
	@echo "📦 Setup:"
	@echo "  make install          - Installiert Dependencies"
	@echo ""
	@echo "💻 Entwicklung (Lokal):"
	@echo "  make dev              - Startet nur ChromaDB (für lokale Entwicklung)"
	@echo "  make run              - Startet Streamlit lokal (benötigt 'make dev')"
	@echo "  make load-docs        - Lädt Dokumente (lokal)"
	@echo "  make load-metadata    - Lädt Metadaten (lokal)"
	@echo ""
	@echo "🚀 Production (Docker):"
	@echo "  make docker-build     - Baut alle Docker Images"
	@echo "  make prod             - Startet komplette App in Docker"
	@echo "  make docker-down      - Stoppt alle Container"
	@echo "  make docker-restart   - Neustart aller Container"
	@echo "  make docker-logs      - Zeigt alle Logs"
	@echo "  make docker-logs-app  - Zeigt nur App-Logs"
	@echo "  make docker-ps        - Zeigt Container-Status"
	@echo ""
	@echo "🧪 Tests & Cleanup:"
	@echo "  make test             - Führt Tests aus"
	@echo "  make clean            - Löscht temp. Dateien"
	@echo "  make clean-all        - Clean + DB-Daten löschen"
	@echo ""
	@echo "💡 Workflow:"
	@echo "  Entwicklung:  make dev && make run"
	@echo "  Production:   make docker-build && make prod"
	@echo ""

# Dependencies installieren
install:
	@echo "📦 Installiere Dependencies..."
	uv sync

# ============================================
# ENTWICKLUNG (Lokal)
# ============================================

dev:
	@echo "💻 Starte ChromaDB für lokale Entwicklung..."
	@echo "   (RAG-App läuft NICHT in Docker)"
	docker-compose up -d chromadb
	@echo ""
	@echo "✅ ChromaDB läuft auf http://localhost:8000"
	@echo "👉 Starte jetzt die App mit: make run"

run:
	@echo "🚀 Starte Streamlit App (lokal)..."
	@echo "   Stelle sicher, dass ChromaDB läuft: make dev"
	@echo ""
	uv run streamlit run src/Home.py

# ============================================
# PRODUCTION (Docker)
# ============================================

docker-build:
	@echo "🔨 Baue Docker Images..."
	docker-compose build --no-cache
	@echo "✅ Build abgeschlossen"

prod: docker-build
	@echo "🚀 Starte komplette App in Docker..."
	docker-compose up -d
	@echo ""
	@echo "⏳ Warte auf Services (30s)..."
	@sleep 30
	@echo ""
	@echo "✅ Services laufen:"
	@echo "   📊 ChromaDB:  http://localhost:8000"
	@echo "   🤖 RAG App:   http://localhost:8501"
	@echo ""
	@echo "📋 Logs ansehen:  make docker-logs"
	@echo "🛑 Stoppen:       make docker-down"

docker-up:
	@echo "🐳 Starte alle Container..."
	docker-compose up -d
	@sleep 10
	@echo "✅ Container gestartet"
	@make docker-ps

docker-down:
	@echo "🛑 Stoppe alle Container..."
	docker-compose down
	@echo "✅ Container gestoppt"

docker-restart:
	@echo "🔄 Starte Container neu..."
	docker-compose restart
	@echo "✅ Container neu gestartet"

docker-logs:
	@echo "📋 Logs aller Container (Ctrl+C zum Beenden):"
	docker-compose logs -f

docker-logs-app:
	@echo "📋 RAG-App Logs (Ctrl+C zum Beenden):"
	docker-compose logs -f rag-app

docker-logs-chroma:
	@echo "📋 ChromaDB Logs (Ctrl+C zum Beenden):"
	docker-compose logs -f chromadb

docker-ps:
	@echo "📊 Container Status:"
	@docker-compose ps

# ============================================
# DATEN LADEN
# ============================================

load-docs:
	@echo "📚 Lade Dokumente..."
	uv run python src/scripts/load_documents.py \
		--folder data/documents \
		--collection documents-collection \
		--batch-size 5

load-docs-clear:
	@echo "🗑️  Lösche DB und lade Dokumente neu..."
	uv run python src/scripts/load_documents.py \
		--folder data/documents \
		--collection documents-collection \
		--batch-size 5 \
		--clear

load-metadata:
	@echo "📊 Lade Metadaten..."
	uv run python src/scripts/load_documents.py \
		--folder data/metadata \
		--collection metadata-collection \
		--batch-size 5

load-metadata-clear:
	@echo "🗑️  Lösche Metadaten und lade neu..."
	uv run python src/scripts/load_documents.py \
		--folder data/metadata \
		--collection metadata-collection \
		--batch-size 5 \
		--clear

load-all: load-docs load-metadata
	@echo "✅ Alle Daten geladen!"

# ============================================
# TESTS
# ============================================

test:
	@echo "🧪 Führe Tests aus..."
	uv run python src/tests/test_chroma_client.py

# ============================================
# CLEANUP
# ============================================

clean:
	@echo "🧹 Lösche temporäre Dateien..."
	@find . -type d -name "__pycache__" -print0 2>/dev/null | xargs -0 rm -rf 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@find . -type d -name ".pytest_cache" -print0 2>/dev/null | xargs -0 rm -rf 2>/dev/null || true
	@find . -type d -name "*.egg-info" -print0 2>/dev/null | xargs -0 rm -rf 2>/dev/null || true
	@find . -type f -name ".coverage" -delete 2>/dev/null || true
	@echo "✅ Cleanup fertig"

clean-all: clean
	@echo "⚠️  Lösche auch ChromaDB Daten..."
	@read -p "Wirklich ALLE Daten löschen? (y/N) " confirm && \
	if [ "$$confirm" = "y" ] || [ "$$confirm" = "Y" ]; then \
		rm -rf data/chromadb/* 2>/dev/null || true; \
		echo "✅ ChromaDB Daten gelöscht"; \
	else \
		echo "❌ Abgebrochen"; \
	fi