# 🤖 RAG Chatbot mit Ollama

Ein intelligenter Chatbot, der mittels **Retrieval Augmented Generation (RAG)** Fragen zu hochgeladenen Dokumenten beantwortet. Nutzt Ollama für LLM und Embeddings, ChromaDB als Vektordatenbank und Streamlit als UI.

## 📋 Inhaltsverzeichnis

- [Features](#-features)
- [Architektur](#-architektur)
- [Voraussetzungen](#-voraussetzungen)
- [Installation](#-installation)
- [Verwendung](#-verwendung)
  - [Entwicklung](#entwicklung-lokal)
  - [Production](#production-docker)
- [Dokumenten-Management](#-dokumenten-management)
- [Konfiguration](#-konfiguration)
- [Projekt-Struktur](#-projekt-struktur)
- [Troubleshooting](#-troubleshooting)

---

## ✨ Features

- 📚 **Multi-Format Support**: PDF, TXT, DOCX
- 🔍 **Semantische Suche**: Findet relevante Textabschnitte in Dokumenten
- 💬 **Streaming Chat**: LLM-Antworten erscheinen in Echtzeit
- 📊 **Zwei Collections**: Separate Verwaltung von Dokumenten und Metadaten
- 🎯 **Quellen-Nachweis**: Zeigt verwendete Textabschnitte mit Relevanz-Scores
- 🐳 **Docker-Ready**: Vollständig containerisiert für einfaches Deployment
- ⚡ **Batch-Processing**: Verarbeitet große Dokumente in Batches

---

## 🏗️ Architektur

```
┌─────────────────┐
│   Streamlit UI  │  ← Browser (Port 8501)
└────────┬────────┘
         │
    ┌────▼─────────────────────────┐
    │     RAG Pipeline             │
    │  (Retrieval + Generation)    │
    └────┬──────────────────┬──────┘
         │                  │
    ┌────▼────────┐    ┌───▼──────────┐
    │  ChromaDB   │    │ Ollama Server│
    │  (Lokal)    │    │ (TH Wildau)  │
    └─────────────┘    └──────────────┘
```

### Komponenten:

- **Streamlit**: Web-UI für Chat und Dokumenten-Upload
- **ChromaDB**: Vektordatenbank für Embeddings (läuft lokal in Docker)
- **Ollama**: LLM (llama3.2) und Embeddings (granite-embedding:278m) auf TH Wildau Server
- **LangChain**: Orchestrierung der RAG-Pipeline

---

## 📦 Voraussetzungen

- **Docker** & **Docker Compose**
- **Python 3.12+** (für lokale Entwicklung)
- **uv** (Python Package Manager) - [Installation](https://docs.astral.sh/uv/)
- **Make** (optional, aber empfohlen)
- Zugang zum **TH Wildau Ollama-Server** (oder eigener Ollama-Server)

### System-Anforderungen:

- **Speicherplatz**: ~5 GB für Docker Images + Daten
- **RAM**: Mindestens 4 GB
- **Netzwerk**: Zugang zu TH Wildau Proxy (bei Verwendung des Uni-Servers)

---

## 🚀 Installation

### 1. Repository klonen

```bash
git clone <repository-url>
cd python
```

### 2. Dependencies installieren (für lokale Entwicklung)

```bash
make install
# oder
uv sync
```

---

## 💻 Verwendung

### Entwicklung (Lokal)

**Ideal für Code-Änderungen und schnelles Testing**

```bash
# 1. Starte ChromaDB in Docker
make dev

# 2. Starte Streamlit-App lokal
make run

# App läuft auf: http://localhost:8501
```

**Vorteile:**
- ⚡ Schnelle Iterationen (kein Container-Rebuild)
- 🐛 Einfaches Debugging
- 🔄 Code-Änderungen sofort sichtbar

**Stoppen:**
```bash
# Streamlit: Strg+C
# ChromaDB:
make docker-down
```

---

### Production (Docker)

**Deployment-Ready, alles in Containern**

```bash
# 1. Baue Docker Images
make docker-build

# 2. Starte alle Services
make prod

# App läuft auf: http://localhost:8501
```

**Services:**
- 📊 ChromaDB: `http://localhost:8000`
- 🤖 RAG App: `http://localhost:8501`

**Management:**
```bash
# Status prüfen
make docker-ps

# Logs ansehen
make docker-logs              # Alle Services
make docker-logs-app          # Nur RAG-App
make docker-logs-chroma       # Nur ChromaDB

# Neustart
make docker-restart

# Stoppen
make docker-down
```

---

## 📚 Dokumenten-Management

### Via Web-UI

1. Öffne `http://localhost:8501`
2. Gehe zu **📚 Dokumente**
3. Wähle Collection (documents/metadata)
4. Lade Dateien hoch

### Via Script (Bulk-Loading)

**Empfohlen für viele Dokumente**

```bash
# Dokumente laden
make load-docs

# Metadaten laden
make load-metadata

# Beide laden
make load-all

# Mit Clear (DB vorher leeren)
make load-docs-clear
```

**Erweiterte Optionen:**
```bash
# Custom Ordner
python src/scripts/load_documents.py \
  --folder /pfad/zu/dokumenten \
  --collection documents-collection \
  --batch-size 5

# Kleinere Batches bei Timeouts
python src/scripts/load_documents.py \
  --folder data/documents \
  --batch-size 3 \
  --clear
```

### Unterstützte Formate

- ✅ PDF (`.pdf`)
- ✅ Text (`.txt`)
- ✅ Word (`.docx`)

---

## ⚙️ Konfiguration

### TH Wildau Proxy

Bei Verwendung im TH Wildau Netzwerk ist der Proxy bereits in `docker-compose.yml` konfiguriert:

```yaml
environment:
  - http_proxy=http://proxy.th-wildau.de:8080
  - https_proxy=http://proxy.th-wildau.de:8080
```

**Für externes Deployment:** Entferne Proxy-Einstellungen aus `docker-compose.yml`.

---

## 📁 Projekt-Struktur

```
.
├── data/
│   ├── chromadb/          # ChromaDB Daten (persistent)
│   ├── documents/         # Dokumente für documents-collection
│   └── metadata/          # Dokumente für metadata-collection
├── src/
│   ├── app/
│   │   ├── chroma_client.py        # ChromaDB Verbindung
│   │   ├── config.py               # Konfiguration
│   │   ├── document_processor.py   # PDF/TXT/DOCX Verarbeitung
│   │   └── rag_pipeline.py         # RAG Logic (Retrieval + Generation)
│   ├── pages/
│   │   ├── 1_📚_Dokumente.py      # Dokumenten-Upload & Verwaltung
│   │   └── 2_💬_Chat.py           # Chat-Interface
│   ├── scripts/
│   │   └── load_documents.py       # Bulk-Loading Script
│   ├── tests/
│   │   └── test_chroma_client.py   # Tests
│   └── Home.py                     # Streamlit Hauptseite
├── docker-compose.yml     # Docker Services Definition
├── Dockerfile            # RAG-App Container
├── Makefile             # Convenience Commands
└── pyproject.toml       # Python Dependencies
```

---

## 🧪 Tests

```bash
# Teste ChromaDB Verbindung
make test

---

## 🧹 Maintenance

### Cleanup

```bash
# Temporäre Dateien löschen
make clean

# Inkl. ChromaDB Daten (Vorsicht!)
make clean-all
```

### Docker Aufräumen

```bash
# Speicherplatz freigeben
sudo docker system prune -a --volumes
```

---

## 🐛 Troubleshooting

### ChromaDB nicht erreichbar

```bash
# Status prüfen
make docker-ps

# Logs ansehen
make docker-logs-chroma

# Neustart
make docker-restart
```

### Timeout beim Dokumenten-Upload

**Problem:** Ollama-Server antwortet nicht rechtzeitig

**Lösung:** Reduziere Batch-Size
```bash
# In UI: Upload-Einstellungen → Batch-Size auf 3-5 setzen
# Via Script:
python src/scripts/load_documents.py --batch-size 3
```

---

## 📚 Verwendete Technologien

- **[Streamlit](https://streamlit.io/)** - Web UI Framework
- **[LangChain](https://www.langchain.com/)** - LLM Application Framework
- **[ChromaDB](https://www.trychroma.com/)** - Vector Database
- **[Ollama](https://ollama.ai/)** - LLM Runtime
- **[uv](https://docs.astral.sh/uv/)** - Fast Python Package Manager
- **[Docker](https://www.docker.com/)** - Containerization

---

## 📝 Makefile Commands Übersicht

```bash
make help              # Zeigt alle verfügbaren Commands

# Development
make dev               # Startet ChromaDB für lokale Entwicklung
make run               # Startet App lokal

# Production
make docker-build      # Baut Docker Images
make prod              # Startet komplette App in Docker
make docker-down       # Stoppt alle Container

# Daten
make load-docs         # Lädt Dokumente
make load-metadata     # Lädt Metadaten
make load-all          # Lädt alles

# Monitoring
make docker-ps         # Container Status
make docker-logs       # Alle Logs
make docker-logs-app   # Nur App-Logs

# Maintenance
make clean             # Cleanup
make test              # Tests ausführen
```

---

**Viel Erfolg mit deinem RAG Chatbot! 🚀**