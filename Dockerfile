# Dockerfile
FROM ghcr.io/astral-sh/uv:0.5.11-python3.12-bookworm

# Build-Args für Proxy (TH Wildau)
ARG http_proxy
ARG https_proxy
ARG no_proxy

# Setze Proxy-Umgebungsvariablen für Build
ENV http_proxy=$http_proxy \
    https_proxy=$https_proxy \
    no_proxy=$no_proxy

# Arbeitsverzeichnis
WORKDIR /app

# System-Dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Kopiere Projekt-Files
COPY pyproject.toml uv.lock* ./

# Dependencies installieren
RUN uv sync --frozen --no-cache

# Kopiere Source Code
COPY src/ ./src/

# Proxy wird zur Laufzeit über docker-compose gesetzt
# (nicht hier entfernen - kommt von compose environment)

# Port für Streamlit
EXPOSE 8501

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Streamlit Config
ENV STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Start Command
CMD ["uv", "run", "streamlit", "run", "src/Home.py", "--server.port=8501", "--server.address=0.0.0.0"]