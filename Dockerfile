# syntax=docker/dockerfile:1

# ---------- stage 1: build the React SPA ----------
FROM node:22-alpine AS web
WORKDIR /web
COPY web/package.json web/package-lock.json* ./
RUN npm install --no-audit --no-fund
COPY web/ ./
RUN npm run build

# ---------- stage 2: python dependencies ----------
FROM python:3.12-slim AS deps
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /build
COPY requirements.txt ./
RUN python -m venv /opt/venv \
 && /opt/venv/bin/pip install --upgrade pip \
 && /opt/venv/bin/pip install -r requirements.txt

# ---------- stage 3: runtime ----------
FROM python:3.12-slim AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    RAG_HOST=0.0.0.0 \
    RAG_PORT=8000 \
    RAG_DATA_DIR=/data \
    RAG_INDEX_DIR=/data/index \
    RAG_CORPUS_DIR=/data/corpus \
    RAG_UPLOAD_DIR=/data/uploads \
    RAG_AUTH_DB_PATH=/data/users.sqlite3

# curl is only here for the container healthcheck.
RUN apt-get update \
 && apt-get install -y --no-install-recommends curl \
 && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --uid 10001 rag
WORKDIR /app

COPY --from=deps /opt/venv /opt/venv
COPY --chown=rag:rag app/ ./app/
COPY --chown=rag:rag eval/ ./eval/
COPY --chown=rag:rag scripts/ ./scripts/
COPY --chown=rag:rag data/golden/ ./data/golden/
COPY --chown=rag:rag pyproject.toml README.md ./
COPY --from=web --chown=rag:rag /web/dist ./web/dist

RUN mkdir -p /data/index /data/corpus /data/uploads && chown -R rag:rag /data
USER rag
VOLUME ["/data"]
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD curl -fsS http://localhost:8000/live || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
