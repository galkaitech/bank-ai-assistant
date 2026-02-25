# ---------------------------------------------------------------------------
# BankMind — Production Dockerfile
# ---------------------------------------------------------------------------
# Multi-stage build to keep the final image lean:
#   Stage 1 (builder): Installs all dependencies
#   Stage 2 (runtime): Copies only what's needed to run the app
#
# Why multi-stage?
# - Separates build tools from the runtime image, reducing attack surface
# - Final image is significantly smaller (no pip, build tools, wheel cache)
# - Better for security scanning — fewer packages = fewer CVEs
#
# Base image choice: python:3.11-slim-bookworm
# - Slim variant: ~50MB vs 1GB for the full image
# - Bookworm (Debian 12): Long-term supported, well-maintained CVE patching
# - Avoid Alpine for Python apps: musl libc causes subtle compatibility issues
#   with some scientific Python packages (NumPy, ChromaDB native extensions)
# ---------------------------------------------------------------------------

# ---- Stage 1: Build ----
FROM python:3.11-slim-bookworm AS builder

# Set consistent, non-interactive environment for apt and pip
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies needed for building Python packages
# build-essential: required for packages with C extensions (e.g., chromadb)
# libpq-dev: required if using psycopg2 for pgvector
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user for the build stage
WORKDIR /build

# Copy requirements first to leverage Docker layer caching.
# If requirements.txt doesn't change, this layer is cached and subsequent
# builds skip the pip install step entirely.
COPY requirements.txt .

# Install dependencies into a venv for cleaner copying to runtime stage
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN pip install --upgrade pip && \
    pip install -r requirements.txt


# ---- Stage 2: Runtime ----
FROM python:3.11-slim-bookworm AS runtime

# Labels for image metadata — useful for registry scanning and audit
LABEL org.opencontainers.image.title="BankMind API" \
      org.opencontainers.image.description="AI-Powered Banking Operations Copilot" \
      org.opencontainers.image.version="0.1.0" \
      org.opencontainers.image.authors="github.com/yourusername"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    # Tell Python where to find the application
    PYTHONPATH="/app"

# Install only runtime system dependencies (no build tools)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user — never run containers as root, especially in banking
# The user is created with no login shell and no home directory
RUN groupadd --gid 1001 bankmind && \
    useradd --uid 1001 --gid bankmind --shell /bin/false --no-create-home bankmind

# Copy the virtual environment from the builder stage
COPY --from=builder /opt/venv /opt/venv

# Set working directory and copy application code
WORKDIR /app

# Copy source code — ordered from least to most frequently changed
# to maximise layer cache effectiveness
COPY config/ ./config/
COPY src/ ./src/
COPY data/sample_docs/ ./data/sample_docs/

# Create directories that the app needs to write to at runtime
RUN mkdir -p /app/data/chroma_db /app/logs && \
    chown -R bankmind:bankmind /app/data /app/logs

# Switch to non-root user
USER bankmind

# Expose the application port
EXPOSE 8000

# Health check — Kubernetes and Docker Compose use this to determine
# if the container is ready to receive traffic
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Production startup command
# - workers=2: safe default; tune based on available CPU cores (2*CPU + 1)
# - --no-access-log: reduces log noise; structured access logging is handled
#   by the middleware in main.py
CMD ["uvicorn", "src.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "2", \
     "--no-access-log"]
