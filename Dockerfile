FROM python:3.11-slim AS base

WORKDIR /app

# System dependencies for Pillow and torch CPU builds.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies before copying source so the layer is cached.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source.
COPY src/ ./src/
COPY pyproject.toml .

# Non-root user for security.
RUN adduser --disabled-password --gecos "" appuser
USER appuser

# HuggingFace cache lives in a writable directory.
ENV HF_HOME=/tmp/hf_cache \
    TRANSFORMERS_CACHE=/tmp/hf_cache \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

EXPOSE 8000

# Warm the model cache at build time so the first request isn't cold.
# Requires network access; skip with --build-arg SKIP_WARMUP=1 if building offline.
ARG SKIP_WARMUP=0
RUN if [ "$SKIP_WARMUP" != "1" ]; then \
        python -c "from transformers import ViTForImageClassification, ViTImageProcessor; \
            ViTImageProcessor.from_pretrained('google/vit-base-patch16-224'); \
            ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')"; \
    fi || true

CMD ["uvicorn", "src.server:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
