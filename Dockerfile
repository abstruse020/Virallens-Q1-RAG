# syntax=docker/dockerfile:1

FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

ENV PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    VIRTUAL_ENV=/app/.venv \
    PATH="/app/.venv/bin:$PATH" \
    HF_HOME=/app/.cache/huggingface \
    HF_HUB_DISABLE_PROGRESS_BARS=1 \
    TQDM_DISABLE=1

WORKDIR /app

# Install dependencies first for better layer caching
COPY pyproject.toml uv.lock ./

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project

# Install the application
COPY src ./src

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# Download the embedding model into the image so it runs offline
RUN python -c \
    "from sentence_transformers import SentenceTransformer; \
    SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

# Copy documents and build the RAG index
COPY data ./data

RUN pdf-rag index

# Runtime configuration template (the API key is never baked into the image)
COPY .env.example ./

ENTRYPOINT ["pdf-rag"]
CMD ["chat"]
