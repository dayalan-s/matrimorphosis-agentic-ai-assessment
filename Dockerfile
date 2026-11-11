# Targets:  app (default, docker compose)  ·  runner (CI: scripted LLM, no model downloads)  ·  test (black-box tests)
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=0
WORKDIR /app
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-install-project --no-dev
COPY . /app


FROM python:3.12-slim-bookworm AS base
# opencv-python-headless (pulled in by RapidOCR) needs glib
RUN apt-get update && apt-get install -y --no-install-recommends libglib2.0-0 && rm -rf /var/lib/apt/lists/* \
 && groupadd --system --gid 999 nonroot && useradd --system --gid 999 --uid 999 --create-home nonroot \
 && mkdir /data && chown nonroot:nonroot /data
COPY --from=builder --chown=nonroot:nonroot /app /app
ENV PATH="/app/.venv/bin:$PATH" PYTHONUNBUFFERED=1 PYTHONPATH=/app/src HF_HOME=/app/.hf
USER nonroot
WORKDIR /app
EXPOSE 8000 7860
CMD ["uvicorn", "app.main:app", "--app-dir", "src", "--host", "0.0.0.0", "--port", "8000"]


# CI: the autograder runs your API with a scripted model and a hash embedder: no API key, no network, no model downloads
FROM base AS runner
ENV LLM_MODE=fake EMBEDDER=fake ASSESSMENT_DEBUG=1 \
    QDRANT_URL=path:/data/qdrant KB_DB=/data/kb.sqlite UPLOAD_DIR=/data \
    CHECKPOINT_DB=/data/checkpoints.sqlite MCP_DB=/data/mcp.sqlite MCP_URL=http://localhost:9000/mcp


FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS test
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=0 PATH="/app/.venv/bin:$PATH" PYTHONUNBUFFERED=1
WORKDIR /app
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-install-project --only-group dev
COPY tests ./tests
CMD ["pytest", "tests/", "--tb=short", "-q"]


# docker compose (last stage = default): the real thing, with the models baked in (bekko embeddings + RapidOCR PP-OCRv6)
FROM base AS app
RUN python -c "from huggingface_hub import hf_hub_download as d; [d('hotchpotch/bekko-embedding-v1-a8m', f) for f in ('tokenizer.json', 'onnx/model.onnx')]" \
 && python -c "from rapidocr import RapidOCR; RapidOCR()"
ENV HF_HUB_OFFLINE=1
