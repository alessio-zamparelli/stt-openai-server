# syntax=docker/dockerfile:1
#
# whisper-api — OpenAI-compatible Whisper server (CPU-only).
#
# Built with uv, following the reference pattern in
# pocket-tts-openai-server/deploy/Dockerfile and the official uv Docker guide:
#   https://docs.astral.sh/uv/guides/integration/docker/
#
# The builder runs on python:3.14-slim (system Python 3.14) and copies the uv
# binary from ghcr.io/astral-sh/uv:0.12.13 — pinned to the exact uv version that
# generated uv.lock. Using the system interpreter with UV_PYTHON_DOWNLOADS=0
# keeps the venv's Python path identical across both stages (the same image
# name, python:3.14-slim, in builder and runtime) so copying .venv over is safe.
#
# Guide patterns applied: bind-mounted manifest files (no lockfile layer),
# UV_COMPILE_BYTECODE, cache mounts, --no-editable, .venv via .dockerignore.
#
# CPU-only by design: faster-whisper runs on CTranslate2's CPU backend with
# INT8 quantization — no CUDA/NVIDIA layers, no torch wheels.
#
# Build:    docker build -f deploy/Dockerfile -t whisper-api .
# Run:      docker run --rm -p 8080:8080 -v whisper-data:/data whisper-api
#
# The first startup downloads the base model (~140 MB int8) into /data/hf.
# Mount /data as a volume so weights persist across restarts.

###############################################################################
# Stage 1 — builder: resolve locked dependencies into a venv
###############################################################################
FROM python:3.14-slim AS builder

# Pin uv to the exact version used to (re)generate uv.lock.
COPY --from=ghcr.io/astral-sh/uv:0.12.13 /uv /uvx /bin/

# Use the system Python already present in the slim image (3.14).
ENV UV_PYTHON_DOWNLOADS=0 \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1

WORKDIR /app

# Layer 1 — install ALL locked dependencies without the project. Manifest
# files are bind-mounted per the uv guide's canonical pattern: they never
# enter an image layer, and this RUN's cache invalidates only when they change.
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    uv sync --locked --no-install-project --no-dev --no-editable

###############################################################################
# Stage 2 — runtime: minimal, non-root, persistent /data volume
###############################################################################
FROM python:3.14-slim AS runtime

# OpenMP runtime needed by CTranslate2 for CPU multithreading, plus CA
# certificates for downloading model weights from the Hugging Face Hub.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Reuse the system Python across stages; put the app venv on PATH.
ENV PATH="/app/.venv/bin:$PATH" \
    UV_PYTHON_DOWNLOADS=0 \
    PYTHONUNBUFFERED=1 \
    WHISPER_HOST=0.0.0.0 \
    WHISPER_PORT=8080 \
    WHISPER_MODEL_NAME=base \
    WHISPER_COMPUTE_TYPE=int8 \
    HF_HOME=/data/hf

WORKDIR /app
COPY --from=builder /app/.venv /app/.venv
COPY app.py ./

# Non-root runtime user; /data must stay writable for downloaded model weights.
RUN useradd --create-home --uid 10001 whisper \
    && mkdir -p /data/hf \
    && chown -R whisper:whisper /data

# Persistent volume: HF model weights (HF_HOME=/data/hf). Mount it to avoid
# re-downloading the model on every container restart.
VOLUME ["/data"]

USER whisper
EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=180s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/health', timeout=4).read()"]

ENTRYPOINT ["python", "app.py"]
