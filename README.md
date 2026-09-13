# 🎙️ whisper-api

<!-- markdownlint-disable MD033 -->

<div align="center">

**OpenAI-compatible Whisper transcription server — powered by [faster-whisper](https://github.com/SYSTRAN/faster-whisper)**

Drop-in replacement for OpenAI's `/v1/audio/*` endpoints. Runs entirely on **CPU** with **INT8** quantization.

[![Docker](https://img.shields.io/badge/docker-ghcr.io-2496ED?logo=docker&logoColor=white)](.github/workflows/docker-publish.yml)
[![Python](https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white)](pyproject.toml)
[![uv](https://img.shields.io/badge/managed%20by-uv-DE5FE9)](https://docs.astral.sh/uv/)
[![CPU only](https://img.shields.io/badge/CPU-only-green)](https://openai.com/research/whisper)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

</div>

---

## ✨ Features

- 🔌 **OpenAI-compatible** — point any OpenAI SDK / client at this server and transcribe audio
- ⚡ **faster-whisper** — CTranslate2-backed Whisper: 4× faster than openai/whisper on CPU
- 🪶 **INT8 by default** — the `base` model quantized to 8-bit ints: ~2× faster than fp32, negligible quality loss
- 💻 **CPU only** — no CUDA, no torch, no NVIDIA layers; runs anywhere x86/ARM Docker runs
- 📦 **uv-managed** — reproducible, lockfile-pinned installs with [uv](https://docs.astral.sh/uv/)
- 🐳 **Docker-ready** — multi-stage image, non-root, healthcheck, persistent model cache
- 🔄 **CI/CD** — GitHub Actions publishes to GHCR on every push to `main` and on version tags

## 🗺️ Endpoints

| Method | Path | Description |
| :------: | ------ | ------------- |
| `POST` | `/v1/audio/transcriptions` | Transcribe audio into its spoken language |
| `POST` | `/v1/audio/translations` | Translate audio into English |
| `GET` | `/v1/models` | List served models (OpenAI-compatible) |
| `GET` | `/health` | Liveness / readiness probe |
| `GET` | `/docs` | Interactive Swagger UI |

## 🚀 Quick Start

### Option 1 — Docker Compose (recommended)

```bash
docker compose up -d --build
```

> ℹ️ The first startup downloads the `base` model (~140 MB int8) into the
> `whisper-data` volume — subsequent restarts are instant.

### Option 2 — Docker (one-liner)

```bash
docker run --rm -p 8080:8080 -v whisper-data:/data ghcr.io/<owner>/whisper-api:latest
```

### Option 3 — Local with uv

```bash
uv sync          # creates .venv from uv.lock
uv run python app.py
```

> 💡 No uv? `pip install -r requirements.txt && python app.py` works too.

### Verify it's alive

```bash
curl http://localhost:8080/health
# {"status":"ok","model":"base","compute_type":"int8"}

curl http://localhost:8080/v1/models
```

## 📡 API Usage

### Transcribe

```bash
curl -X POST http://localhost:8080/v1/audio/transcriptions \
  -F "file=@meeting.mp3" \
  -F "language=en" \
  -F "response_format=json"
```

<details>
<summary><b>Response</b></summary>

```json
{
  "text": "Welcome to the meeting. Let's get started.",
  "task": "transcribe",
  "language": "en",
  "segments": [
    {
      "id": 0,
      "start": 0.0,
      "end": 3.42,
      "text": " Welcome to the meeting.",
      "no_speech_prob": 0.017,
      "avg_logprob": -0.31,
      "compression_ratio": 1.4,
      "...": "..."
    }
  ]
}
```

</details>

### Translate (any language → English)

```bash
curl -X POST http://localhost:8080/v1/audio/translations \
  -F "file=@entrevista.flac" \
  -F "response_format=text"
```

### From the OpenAI Python SDK

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8080/v1", api_key="unused")

with open("meeting.mp3", "rb") as f:
    result = client.audio.transcriptions.create(model="whisper-1", file=f)

print(result.text)
```

### Supported audio formats

`.wav` · `.mp3` · `.flac` · `.ogg` · `.m4a` — anything ffmpeg/PyAV can decode.

### `response_format` options

| Value | Output |
| ------- | -------- |
| `json` *(default)* | JSON with `text` + timestamped `segments` |
| `text` | Plain text |
| `verbose_json` | Full JSON with all segment details |
| `srt` | SubRip subtitles |
| `vtt` | WebVTT subtitles |

## ⚙️ Configuration (env vars)

| Variable | Default | Description |
| ---------- | --------- | ------------- |
| `WHISPER_MODEL_NAME` | `base` | Model size: `tiny`, `base`, `small`, `medium`, `large-v3` |
| `WHISPER_COMPUTE_TYPE` | `int8` | Quantization: `int8`, `int4`, `float16`, `float32` |
| `WHISPER_DEVICE` | `cpu` | Fixed to CPU |
| `WHISPER_HOST` | `0.0.0.0` | Bind host |
| `WHISPER_PORT` | `8080` | Bind port |
| `WHISPER_LAZY_LOAD` | `false` | Defer model load until the first request |
| `HF_HOME` | `/data/hf` (container) | Where model weights are downloaded/cached |

> 📌 The `model` field in requests is accepted for OpenAI compatibility but
> ignored — the server always serves the configured model.

### Why `base` + `int8`?

| Model | Params | VRAM/RAM (int8) | Relative speed |
| ------- | -------- | ----------------- | :--------------: |
| `tiny` | 39 M | ~0.1 GB | 🚀🚀🚀🚀 |
| **`base`** | **74 M** | **~0.15 GB** | 🚀🚀🚀 |
| `small` | 244 M | ~0.3 GB | 🚀🚀 |
| `medium` | 769 M | ~0.8 GB | 🚀 |
| `large-v3` | 1550 M | ~1.6 GB | 🐢 |

`base` is the sweet spot for CPU serving: good accuracy across languages at a
fraction of the footprint, and `int8` roughly halves memory again with ~2×
speedup over fp32 (via FBGEMM on x86 / AVX2).

## 🏗️ Architecture

```text
┌─────────────┐   multipart/form-data   ┌──────────────────┐   CTranslate2    ┌────────────┐
│ OpenAI client│ ─────────────────────▶ │  FastAPI (uvicorn)│ ───────────────▶ │ faster-    │
│ / SDK / curl │                        │  /v1/audio/*      │   CPU · int8     │ whisper    │
└─────────────┘                         └──────────────────┘                  └────────────┘
                                              │
                                              ▼
                                        temp file → transcribe → segments → OpenAI JSON
```

```text
.
├── app.py                        # FastAPI app: routes, settings, model lifecycle
├── tests/
│   └── e2e_test.py               # Full end-to-end suite (real server, real model)
├── pyproject.toml                # uv project definition + dependencies
├── uv.lock                       # Pinned dependency lockfile
├── requirements.txt              # pip fallback
├── deploy/
│   └── Dockerfile                # Multi-stage, uv-based, CPU-only image
├── docker-compose.yml            # Local dev / self-host stack
└── .github/workflows/
    └── docker-publish.yml        # GHCR publish on main + tags (with provenance)
```

## 🐳 Production notes

- **Model cache** — mount `/data` (HF weights persist; restarts don't re-download)
- **Healthcheck** — built-in `HEALTHCHECK` hits `/health` every 30 s
- **Non-root** — container runs as uid `10001`
- **Scaling** — the model is loaded per process; scale horizontally with `docker compose --scale whisper=N` behind a reverse proxy
- **Provenance** — published images carry SLSA build attestations
- **Security** — no API key enforcement is built in; put an auth proxy (e.g. Traefik forward-auth, nginx) in front for public deployments

## 🛠️ Development

```bash
uv sync                              # set up the environment
uv run python app.py                 # start the dev server
uv run python -c "import app"        # smoke check
```

### Testing

```bash
uv run pytest tests/e2e_test.py -v
```

The e2e suite boots the **real server** (real faster-whisper model, real
inference — no mocks) on a random port and exercises every endpoint over HTTP:

- `/health`, `/v1/models`, `/openapi.json`
- transcription in all 5 response formats (`json`, `text`, `srt`, `vtt`, `verbose_json`)
- transcript content verified against a known speech sample (OpenAI's `jfk.flac`)
- translation endpoint (both JSON and text formats)
- error paths: missing file → `422`, invalid `response_format` → `400`, empty upload rejected without crashing
- `model` parameter accepted and ignored (OpenAI SDK compatibility)

> First run downloads the speech sample (~1 MB) and the `base` model (~140 MB)
> if not already cached; subsequent runs complete in ~20 s.

Update dependencies:

```bash
uv lock --upgrade
```

## 📄 License

MIT
