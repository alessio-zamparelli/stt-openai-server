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
# {"status":"ok","model":"base","compute_type":"int8",
#  "loaded":true,"idle_unload_s":300,"last_request_age_s":0.0,
#  "unloads":0,"reloads":1}

curl http://localhost:8080/v1/models
```

The `loaded`, `unloads`, `reloads` and `last_request_age_s` fields expose the
[Idle memory](/docs/PLAN-idle-unload.md) lifecycle. Health probes deliberately do
**not** reset the idle timer, so monitoring traffic never defeats eviction.

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
| `WHISPER_IDLE_UNLOAD_S` | `300` | Evict the model from RAM after this many seconds with no API requests (`0` disables; next request reloads it ~0.5s warm). See [Idle memory](/docs/PLAN-idle-unload.md) |
| `WHISPER_IDLE_POLL_S` | `30` | How often the watchdog re-checks the idle window |
| `WHISPER_LANGUAGE` | *(auto)* | Default language when a request omits `language` (ISO-639-1 code, e.g. `it`). Pinning stabilizes short / low-quality clips |
| `WHISPER_LANGUAGES` | *(auto)* | Comma-separated allowlist, e.g. `it,en`. A single code is forced for every request; several codes run auto-detection *constrained* to that set (whisper's default detection ignores any allowlist) |
| `WHISPER_INITIAL_PROMPT` | *(none)* | Prompt applied whenever a request sends none (conditions the decoder, e.g. `The transcript is:`) |
| `WHISPER_MAX_CONCURRENT` | `2` | Cap on simultaneous in-flight transcriptions. `>0` runs inference in a bounded threadpool (backpressures excess requests); `0` = unbounded. Inference offloads the event loop, so `/health` stays responsive under load. See [Robustness & concurrency](/docs/PLAN-robustness-concurrency.md) |
| `WHISPER_MAX_UPLOAD_MB` | `100` | Reject uploads whose byte size exceeds this (`413`, mid-stream; `0` = unlimited). Streams to disk in 1 MB chunks so RAM stays flat |
| `WHISPER_MAX_AUDIO_SECONDS` | `3600` | Reject audio longer than this in container duration — OpenAI-style `400` *before* the model runs (`0` = off). Whisper processes audio in fixed ~30 s windows, so very long inputs cost multiples of the per-window floor |
| `WHISPER_REQUEST_TIMEOUT_S` | `300` | Abandon-the-call timeout → `504` if inference exceeds it (`0` = off). CT2 can't be preempted, so the orphan drains in the bounded pool |
| `WHISPER_BEAM_SIZE` | `5` | Beam width. Accuracy-first default; `2` ≈ 2× faster on multi-window clips but parity is run-dependent (see [Latency tuning](#latency-tuning)) |
| `WHISPER_BEST_OF` | `5` | Candidates sampled when `temperature > 0`. Pair with `WHISPER_BEAM_SIZE` for latency tuning |
| `WHISPER_TEMPERATURES` | *(fw default)* | Comma-separated fallback schedule, e.g. `0,0.2,0.4`. Trims whisper's default `[0…1.0]` ramp (a hidden multiplier) and overrides the per-request `temperature` scalar when set. Bad values fall back to the default schedule (no 500) |
| `WHISPER_CPU_THREADS` | *(fw default)* | CT2 intra-op threads. `0` = faster-whisper default. Rule of thumb: `cores / WHISPER_MAX_CONCURRENT` — don't oversubscribe |
| `HF_HOME` | `/data/hf` (container) | Where model weights are downloaded/cached |

> 💡 **Priority for `language`**: per-request `language` field → `WHISPER_LANGUAGE`
> → `WHISPER_LANGUAGES` (single forced / multi constrained-detect) → faster-whisper
> full auto-detect.

> 📌 The `model` field in requests is accepted for OpenAI compatibility but
> ignored — the server always serves the configured model.

### Latency tuning

Measured on the reference hardware (`base`/int8, CPU-only; jfk ≈ 11 s, "long" = jfk×3 ≈ 33 s). Whisper processes fixed ~30 s windows, so the gains compound on multi-window audio:

| Config | 11 s | 33 s | Notes |
| --- | --- | --- | --- |
| default `beam=5/best=5` | 1.37 s | 8.35 s | accuracy-first baseline |
| `beam=2/best=2` | 1.20 s | 4.23 s | ≈2× on long clips |
| beam2 + `WHISPER_TEMPERATURES=0,0.2,0.4` | 1.20 s | 3.02 s | ≈2.8×; trims fallback passes |
| Batched bs4 beam2 | — | 1.95 s | ≈4.3×; opt-in (see `WHISPER_BATCH_SIZE`) |
| beam2 + VAD | — | 3.36 s | VAD is a *cost* on all-speech clips |
| `cpu_threads` 1 / 2 / 4 | — | 9.31 / 5.86 / 5.51 s | saturates ≈ cores |

📌 **Defaults stay accuracy-first.** An accuracy gate (jfk + Italian TTS clips)
showed beam2 text parity is *run-dependent* — e.g. `grazie mille per il tuo aiuto`
degraded to `per il tuo aiuto` on one utterance — so the server ships `beam=5/best=5`
by default and these knobs are **opt-in** for latency-sensitive workloads. Treat the
numbers as *ratios*, not absolutes: machine variance is large, re-measure on your own
hardware.

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

### 🎯 Getting accurate transcriptions (short audio)

Measured against real TTS clips (0.3–2 s), the biggest accuracy lever is
**audio length, not model size**:

- **Pin the language.** `language="it"` / `WHISPER_LANGUAGE=it` reliably beats
  auto-detect on short clips (auto-language-ID misfires below ~1 s). A
  multilingual deployment can restrict detection with `WHISPER_LANGUAGES=it,en`.
- **Longer utterances transcribe much better.** Isolated words were
  ~30–60% exact over repeated trials; full sentences/phrases hit ~100% on the
  same `base` model. If you control input, transcribe a longer span.
- **Model size fixes the failure *mode*, not the floor.** `base` falls into
  repetition loops ("non si si si…") on sub-second clips; `small`/`medium`
  avoid that and nailed clean words, but none of them recover a 0.2 s clip.
- **"The user says:" as a text prompt hurts** (it conditions on mismatched
  context, often yielding empty output). Spoken carrier-sentence context
  (TTS reads "l'utente dice forse") *does* help — but only when it genuinely
  lengthens the audio, which TTS compressors often refuse to do.
- Speed-stretching the clip (`atempo 0.8`) was a wash (~−8% net) once TTS
  clip variance is averaged in.

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
