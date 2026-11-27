# stt-api

**OpenAI-compatible speech-to-text server powered by [faster-whisper](https://github.com/SYSTRAN/faster-whisper).**

`stt-api` exposes `/v1/audio/transcriptions` and `/v1/audio/translations`
with the same request shapes OpenAI clients already use. It runs on CPU
through CTranslate2 by default, defaults to the `base` model with INT8
quantization, and ships with a small production-oriented FastAPI server: bounded
concurrency, buffered uploads, duration/size guards, optional SSE streaming,
per-stage latency metrics, and idle model eviction. An opt-in CUDA image
(`deploy/Dockerfile.gpu`) runs the same server on an NVIDIA GPU.

- CPU by default — the default image has no CUDA, no PyTorch, no NVIDIA layers.
  The GPU variant is a separate image and stays opt-in.
- OpenAI-compatible transcription, translation, and model listing.
- Optional Wyoming ASR listener for Home Assistant (with mDNS discovery).
- One resident model per process, with optional eviction after an idle window.
- Docker image built with `uv`, run as a non-root user, with a persistent model
  cache and a built-in healthcheck.

> **Renamed:** environment variables now use the `STT_*` prefix. The historical
> `WHISPER_*` names still work as deprecated aliases for one release, and
> `STT_*` wins when both are set. The engine is still faster-whisper.

---

## Contents

- [What it is](#what-it-is)
- [Quick start](#quick-start)
- [API reference](#api-reference)
  - [Transcriptions](#post-v1audiotranscriptions)
  - [Translations](#post-v1audiotranslations)
  - [Streaming responses](#streaming-responses-streamtrue)
  - [Models, health, and docs](#get-v1models--get-health--get-docs)
  - [Response formats](#response-formats)
  - [Errors](#errors)
- [Wyoming protocol (Home Assistant)](#wyoming-protocol-home-assistant)
- [Architecture](#architecture)
  - [Request lifecycle](#request-lifecycle)
  - [Model lifecycle and idle eviction](#model-lifecycle-and-idle-eviction)
  - [Concurrency and backpressure](#concurrency-and-backpressure)
  - [Language resolution](#language-resolution)
  - [Process lifecycle](#process-lifecycle)
  - [Repository layout](#repository-layout)
- [Configuration](#-configuration)
- [Tuning and performance](#tuning-and-performance)
- [Docker](#docker)
  - [Dockerfile walkthrough](#dockerfile-walkthrough)
  - [GPU image (CUDA)](#gpu-image-cuda)
  - [docker-compose.yml walkthrough](#docker-composeyml-walkthrough)
  - [Building and running](#building-and-running)
- [Production notes](#production-notes)
- [Development and testing](#development-and-testing)
- [Limitations](#limitations)
- [License](#license)

---

## What it is

| Capability | Behaviour |
| --- | --- |
| Speech-to-text | `POST /v1/audio/transcriptions` in the spoken language |
| Speech translation | `POST /v1/audio/translations` into English |
| Output formats | `json`, `text`, `srt`, `vtt`, `verbose_json` |
| Streaming | OpenAI-style SSE for both audio endpoints (`stream=true`) |
| Wyoming ASR | Optional TCP listener (`:10300`) for Home Assistant, with optional zeroconf |
| Model selection | Server-owned; request `model` is accepted and ignored |
| Language control | Per request, server default, constrained allowlist, or auto-detect |
| Decoding controls | Beam width, `best_of`, temperature schedule, VAD, batching, CPU threads |
| Resource guards | Upload bytes, audio duration, inference timeout, concurrency cap |
| Memory control | Optional idle eviction with single-flight warm reload |
| Observability | `/health` exposes lifecycle state and rolling p50/p95 stage timings |
| Authentication | None built in; put an auth proxy in front for public deployments |

The server uses **one shared `WhisperModel` instance**. faster-whisper /
CTranslate2 is not safe for concurrent `transcribe()` calls on one instance, so
decodes are serialized with a lock. `STT_MAX_CONCURRENT` bounds the request
queue and keeps setup work off the event loop; it does not create parallel
decoders on the same model.

---

## Quick start

### Docker Compose (recommended)

```bash
docker compose up -d --build
```

Then verify:

```bash
curl http://localhost:8080/health
curl http://localhost:8080/v1/models
```

The first start downloads the `base` model (~140 MB INT8) into the
`stt-data` volume. Later starts reuse it.

### Docker one-liner

```bash
docker run --rm -p 8080:8080 -p 10300:10300 \
  -e STT_WYOMING=true \
  -v stt-data:/data \
  ghcr.io/<owner>/stt-openai-server:latest
```

Replace `<owner>` with the GitHub owner that published the image through the
workflow in `.github/workflows/docker-publish.yml`. Drop the `10300` mapping and
`STT_WYOMING=true` if you only need the HTTP API.

The CUDA build is published to the same package with a `-cuda` suffix:
`:latest-cuda` from `main`, and `:<version>-cuda` (e.g. `:0.2.0-cuda`) from a
`v*.*.*` tag. Run it with `--gpus all` — see [GPU image (CUDA)](#gpu-image-cuda).

### Local with `uv`

```bash
uv sync --locked
uv run python app.py
```

No `uv`? The pip fallback is:

```bash
pip install -r requirements.txt
python app.py
```

The app lives in the `stt_api/` package; `app.py` is a compatibility launcher
that keeps `python app.py` and `--healthcheck` working. You can also run
`uv run uvicorn stt_api.main:app` or `uv run stt-api`. With `package = true` in
`pyproject.toml`, the package is installed into the repo-local `.venv` by
`uv sync`.

### First startup behaviour

The model is loaded eagerly inside FastAPI's lifespan startup, **before uvicorn
binds the port**, unless `STT_LAZY_LOAD=true`. During a cold Docker start:

- the container stays in Docker's `starting` health state;
- local health probes may log `connection refused`;
- the built-in healthcheck allows a 10-minute start period.

Set `STT_LAZY_LOAD=true` if you want the HTTP server to bind immediately and
load the model on the first API request instead. With idle eviction enabled, the
same warm reload path is used after eviction.

---

## API reference

All audio endpoints consume `multipart/form-data`. The scalar fields are
declared as form fields, not query parameters, so send them with `-F` in curl or
`data=` in an HTTP client. The Pydantic request models in
`stt_api/schemas.py` document the OpenAI shape but the live endpoints require
multipart uploads.

### `POST /v1/audio/transcriptions`

Transcribe audio into its spoken language.

```bash
curl -X POST http://localhost:8080/v1/audio/transcriptions \
  -F "file=@meeting.mp3" \
  -F "language=en" \
  -F "response_format=json"
```

#### Form fields

| Field | Type | Default | Meaning |
| --- | --- | --- | --- |
| `file` | file | required | Audio upload. The field name must be `file`. |
| `model` | string | — | Accepted for OpenAI compatibility and **ignored**. The served model is always `STT_MODEL_NAME`. |
| `language` | ISO-639-1 string | server default | Source language hint, e.g. `en`, `it`. Empty/whitespace counts as absent. |
| `prompt` | string | `STT_INITIAL_PROMPT` | Decoder prompt. The per-request value wins over the server default. |
| `response_format` | string | `json` | `json`, `text`, `srt`, `vtt`, or `verbose_json`. |
| `temperature` | float, 0–1 | `0.0` | Sampling temperature. Ignored when `STT_TEMPERATURES` is set. |
| `timestamps` | bool | `true` | When false, JSON responses omit `segments` (`segments: null`). |
| `stream` | bool | `false` | Return SSE instead of the chosen response format. See [Streaming](#streaming-responses-streamtrue). |

`language` is validated against the loaded model's supported language set and
returns `400` for an unsupported code. The full resolution order is documented
in [Language resolution](#language-resolution).

#### JSON response

```json
{
  "text": "Welcome to the meeting. Let's get started.",
  "task": "transcribe",
  "language": "en",
  "segments": [
    {
      "id": 0,
      "seek": 0,
      "start": 0.0,
      "end": 3.42,
      "text": " Welcome to the meeting.",
      "tokens": [50364, 4066, 264, 3022, 13],
      "temperature": 0.0,
      "avg_logprob": -0.31,
      "compression_ratio": 1.4,
      "no_speech_prob": 0.017
    }
  ]
}
```

`text` is the concatenation of the segment texts, stripped. `segments` is a
list of the full faster-whisper segment dictionaries (`id`, `seek`, `start`,
`end`, `text`, `tokens`, `temperature`, `avg_logprob`, `compression_ratio`,
`no_speech_prob`).

### `POST /v1/audio/translations`

Translate audio into English. The shape is the same as transcription, but the
task is `translate` and there is no request `language` field — the server's
language defaults/allowlist still steer source-language detection.

```bash
curl -X POST http://localhost:8080/v1/audio/translations \
  -F "file=@entrevista.flac" \
  -F "response_format=text"
```

Accepted fields: `file`, `model` (ignored), `prompt`, `response_format`,
`temperature`, `timestamps`, `stream`.

The response uses `"task": "translate"` and `"language"` reports the detected
source language.

### Streaming responses (`stream=true`)

Both audio endpoints support SSE. When `stream=true` the endpoint returns
`text/event-stream` with `Cache-Control: no-cache` and forwards faster-whisper
segments as they are decoded. `response_format` is validated first, but the
stream itself is always SSE — it does not produce `text`, `srt`, `vtt`, or a
single JSON document.

Wire format:

```text
event: transcript
data: {"delta": " Hello", "timestamp": {"start": 0.0, "end": 1.2}}

event: transcript
data: {"delta": " world.", "timestamp": {"start": 1.2, "end": 2.1}}

event: transcript.done
data: {"text": "Hello world.", "language": "en", "task": "transcribe"}

event: done
data: [DONE]
```

On a backend error the stream emits:

```text
event: error
data: {"error": "'<repr of exception>'"}
```

Notes:

- The producer thread holds the same inference serialization lock as the
  non-streaming path, so only one decode runs on the model at a time.
- The buffered upload is owned by the stream generator and is closed only after
  the producer drains, including client disconnects.
- The timeout (`STT_REQUEST_TIMEOUT_S`) applies to the non-streaming call.
  A streaming response is drained/closed by the generator lifecycle.
- SSE parity with a specific OpenAI SDK stream consumer is best-effort; the
  events above are the server's own contract.

### `GET /v1/models` · `GET /health` · `GET /docs`

| Endpoint | Purpose |
| --- | --- |
| `GET /v1/models` | OpenAI-shaped list with one model: the configured `STT_MODEL_NAME`. |
| `GET /health` | Liveness, effective configuration, lifecycle counters, and rolling stage timings. |
| `GET /docs` | Interactive Swagger UI generated by FastAPI. |
| `GET /openapi.json` | OpenAPI schema. |

`GET /v1/models` response:

```json
{
  "object": "list",
  "data": [
    {
      "id": "base",
      "object": "model",
      "created": 1730000000,
      "owned_by": "faster-whisper"
    }
  ]
}
```

`GET /health` response (abridged; `latency_ms` contains only stages that have
samples):

```json
{
  "status": "ok",
  "model": "base",
  "compute_type": "int8",
  "loaded": true,
  "idle_unload_s": 300,
  "last_request_age_s": 0.0,
  "unloads": 0,
  "reloads": 1,
  "default_language": null,
  "allowed_languages": null,
  "max_concurrent": 2,
  "max_upload_mb": 100,
  "max_audio_seconds": 3600,
  "request_timeout_s": 300.0,
  "beam_size": 5,
  "best_of": 5,
  "temperature_schedule": null,
  "cpu_threads": 0,
  "batch_size": 0,
  "batch_min_seconds": 0,
  "vad_filter": false,
  "hf_offline": false,
  "wyoming": {
    "enabled": true,
    "host": "0.0.0.0",
    "port": 10300,
    "zeroconf": false,
    "name": null,
    "requests": 12,
    "errors": 0,
    "active": 0
  },
  "latency_ms": {
    "upload": {"n": 10, "p50_ms": 0.1, "p95_ms": 0.3},
    "duration": {"n": 10, "p50_ms": 0.8, "p95_ms": 1.5},
    "language": {"n": 10, "p50_ms": 1.2, "p95_ms": 2.0},
    "inference": {"n": 10, "p50_ms": 1370.2, "p95_ms": 2200.0},
    "total": {"n": 10, "p50_ms": 1400.0, "p95_ms": 2300.0},
    "wyoming": {"n": 4, "p50_ms": 1420.0, "p95_ms": 2100.0}
  }
}
```

`status` is always `"ok"`; `loaded` is false after an idle eviction (the next
API request reloads the model). Health probes deliberately do **not** touch the
idle timer, so monitoring traffic cannot prevent eviction.

The rolling timing window is 200 samples per stage. `latency_ms` reports
nearest-rank `p50`/`p95` in milliseconds. Streaming requests record the setup
stages and the route total; inference completes after the response has started.

### Response formats

| `response_format` | Content type | Result |
| --- | --- | --- |
| `json` *(default)* | `application/json` | `{text, task, language, segments}` |
| `verbose_json` | `application/json` | Same shape as `json`; all segment details are returned. |
| `text` | `text/plain` | The transcript text only. |
| `srt` | `application/x-subrip` | SubRip subtitles, numbered from 1, `HH:MM:SS,mmm`. |
| `vtt` | `text/vtt` | WebVTT with a `WEBVTT` header and `HH:MM:SS.mmm`. |

`timestamps=false` only changes JSON/verbose JSON: `segments` becomes `null`.
Text and subtitle formats always use the decoded segments.

### Errors

The API uses FastAPI's standard error body (`{"detail": "..."}`) rather than
OpenAI's nested `{"error": {...}}` envelope.

| Status | Cause |
| --- | --- |
| `400` | Invalid `response_format`; unsupported `language`; audio longer than `STT_MAX_AUDIO_SECONDS`. |
| `413` | Upload larger than `STT_MAX_UPLOAD_MB`. Checked from `Content-Length` at route entry and from the buffered upload size before inference; neither can run before FastAPI parses the multipart body. |
| `422` | Missing/invalid required form field (for example no `file`, or a temperature outside 0–1). |
| `500` | Unexpected decode/inference failure. The detail is the exception string. |
| `504` | Inference exceeded `STT_REQUEST_TIMEOUT_S`. CTranslate2 cannot be preempted; the abandoned call drains in the bounded pool. |

An empty upload is rejected without crashing the process; the server remains
healthy afterwards.

### Client examples

#### OpenAI Python SDK

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8080/v1", api_key="unused")

with open("meeting.mp3", "rb") as f:
    result = client.audio.transcriptions.create(
        model="whisper-1",   # accepted and ignored by this server
        file=f,
        response_format="json",
    )
print(result.text)

with open("entrevista.flac", "rb") as f:
    translation = client.audio.translations.create(
        model="whisper-1",
        file=f,
        response_format="text",
    )
print(translation)
```

#### curl with subtitles

```bash
curl -X POST http://localhost:8080/v1/audio/transcriptions \
  -F "file=@lecture.m4a" \
  -F "response_format=srt" \
  -o lecture.srt
```

#### Supported containers

`.wav`, `.mp3`, `.flac`, `.m4a`, `.ogg`, `.opus`, `.aac`, `.wma`, `.webm`,
`.mp4`, `.avi`, `.mov`, `.mkv`, `.amr`, `.3gp`, `.aiff` are accepted. The
upload is handed to PyAV/libavformat exactly as received; no temp copy or
sanitized extension is created. PyAV sniffs the actual container from the
bytes, so the filename suffix is not part of the decoder contract and unknown
extensions still decode.

---

## Wyoming protocol (Home Assistant)

The server can expose a Wyoming ASR endpoint so Home Assistant can use it as a
speech-to-text provider. It is **opt-in** and shares the same model, idle
eviction, language rules, and inference serialization as the HTTP API.

### Enable it

```bash
docker run --rm -p 8080:8080 -p 10300:10300 \
  -e STT_WYOMING=true \
  -v stt-data:/data \
  ghcr.io/<owner>/stt-openai-server:latest
```

Local:

```bash
STT_WYOMING=true uv run python app.py
```

The listener binds `0.0.0.0:10300` by default. Configure it with
`STT_WYOMING_HOST`, `STT_WYOMING_PORT`, and `STT_WYOMING_NAME`.

### Add it to Home Assistant

Without discovery:

1. **Settings → Devices & services → Add integration → Wyoming Protocol**.
2. Enter the host running this server and port `10300`.

With mDNS auto-discovery, set `STT_WYOMING_ZEROCONF=true`. The service is
registered as `<STT_WYOMING_NAME or MAC>._wyoming._tcp.local.`. Docker bridge
networking often blocks mDNS; use `network_mode: host` or manual host/port.

### Event flow

```text
client → describe
client ← info            (asr program "stt-api", model = STT_MODEL_NAME)
client → transcribe      (optional language; name is ignored)
client → audio-start
client → audio-chunk*    (raw PCM)
client → audio-stop
client ← transcript      ({text, language})
                         (server then closes the connection)
```

### Behaviour and limits

- **One utterance per connection.** Home Assistant opens a connection per
  request; the server closes it after `transcript`, matching the reference
  `wyoming-faster-whisper` server.
- **Audio is normalized** to 16 kHz mono 16-bit PCM. Native Home Assistant
  chunks are already in that format, so the converter is a no-op.
- **Language**: the `transcribe.language` hint wins, then `STT_LANGUAGE`, then a
  multi-code `STT_LANGUAGES` allowlist, then auto-detect. `"auto"`, empty values,
  and region-qualified tags (`en-US`) are normalized. An unsupported code logs a
  warning and falls back instead of dropping the connection.
- **Model name is ignored.** Like the HTTP `model` field, `transcribe.name` is
  accepted for compatibility but the served model is always `STT_MODEL_NAME`.
- **VAD is external.** `info` advertises `requires_external_vad=true`; Home
  Assistant/its satellite detects the end of the command and sends
  `audio-stop`.
- **Limits** mirror the HTTP API: accumulated PCM is checked against
  `STT_MAX_AUDIO_SECONDS` and `STT_MAX_UPLOAD_MB` while chunks arrive. An
  oversized utterance returns `transcript` with an empty `text` and is logged.
- **Failures** (decode error, timeout, empty audio) also return an empty
  transcript; Wyoming has no error event. Check logs and `/health.wyoming.errors`.
- **No interim streaming.** `supports_transcript_streaming` is `false`; one final
  transcript is sent after `audio-stop` (the Home Assistant Whisper app behaves
  the same way).
- **Idle eviction**: `transcribe`/`audio-stop` count as API activity;
  `describe` does not, and `describe` never loads the model.
- **No authentication or encryption.** Wyoming is designed for a trusted
  network; do not expose port `10300` to the internet.

`/health.wyoming` reports the listener state and counters:

```json
"wyoming": {
  "enabled": true,
  "host": "0.0.0.0",
  "port": 10300,
  "zeroconf": false,
  "name": null,
  "requests": 12,
  "errors": 0,
  "active": 0
}
```

---

## Architecture

### High-level flow

```text
┌─────────────────┐   multipart/form-data    ┌──────────────────────────┐
│ OpenAI SDK /    │ ───────────────────────▶ │ FastAPI + uvicorn        │
│ curl / client   │                          │ /v1/audio/transcriptions │
└─────────────────┘                          │ /v1/audio/translations   │
                                             └────────────┬─────────────┘
┌─────────────────┐   Wyoming JSONL + PCM                 │
│ Home Assistant  │ ───────────────────────▶ :10300 ──────┤
└─────────────────┘                                      │
                     ┌────────────────────────────────────┼──────────────────────────────┐
                     │                                    │                              │
                     ▼                                    ▼                              ▼
          use buffered UploadFile                 resolve language / limits       bounded thread pool
          (Starlette SpooledTemporaryFile)        (PyAV metadata, allowlist)     (STT_MAX_CONCURRENT)
                     │                                    │                              │
                     └────────────────────────────────────┼──────────────────────────────┘
                                                          ▼
                                            one shared WhisperModel
                                            CTranslate2 · CPU INT8 (default)
                                            or CUDA FP16 (GPU image)
                                            inference serialized by _inference_lock
                                                          │
                                                          ▼
                                            HTTP: OpenAI JSON / text / SRT / VTT
                                            HTTP: SSE when stream=true
                                            Wyoming: transcript
```

The runtime is a FastAPI package (`stt_api/`). `main.py` owns the app and
lifespan; `routers/` contains the HTTP surface; `services/` contains model
lifecycle, audio/language/inference helpers, response formatting, and the
Wyoming listener; `runtime.py` owns the process-wide settings, model store,
inference pool, lock, and stage timings. `app.py` is a thin compatibility
launcher for `python app.py` and `--healthcheck`.

### Request lifecycle

Both audio routes follow the same sequence:

1. **Validate `response_format`** (`400` before any model work).
2. **Reject an oversized `Content-Length` fast** (`413`; FastAPI has already
   parsed the multipart body, so this avoids inference, not the body read).
3. **Mark API activity** with `store.touch()`.
4. **Capture the model** with `store.get()`. This reloads single-flight if the
   model was evicted. Capturing a local reference keeps the in-flight request
   safe if the watchdog clears the store concurrently.
5. **Use the buffered upload directly.** Starlette has already parsed the
   multipart body into an `UploadFile` backed by a `SpooledTemporaryFile`
   (1 MB in memory, then disk). The route passes that file-like object to
   PyAV/faster-whisper instead of copying it to a second temp file, and
   enforces `STT_MAX_UPLOAD_MB` from `UploadFile.size` before inference.
6. **Probe container duration** with PyAV metadata. No model load; over the
   `STT_MAX_AUDIO_SECONDS` cap → `400`. The same duration selects the
   runner when `STT_BATCH_MIN_SECONDS > 0`.
7. **Resolve the language** (client field → server default → allowlist →
   constrained detection → full auto). Multi-code allowlists decode only the
   first 30 s window, run faster-whisper's detector, then pick the
   highest-probability allowed code.
8. **Build decoding kwargs** from the server settings and request fields. The
   runner is constructed here: `BatchedInferencePipeline` when
   `STT_BATCH_SIZE > 0` and (if set) the clip meets
   `STT_BATCH_MIN_SECONDS`. Text-only output (`text` / `timestamps=false`)
   sets `without_timestamps` and skips building segment dictionaries.
9. **Run inference off the event loop** in the bounded pool. The shared-model
   serialization lock covers both `transcribe()` and the lazy segment
   iteration, because the decode happens while iterating segments.
10. **Format the response**:
    - text/SRT/VTT → `PlainTextResponse`;
    - JSON/verbose JSON → `TranscriptionResponse` / `TranslationResponse`.
11. **Record stage timings** and **close the upload** in `finally`. With
    `stream=true`, ownership moves to `_sse_stream`, which closes it after the
    producer drains. FastAPI also auto-closes form files at request end.

### Model lifecycle and idle eviction

`ModelStore` owns the resident `WhisperModel`:

| Method | Responsibility |
| --- | --- |
| `get()` | Return the model; if absent, invoke the loader under a lock. Concurrent callers share one rebuild. Increments `reloads`. |
| `touch()` | Record real API activity. Called only by audio routes. |
| `maybe_unload(now)` | If idle past `STT_IDLE_UNLOAD_S`, acquire the store lock non-blocking, clear the model, run `gc.collect()`, and best-effort `malloc_trim(0)`. Returns whether it evicted. |
| `last_request_age()` | Seconds since the last API request; exposed by `/health`. |

The lifespan starts a daemon watchdog only when eviction is enabled. The actual
poll interval is:

```text
min(STT_IDLE_POLL_S, max(5, STT_IDLE_UNLOAD_S // 2))
```

The measured behavior on the reference host (`base`/INT8, CPU) is roughly
`507 MB → 108 MB` after eviction, with a warm reload of about `0.5 s` from the
HF cache. Mount `/data` so that cache survives container restarts.

`STT_LAZY_LOAD` and `STT_IDLE_UNLOAD_S` compose:

- lazy load controls the **initial** load (startup vs first request);
- idle eviction controls **subsequent** drops;
- both on ⇒ load on first request, evict when idle, reload on the next request.

### Concurrency and backpressure

- Blocking chunks (`store.get`, duration probing, language detection,
  `transcribe` and segment iteration) run in a `ThreadPoolExecutor` when
  `STT_MAX_CONCURRENT > 0`. The pool is initialized eagerly in the lifespan
  and protected by double-checked locking.
- `STT_MAX_CONCURRENT=0` switches to Starlette/anyio's threadpool and is
  effectively unbounded. The default is `2`.
- `_inference_lock` serializes all decodes on the shared model. With
  `max_concurrent=2`, two requests can be prepared at once, but only one can
  decode; the other waits on the lock. Excess requests wait in the bounded
  pool/queue instead of racing the model.
- `/health` remains responsive while inference runs because the event loop is
  never blocked by CTranslate2 (it releases the GIL).
- `STT_REQUEST_TIMEOUT_S` abandons the wait and returns `504`; the worker
  thread and model decode continue until they finish. The pool bound prevents
  orphaned calls from growing without limit.
- There is no API-level rate limiter or queue-depth metric. Use a reverse proxy
  for rate limits and request timeouts.

Sizing guidance:

- Inference is serialized, so size `STT_CPU_THREADS` near the available
  physical cores (or leave `0` to let faster-whisper decide). More threads than
  cores oversubscribes the single decode.
- `STT_MAX_CONCURRENT` bounds queued/setup work, not parallel decodes.
- Batched mode (`STT_BATCH_SIZE > 0`) has a different CPU profile: batch
  size × intra-op threads is the footprint. For long files, a common starting
  point is `MAX_CONCURRENT=1`, `BATCH_SIZE=4`, `CPU_THREADS=2`, then measure.
- Run one resident model per host/CPU quota. Multiple STT containers on one
  CPU produce cold-start and queue contention; prefer fewer, larger instances or
  batched mode over many small ones.

### Language resolution

Priority, highest first:

1. Per-request `language` (transcriptions only). Empty/whitespace is treated as
   absent.
2. `STT_LANGUAGE` (`default_language`).
3. `STT_LANGUAGES` allowlist:
   - **one code** → forced for every request, no detection;
   - **multiple codes** → decode only the first 30 s window, run
     faster-whisper's detector, then choose the highest-probability code inside
     the allowlist. Plain auto-detect ignores the allowlist, so this constrained
     pass is done explicitly.
4. Faster-whisper full auto-detection.

Unsupported ISO codes produce a `400` with the supported set. The language is
also reported in JSON responses as the detected/effective language.

### Process lifecycle

`lifespan` startup:

1. Eagerly load the model unless `STT_LAZY_LOAD=true`.
2. Eagerly create the inference pool with the configured worker count.
3. Start the idle watchdog if `STT_IDLE_UNLOAD_S > 0`.
4. Start the Wyoming listener if `STT_WYOMING=true` (bind failure aborts
   startup; zeroconf failure only logs a warning).
5. Yield to uvicorn.

`lifespan` shutdown:

1. Stop the Wyoming listener and unregister zeroconf.
2. Set the watchdog stop event.
3. Close and drop the inference pool.

`python app.py --healthcheck` runs a one-shot `http.client` probe against
`/health`. It honors `STT_HOST`/`STT_PORT` (including a `0.0.0.0` bind),
bypasses `HTTP(S)_PROXY`, and returns a single diagnostic line instead of a
traceback while the model is still loading.

### Repository layout

```text
.
├── app.py                         # compatibility launcher (python app.py / --healthcheck)
├── stt_api/
│   ├── __init__.py                # version
│   ├── main.py                    # FastAPI app, lifespan, CLI
│   ├── config.py                  # Settings + STT_*/WHISPER_* env resolution
│   ├── runtime.py                 # settings/store, pool, lock, timings
│   ├── dependencies.py            # get_settings(), get_store()
│   ├── schemas.py                 # Pydantic request/response models
│   ├── routers/
│   │   ├── transcriptions.py      # POST /v1/audio/transcriptions
│   │   ├── translations.py        # POST /v1/audio/translations
│   │   └── system.py              # GET /v1/models, /health
│   └── services/
│       ├── model_store.py         # ModelStore, idle eviction
│       ├── audio.py               # limits, duration, prefix decode
│       ├── formatting.py          # JSON/text/SRT/VTT
│       ├── language.py            # language resolution
│       ├── inference.py           # kwargs, batching, SSE
│       └── wyoming.py             # Wyoming ASR listener + zeroconf
├── tests/
│   ├── test_config.py             # STT_* / legacy alias resolution
│   ├── test_idle_unload.py        # eviction/reload unit tests
│   ├── test_language.py           # language resolution tests
│   ├── test_plan_docs.py          # README/compose/package drift guard
│   ├── test_robustness.py         # limits, timeout, SSE, pool, serialization
│   ├── test_timings.py            # rolling stage statistics
│   ├── test_tuning.py             # decoding-kwarg plumbing
│   ├── test_wyoming.py            # Wyoming protocol unit/integration tests
│   ├── e2e_test.py                # real server, real model, all endpoints
│   ├── e2e_language_test.py       # language behaviour on a real model
│   ├── e2e_robustness_test.py     # resource guards against a real server
│   ├── e2e_wyoming_test.py        # Wyoming round-trip on a real model
│   └── e2e_performance_test.py    # opt-in wall-time assertions
├── docs/
│   ├── PLAN-idle-unload.md
│   ├── PLAN-performance.md
│   ├── PLAN-robustness-concurrency.md
│   └── PLAN-wyoming.md
├── deploy/
│   ├── Dockerfile                 # multi-stage CPU-only uv image
│   └── Dockerfile.gpu             # same image + CUDA 12 cuBLAS wheels
├── docker-compose.yml             # local build/run convenience
├── docker-compose.gpu.yml         # GPU overlay (device + float16 defaults)
├── pyproject.toml                 # uv package + pytest config
├── uv.lock                        # pinned dependency graph
├── requirements.txt               # pip fallback
└── .github/workflows/
    └── docker-publish.yml         # smoke gate + GHCR publish (CPU + -cuda)
```

Docker copies `app.py` and the `stt_api/` package into the runtime image. If you
add another runtime module or data file, update both `deploy/Dockerfile` and
`deploy/Dockerfile.gpu` or the containers will fail at import/startup.

---

## ⚙️ Configuration

Environment variables are read **once at import** into the module-global
`settings`. Changing `os.environ` after import has no effect; restart the
process (or construct `Settings(...)` in tests). Integers/floats with bad
values fall back to their defaults; booleans accept `1`, `true`, `yes`, `on`
(case-insensitive).

Canonical names use `STT_*`. Every knob also accepts the legacy `WHISPER_*`
spelling for one release; `STT_*` wins when both are set and a one-time warning
is logged for each legacy variable.

| Variable | Default | Description |
| --- | --- | --- |
| `STT_MODEL_NAME` | `base` | faster-whisper model size or local CTranslate2 model path. Tested sizes: `tiny`, `base`, `small`, `medium`, `large-v3`, `large-v3-turbo`. |
| `STT_COMPUTE_TYPE` | `int8` | CTranslate2 compute type passed to `WhisperModel`. Common CPU values: `int8`, `int8_float32`, `int16`, `float32`. GPU values (GPU image only): `float16` (default there), `int8_float16`, `bfloat16` (Ampere+). The CPU image has no CUDA libraries, so GPU types fail there. |
| `STT_DEVICE` | `cpu` | Device passed to faster-whisper: `cpu` (default image) or `cuda` (GPU image, `deploy/Dockerfile.gpu`). |
| `STT_HOST` | `0.0.0.0` | uvicorn bind host. |
| `STT_PORT` | `8080` | uvicorn bind port and healthcheck port. |
| `STT_LAZY_LOAD` | `false` | Defer the initial model load until the first API request. The server binds immediately. |
| `STT_IDLE_UNLOAD_S` | `300` | Evict the model from RAM after this many seconds without API requests. `0` disables eviction. Next API request reloads (~0.5 s warm cache). |
| `STT_IDLE_POLL_S` | `30` | Watchdog poll interval. The effective interval is capped at `max(5, idle_unload_s // 2)`. |
| `STT_LANGUAGE` | *(auto)* | Default language when a request omits `language`. ISO-639-1 code such as `it`. Pinning stabilizes short/low-quality clips. |
| `STT_LANGUAGES` | *(auto)* | Comma-separated allowlist, e.g. `it,en`. One code forces it; multiple codes constrain detection to that set (decoding only the first 30 s window). |
| `STT_INITIAL_PROMPT` | *(none)* | Decoder prompt used when a request sends none. The per-request `prompt` wins. |
| `STT_MAX_CONCURRENT` | `2` | Size of the bounded inference/setup thread pool. `0` = unbounded Starlette pool. Bounds queued work; decodes still serialize on one model. |
| `STT_MAX_UPLOAD_MB` | `100` | Reject uploads over this many MB with `413`, from `Content-Length` (route entry) and the buffered upload size, both before inference. `0` = unlimited. FastAPI parses the multipart body before the route runs, so an oversized body is already buffered when it is rejected. |
| `STT_MAX_AUDIO_SECONDS` | `3600` | Reject audio longer than this container duration with an OpenAI-style `400`, before the model runs. `0` = off. |
| `STT_REQUEST_TIMEOUT_S` | `300` | Abandon-the-call inference timeout → `504`. CTranslate2 cannot be preempted; the orphan drains in the pool. `0` = off. |
| `STT_BEAM_SIZE` | `5` | Beam width. Accuracy-first default; `2` is faster on multi-window clips but measurably worse on short phrases. See [Tuning](#tuning-and-performance). |
| `STT_BEST_OF` | `5` | Candidate sampling when `temperature > 0`. Inert at the default `temperature=0` (beam search uses `STT_BEAM_SIZE`). |
| `STT_TEMPERATURES` | *(faster-whisper default)* | Comma-separated fallback schedule, e.g. `0,0.2,0.4`. When set, it replaces the per-request `temperature` scalar. Bad values are ignored (default schedule). The `>0` fallback passes use an unseeded RNG, so borderline clips can vary run-to-run. |
| `STT_CPU_THREADS` | `0` | CTranslate2 intra-op threads. `0` = faster-whisper default. Since decodes are serialized, size near the core count; accuracy-neutral in measurements. |
| `STT_BATCH_SIZE` | `0` | `>0` enables `BatchedInferencePipeline`, which batches whisper's ~30 s windows (large speedups on long audio). Opt-in; batched mode ignores `condition_on_previous_text`, so casing/segmentation can drift. |
| `STT_BATCH_MIN_SECONDS` | `0` | With `STT_BATCH_SIZE > 0`, batch only clips at least this many seconds long (probed from container metadata). `0` = always batch; unknown duration falls back to sequential. |
| `STT_VAD` | `false` | Run Silero VAD pre-filtering in sequential mode. Helps sparse/quiet audio; on dense speech it adds a full VAD pass and is a net cost. Batched mode always runs VAD because `BatchedInferencePipeline` needs speech chunks to batch. |
| `STT_HF_OFFLINE` | `false` | Load with `local_files_only=True`; never contact the Hugging Face hub. Fails fast if weights are not already cached in `HF_HOME`. |
| `STT_WYOMING` | `false` | Enable the Wyoming ASR TCP listener for Home Assistant. |
| `STT_WYOMING_HOST` | `0.0.0.0` | Wyoming listener bind host. |
| `STT_WYOMING_PORT` | `10300` | Wyoming listener port. |
| `STT_WYOMING_ZEROCONF` | `false` | Register `_wyoming._tcp.local.` over mDNS for Home Assistant auto-discovery. |
| `STT_WYOMING_NAME` | *(MAC-derived)* | mDNS instance name. Set a unique friendly name when running multiple instances. |
| `HF_HOME` | `/data/hf` in Docker | Hugging Face model/weights cache. Mount it persistently. |
| `HF_HUB_OFFLINE` | *(unset)* | Library-level offline switch for `huggingface_hub`; `1` refuses network calls. Useful for air-gapped deployments/tests. faster-whisper also honors it. |

### Env template (all optional, commented)

Uncomment only what you need; values are examples.

```sh
# --- server / model ---
# STT_MODEL_NAME=base
# STT_COMPUTE_TYPE=int8
# STT_DEVICE=cpu
# GPU image (deploy/Dockerfile.gpu) instead uses CUDA; these are its defaults:
# STT_DEVICE=cuda
# STT_COMPUTE_TYPE=float16
# STT_HOST=0.0.0.0
# STT_PORT=8080
# STT_LAZY_LOAD=false
# STT_IDLE_UNLOAD_S=300
# STT_IDLE_POLL_S=30

# --- language / decoding ---
# STT_LANGUAGE=it
# STT_LANGUAGES=it,en
# STT_INITIAL_PROMPT=The transcript is:
# STT_BEAM_SIZE=5
# STT_BEST_OF=5
# STT_TEMPERATURES=0,0.2,0.4
# STT_CPU_THREADS=2
# STT_BATCH_SIZE=4
# STT_BATCH_MIN_SECONDS=30
# STT_VAD=true

# --- robustness / resources ---
# STT_MAX_CONCURRENT=2
# STT_MAX_UPLOAD_MB=100
# STT_MAX_AUDIO_SECONDS=3600
# STT_REQUEST_TIMEOUT_S=300

# --- Wyoming / Home Assistant ---
# STT_WYOMING=true
# STT_WYOMING_HOST=0.0.0.0
# STT_WYOMING_PORT=10300
# STT_WYOMING_ZEROCONF=true
# STT_WYOMING_NAME=stt-api

# --- cache / offline ---
# STT_HF_OFFLINE=true
# HF_HOME=/data/hf
# HF_HUB_OFFLINE=1
```

### Migrating from `whisper-api` / `WHISPER_*`

- Env vars: `STT_*` is canonical. `WHISPER_*` still works as a deprecated
  alias (one warning per variable); `STT_*` wins when both are set.
- Python imports: `from stt_api.config import Settings`,
  `import stt_api.runtime as runtime`. The old `app.settings` / `app.store`
  module attributes no longer exist; `app.py` re-exports only the FastAPI
  instance and `run()`.
- CLI: `stt-api` is the new console script; `whisper-api` remains as a
  one-release alias.
- Docker volume: `stt-data` replaces `whisper-data`. Copy the old cache to avoid
  re-downloading the model:

  ```bash
  docker volume create stt-data
  docker run --rm -v whisper-data:/from -v stt-data:/to alpine \
    sh -c 'cp -a /from/. /to/'
  ```

### Precedence and important semantics

- **`language` priority**: request field → `STT_LANGUAGE` →
  `STT_LANGUAGES` (single forced / multi constrained detection) →
  faster-whisper full auto-detect.
- **`prompt` priority**: request `prompt` → `STT_INITIAL_PROMPT`.
- **`temperature` priority**: `STT_TEMPERATURES` (when non-empty) overrides
  the request scalar. Otherwise the request `temperature` is used (default `0`).
- **`model` is ignored**. The server always serves `STT_MODEL_NAME`; this
  keeps OpenAI SDK calls that send `whisper-1` working.
- **`condition_on_previous_text=False`** is always set. Batched mode has its own
  windowing and does not support this option at all.
- **`STT_LANGUAGES` validation**: every allowlist entry is validated against
  the model's supported languages before detection. A bad code returns `400`.

---

## Tuning and performance

Measurements below are from the reference host (`base`/INT8, CPU only;
`jfk ≈ 11 s`, "long" = `jfk × 3 ≈ 33 s`). Whisper processes fixed ~30 s windows,
so gains compound on multi-window audio. Treat the numbers as **ratios**, not
universal constants — machine load changes absolute timings substantially. No
GPU benchmarks are published yet; the GPU image changes the backend, not the
window/batching behaviour these ratios describe.

### Latency

| Config | 11 s | 33 s | Notes |
| --- | ---: | ---: | --- |
| default `beam=5/best=5` | 1.37 s | 8.35 s | accuracy-first baseline |
| `beam=2/best=2` | 1.20 s | 4.23 s | ≈2× on long clips |
| beam2 + `STT_TEMPERATURES=0,0.2,0.4` | 1.20 s | 3.02 s | ≈2.8×; trims fallback passes |
| batched `batch_size=4`, beam2 | — | 1.95 s | ≈4.3×; opt-in |
| beam2 + VAD | — | 3.36 s | VAD is a cost on all-speech clips |
| `cpu_threads` 1 / 2 / 4 | — | 9.31 / 5.86 / 5.51 s | saturates near cores |

### Accuracy impact per knob

Reference corpus: `jfk` plus 9 short Italian TTS clips, `base`/INT8, CPU,
`lang=it`. Each row changes one knob from the `beam=5/best=5` defaults.

| Config | jfk | Short phrases | Sub-second words | Corpus time | Verdict |
| --- | --- | --- | --- | ---: | --- |
| default | 1/1 exact | 2/5 exact | 0/4 | 10.5 s | baseline |
| `beam=2` + `best_of=2` | 1/1 | **1/5** | 0/4 | 7.3 s | **worse** on short phrases; one hallucination loop |
| + `STT_TEMPERATURES=0,0.2,0.4` | 1/1 | 2/5¹ | 0/4 | 9.3 s | ≈ neutral; fallback can flake on borderline clips |
| `STT_CPU_THREADS=2` | 1/1 | 2/5 | 0/4 | 11.3 s | identity: byte-identical transcripts |
| `STT_CPU_THREADS=4` | 1/1 | 2/5 | 0/4 | 11.3 s | identity: byte-identical transcripts |
| `STT_BATCH_SIZE=4` | 1/1 | 2/5 | 0/4 | 7.8 s | no measured regression; one word improved |

¹ `come stai` sits at the fallback threshold: 5 of 6 passes returned the correct
text, one flaked.

Key takeaways:

- **`cpu_threads` is accuracy-neutral.** It changes parallelism only.
- **`beam=2` is measurably worse on short phrases.** Keep `beam=5` for quality.
- **A trimmed temperature schedule neither helps nor hurts the deterministic
  path.** Its only risk is the stochastic fallback on borderline clips.
- **Batched mode showed no accuracy regression on this corpus**, but its
  structural `condition_on_previous_text=False` caveat remains.
- **Sub-second isolated words score poorly in every config.** This is a
  model/corpus floor, not a knob regression. Pinning `STT_LANGUAGE` is the
  mitigation.

### 🧪 Reproduce the measured baseline

Each row maps to the commented preset blocks in `docker-compose.yml`. The
documentation drift-guard test (`tests/test_plan_docs.py`) asserts every knob
below appears in the configuration table and the compose file.

| Preset | Environment | vs. default (`base`/INT8, CPU, 33 s clip) |
| --- | --- | --- |
| default (accuracy-first) | *(none — server defaults)* | baseline 8.35 s |
| beam2 | `STT_BEAM_SIZE=2 STT_BEST_OF=2` | ≈2× faster |
| beam2 + trimmed schedule | + `STT_TEMPERATURES=0,0.2,0.4` | ≈2.8× faster |
| batched beam2 | + `STT_BATCH_SIZE=4` | ≈4.3× faster |
| adaptive batching | + `STT_BATCH_MIN_SECONDS=30` | ≈4.3× on clips ≥30 s; short clips stay sequential |
| VAD | + `STT_VAD=true` | cost on dense speech; helps silence-heavy audio |
| CPU threads | `STT_CPU_THREADS=1` / `2` / `4` | saturates near core count |

`STT_BATCH_MIN_SECONDS` makes batching duration-aware: the duration already
probed for the size guard selects the runner per request, so short clips keep
sequential decoding while long clips get the batched speedup. A mixed workload
can use `STT_BATCH_SIZE=4 STT_BATCH_MIN_SECONDS=30` without changing
short-clip output. If the duration cannot be probed, the request stays
sequential.

### Why `base` + `int8`?

| Model | Params | Approx. RAM (INT8) | Relative speed |
| --- | ---: | ---: | :---: |
| `tiny` | 39 M | ~0.1 GB | 🚀🚀🚀🚀 |
| **`base`** | **74 M** | **~0.15 GB** | 🚀🚀🚀 |
| `small` | 244 M | ~0.3 GB | 🚀🚀 |
| `medium` | 769 M | ~0.8 GB | 🚀 |
| `large-v3` | 1550 M | ~1.6 GB | 🐢 |

`base` is the CPU-serving sweet spot: usable multilingual accuracy at a small
footprint. INT8 roughly halves memory versus FP32 and uses optimized CPU kernels
(FBGEMM on x86 / AVX2 where available).

### Getting accurate short-audio transcriptions

- **Pin the language.** `language="it"` or `STT_LANGUAGE=it` reliably beats
  auto-detect below ~1 s. For multilingual deployments, constrain detection with
  `STT_LANGUAGES=it,en`.
- **Longer utterances help more than a bigger model.** Isolated words were
  ~30–60% exact over repeated trials; full phrases reached ~100% on `base`.
- **Model size changes the failure mode, not the floor.** `base` can enter
  repetition loops on sub-second clips; `small`/`medium` avoid that but still
  cannot recover a 0.2 s clip.
- **Text prompts can hurt.** A mismatched prompt such as `"The user says:"`
  often yields empty output. Spoken carrier-sentence context only helps when it
  genuinely lengthens the audio.
- **Speed-stretching was a wash.** `atempo 0.8` gained ~8% at best once clip
  variance was averaged.

### Stage timings

`/health.latency_ms` exposes rolling `upload`, `duration`, `language`,
`inference`, `total`, and `wyoming` statistics. This is the first place to look
if latency regresses: `upload` covers the post-parse size-validation step
(Starlette has already buffered the multipart body before the route runs, so it
is near zero), `duration` the PyAV probe, `language` constrained detection,
`inference` the decode configuration, and `wyoming` the end-to-end
audio-stop → transcript path. There is no Prometheus endpoint; scrape `/health`
or put it behind your monitoring proxy.

---

## Docker

### Dockerfile walkthrough

`deploy/Dockerfile` is a two-stage, CPU-only image following the
[uv Docker guide](https://docs.astral.sh/uv/guides/integration/docker/). The
opt-in GPU variant (`deploy/Dockerfile.gpu`) mirrors it and is described in
[GPU image (CUDA)](#gpu-image-cuda); the CPU image never installs CUDA.

#### Stage 1 — builder

```dockerfile
FROM python:3.14-slim AS builder
COPY --from=ghcr.io/astral-sh/uv:0.13.1 /uv /uvx /bin/
ENV UV_PYTHON_DOWNLOADS=0 \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1
WORKDIR /app
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    uv sync --locked --no-install-project --no-dev --no-editable
```

Details that matter:

- **uv 0.13.1** is copied from the official image, pinned to the version that
  generated `uv.lock`. Updating dependencies requires regenerating the lock and
  keeping the Docker pin compatible, otherwise `--locked` builds fail.
- **`UV_PYTHON_DOWNLOADS=0`** forces use of the system Python 3.14 in the slim
  image, keeping the interpreter path identical between builder and runtime.
- **`UV_LINK_MODE=copy`** avoids hardlink issues across image layers.
- **`UV_COMPILE_BYTECODE=1`** precompiles `.pyc` files for faster startup.
- **Bind-mounted manifests** keep `pyproject.toml` and `uv.lock` out of image
  layers; the dependency layer invalidates only when those files change.
- **`--no-install-project`** installs only dependencies into the image venv;
  `app.py` and `stt_api/` are copied in the runtime stage instead of building a
  wheel.
- **`--no-dev`** excludes pytest/httpx from the runtime venv.
- **`--no-editable`** produces a self-contained venv; no source tree is needed
  at runtime.

#### Stage 2 — runtime

```dockerfile
FROM python:3.14-slim AS runtime
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 ca-certificates \
    && rm -rf /var/lib/apt/lists/*
ENV PATH="/app/.venv/bin:$PATH" \
    UV_PYTHON_DOWNLOADS=0 \
    PYTHONUNBUFFERED=1 \
    STT_HOST=0.0.0.0 \
    STT_PORT=8080 \
    STT_MODEL_NAME=base \
    STT_COMPUTE_TYPE=int8 \
    HF_HOME=/data/hf
WORKDIR /app
COPY --from=builder /app/.venv /app/.venv
COPY app.py ./
COPY stt_api ./stt_api
RUN useradd --create-home --uid 10001 stt \
    && mkdir -p /data/hf \
    && chown -R stt:stt /data
VOLUME ["/data"]
USER stt
EXPOSE 8080 10300
HEALTHCHECK --interval=30s --timeout=5s --start-period=600s --retries=3 \
    CMD ["python", "app.py", "--healthcheck"]
ENTRYPOINT ["python", "app.py"]
```

Runtime details:

- **`libgomp1`** supplies the OpenMP runtime used by CTranslate2.
- **`ca-certificates`** allows the initial Hugging Face download.
- **`app.py` + `stt_api/` are copied.** There is no `tests/`, no `docs/`, no
  `src/`. Adding another runtime import requires updating the COPY lines.
- **Non-root**: uid `10001`, home `/home/stt`, with `/data` writable.
- **`/data` volume** holds `HF_HOME=/data/hf`, so weights persist across
  container recreation. A fresh volume starts cold and downloads the model.
- **Ports 8080 and 10300** are exposed; `10300` is only bound when
  `STT_WYOMING=true`.
- **Healthcheck** runs `python app.py --healthcheck` every 30 s, with a 5 s
  timeout, 3 retries, and a **600 s start period** because the eager model load
  happens before uvicorn binds.
- **Entrypoint** is the app itself; no shell wrapper.

No `ffmpeg` is installed: decoding uses PyAV's bundled libraries, and there is
no compressed-output encoder in this server.

### GPU image (CUDA)

`deploy/Dockerfile.gpu` builds the same server with CTranslate2's CUDA backend.
It is a separate image on purpose: the CPU image must stay free of CUDA
libraries, and the NVIDIA wheels add roughly 1 GB (cuBLAS + NVRTC).

**Host prerequisites** (the container ships the CUDA user-space libraries, so
the host needs no CUDA toolkit):

- NVIDIA driver **≥ 525.60.13** (the CUDA 12.x floor; the R535 branch works via
  minor-version compatibility); **≥ 550** recommended. `nvidia-smi` may report
  `CUDA Version: 12.x` or newer (e.g. 13.x): drivers are backward compatible
  with CUDA 12 user-space libraries.
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
  so `--gpus all` (or the Compose device reservation) works.

Keep the `cu12` wheel family even on a CUDA 13 driver: CTranslate2 4.8.2 is
built against CUDA 12.4 and `dlopen`s `libcublas.so.12`. Installing
`nvidia-cublas-cu13` would provide a different soname and would not be found.

Running the GPU image **without** a visible GPU fails at startup (the eager
model load happens before uvicorn binds); with `STT_LAZY_LOAD=true` it fails on
the first transcription instead.

**Build and run:**

```bash
docker build -f deploy/Dockerfile.gpu -t stt-api:gpu .

docker run --rm --gpus all -p 8080:8080 -v stt-data:/data stt-api:gpu

# or, with the overlay (keeps the base compose file CPU-default):
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --build
```

The publish workflow pushes this image to the same GHCR package as the CPU
image, suffixed with `-cuda`: `<version>-cuda` and `<major>.<minor>-cuda` on
`v*.*.*` tag pushes, `latest-cuda` on `main`.

**What the image adds:**

- `uv sync --extra gpu` pulls `nvidia-cublas-cu12` (declared in
  `pyproject.toml` under `[project.optional-dependencies]`; it brings
  `nvidia-cuda-nvrtc-cu12` with it). The dependency is linux/x86-64-only and is
  not installed into the CPU image.
- **No cuDNN.** CTranslate2 4.8.2 is built against CUDA 12.4 and loads cuBLAS at
  runtime; since 4.6.3, cuDNN is optional (conv1d has a pure CUDA
  implementation) and the wheel links no cuDNN at all. `nvidia-cudnn-cu12` is
  therefore deliberately absent — it would add ~1.7 GB for nothing.
- `ldconfig` is pointed at every `site-packages/nvidia/*/lib` directory, because
  CTranslate2 `dlopen`s `libcublas.so.12` / `libcublasLt.so.12` from there. The
  build fails early if those sonames or `libctranslate2`'s `NEEDED` libraries
  cannot be resolved.
- Defaults are `STT_DEVICE=cuda` and `STT_COMPUTE_TYPE=float16`; the app code is
  unchanged. `docker-compose.gpu.yml` reserves one NVIDIA device and sets the
  same values explicitly.

**Tuning on GPU:**

| Compute type | Notes |
| --- | --- |
| `float16` | Default. Best accuracy/speed balance on Turing and newer (and on Volta/Pascal datacenter cards). |
| `int8_float16` | Less VRAM, slightly lower accuracy; useful for `large-v3` on smaller cards. |
| `bfloat16` | Ampere (sm_80) or newer only. |
| `int8`, `float32` | Work on GPU but are slower / larger than `float16`; leave them to the CPU image. |

- **Decodes still serialize** on the single shared model (`_inference_lock`);
  `STT_MAX_CONCURRENT` bounds the queue, not parallel GPU work. The big GPU win
  is `STT_BATCH_SIZE`, which batches Whisper's ~30 s windows (opt-in, and it
  changes casing/segmentation like on CPU). `STT_CPU_THREADS` is inert under
  CTranslate2's CUDA backend.
- **VRAM** is roughly 1 GB for `base`/`small` FP16 and ~5 GB for `large-v3`
  FP16 (activations and batching included); `int8_float16` cuts that further.
  Idle eviction still works, but it frees VRAM only when the model object is
  collected — verify with `nvidia-smi` if you rely on it.
- **One model per GPU.** The server does not shard across devices; run one
  container per GPU (pin with `NVIDIA_VISIBLE_DEVICES`/Compose `device_ids`) or
  prefer a single larger instance.
- **CPU image unaffected.** `uv sync --locked ... --no-dev` without
  `--extra gpu` installs none of the NVIDIA wheels, so the CPU image size,
  startup, and tests are unchanged.

### `docker-compose.yml` walkthrough

```yaml
name: stt-api

services:
  stt:
    build:
      context: .
      dockerfile: deploy/Dockerfile
    image: stt-api:local
    ports:
      - "8080:8080"
      - "10300:10300"
    environment:
      - STT_HOST=0.0.0.0
      - STT_PORT=8080
      - STT_WYOMING=true
      - STT_WYOMING_HOST=0.0.0.0
      - STT_WYOMING_PORT=10300
      # ...commented optional knobs...
    volumes:
      - stt-data:/data
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "python", "app.py", "--healthcheck"]
      interval: 30s
      timeout: 5s
      start_period: 600s
      retries: 3

volumes:
  stt-data:
```

- **`name`** sets the Compose project name to `stt-api`.
- **`build`** builds `deploy/Dockerfile` from the repository root.
- **`image: stt-api:local`** tags the local build; GHCR publishing is done
  by the workflow, not by Compose.
- **Port mapping** `8080:8080` for HTTP and `10300:10300` for Wyoming; change
  the host side only if you also keep the container side at the configured
  `STT_PORT` / `STT_WYOMING_PORT`.
- **Active environment**: bind host/port plus the Wyoming listener. Every
  optional knob is present as a commented line, including the preset blocks.
  Compose comments are documentation; they are not passed to the container until
  uncommented.
- **Volume** `stt-data:/data` persists the HF cache (`/data/hf`). Without it
  every fresh container re-downloads the model.
- **`restart: unless-stopped`** keeps the service running across daemon restarts
  and crashes.
- **Healthcheck** duplicates the image healthcheck so `docker compose ps` shows
  health even if a deployment overrides the image.

The compose file's preset blocks mirror the README presets. Enable **one preset
at a time**, and remember that `STT_TEMPERATURES` overrides per-request
`temperature`.

### Building and running

```bash
# Build locally (CPU)
docker build -f deploy/Dockerfile -t stt-api .

# Build the GPU variant
docker build -f deploy/Dockerfile.gpu -t stt-api:gpu .

# Run with a persistent cache
docker run --rm -p 8080:8080 -p 10300:10300 -e STT_WYOMING=true \
  -v stt-data:/data stt-api

# Run on GPU
docker run --rm --gpus all -p 8080:8080 -v stt-data:/data stt-api:gpu

# Compose (CPU)
docker compose up -d --build
docker compose logs -f stt
docker compose down

# Compose (GPU overlay)
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --build

# Rebuild after dependency changes
uv lock --upgrade
docker compose build --no-cache
docker compose up -d
```

There are no Docker build args in this project. Model selection and all runtime
knobs are environment variables; the GPU variant is selected by Dockerfile, not
by a build argument.

---

## Production notes

- **One resident model per host/CPU quota.** Each loaded `base`/INT8 model holds
  roughly 450–500 MB while active; inference is serialized. Several containers
  on one box cause cold-start and queue contention. Prefer fewer, larger
  instances, or batched mode for long-file workloads.
- **GPU deployments.** Use `deploy/Dockerfile.gpu` (or the
  `docker-compose.gpu.yml` overlay) with driver ≥ 550 and the NVIDIA Container
  Toolkit. One container per GPU; size `STT_MODEL_NAME` to the card's VRAM
  (`large-v3` FP16 needs ~5 GB) and prefer `STT_BATCH_SIZE` over extra
  containers for throughput. `/health` does not report the device, so confirm
  `cuda` in the startup log.
- **Model cache.** Always mount `/data`. It is the difference between a ~0.5 s
  warm reload and a full download.
- **Healthcheck.** The built-in probe honors `STT_PORT`, bypasses proxies,
  and has a 10-minute cold-start grace. Do not shorten the start period unless
  the model is pre-cached in the image.
- **Idle eviction.** Default is 5 minutes. Lower it if RAM matters more than
  wake latency; set `0` for an always-hot service. Health probes never reset the
  idle window, so monitoring cannot defeat eviction.
- **Scaling.** Scale horizontally behind a reverse proxy
  (`docker compose --scale stt=N`) only if you have CPU capacity for N
  models. Sticky sessions are not required; every instance serves the same
  configured model. On GPU, one container per device instead of N per host.
- **Security.** There is no built-in auth or rate limiting. Put Traefik
  forward-auth, nginx, or an API gateway in front for public deployments. The
  server also returns exception strings in `500` details and exposes
  configuration in `/health`; do not expose it directly to untrusted clients.
  The Wyoming listener is equally unauthenticated — publish `10300` only on a
  trusted network, and do not expose it through the reverse proxy.
- **Wyoming discovery.** mDNS does not cross Docker bridge networking reliably;
  use `network_mode: host` or add the integration manually by host/port.
  `STT_WYOMING_ZEROCONF` registration failures are warnings, not startup
  failures.
- **Offline/air-gapped.** Pre-populate `/data/hf`, set `STT_HF_OFFLINE=true`
  and `HF_HUB_OFFLINE=1`, and verify the model loads without network access.
- **Provenance.** The publish workflow attaches SLSA build attestations to both
  the CPU and `-cuda` GHCR images.

---

## Development and testing

### Setup

```bash
uv sync --locked                     # repo-local .venv, including dev deps
uv run python app.py                 # dev server on :8080
uv run uvicorn stt_api.main:app      # ASGI entrypoint
uv run python -c "import stt_api.main"  # import smoke check; no model load
```

Use `uv run python -m pytest`, not bare `uv run pytest`: the current pytest
version does not always put the repository root on `sys.path`.

### Unit tests (fast, offline, fake model)

```bash
uv run python -m pytest tests/test_*.py -q
```

They cover:

- idle eviction/reload and health semantics;
- language precedence and constrained detection;
- upload/duration/timeout guards, SSE, pool initialization, inference locking;
- stage-timing statistics and decoding-kwarg plumbing;
- canonical `STT_*` env vars and legacy `WHISPER_*` aliases;
- the Wyoming protocol (describe/info, transcription, limits, zeroconf);
- a documentation drift guard for `STT_*` knobs.

### End-to-end tests (real model, real inference)

```bash
uv run python -m pytest tests/e2e_test.py -v
uv run python -m pytest tests/e2e_language_test.py -v
uv run python -m pytest tests/e2e_robustness_test.py -v
uv run python -m pytest tests/e2e_wyoming_test.py -v
```

Each e2e file boots a real `app.py` subprocess on a random port and waits up to
180 s for startup. The first run downloads the `base` model (~140 MB) and
OpenAI's `jfk.flac` sample (~1 MB); later runs use the cache (~20 s).

The functional e2e suite exercises all five response formats, translation,
model-field tolerance, missing files (`422`), invalid formats (`400`), empty
uploads, and `/openapi.json`.

Performance assertions are opt-in:

```bash
PERF_WALLTIME=1 uv run python -m pytest tests/e2e_performance_test.py -v
```

Run those only on quiet hardware; shared-host noise can reverse timing ratios.

### Documentation drift guard

`tests/test_plan_docs.py` treats three sources of truth for tuning: the
`stt_api/` package, the README configuration table (plus the README preset
table), and `docker-compose.yml`. If you add/rename a `STT_*` knob, update all of
them together. The test scans tokens anywhere in those sections, including
comments. `WHISPER_*` aliases are resolved generically in `config.py`, so they do
not need per-knob documentation.

### Docker/CI

- `.dockerignore` keeps `.venv`, `.git`, tests, compose, and caches out of the
  build context.
- `.github/workflows/docker-publish.yml` runs an import/OpenAPI smoke test, then
  builds and pushes **both** images to GHCR on pushes to `main` and `v*.*.*`
  tags: the CPU image (`latest`, `<version>`) and the CUDA image
  (`latest-cuda`, `<version>-cuda`). There is **no unit/e2e test job, no GPU
  inference test, and no PR trigger**; run the suites locally before pushing.
- The workflow publishes `linux/amd64` only, with GHA cache and SLSA
  attestations. Docker copies `app.py` + `stt_api/`, so adding a runtime module
  requires updating both Dockerfiles.

---

## Limitations

- **Serialized inference.** One shared model and one decode at a time. More
  concurrency means more queueing, not more parallel transcription. This holds
  on GPU too; use `STT_BATCH_SIZE` for throughput on long audio.
- **GPU support is a separate image.** `deploy/Dockerfile.gpu` adds the CUDA 12
  runtime (cuBLAS; no cuDNN is needed by this CTranslate2 build) and requires
  host driver ≥ 525.60.13 (recommended ≥ 550); the default image has no
  CUDA/torch and cannot run GPU compute types. `/health` reports
  `compute_type` but not `device`, so check the startup log to confirm the
  backend.
- **No built-in auth/rate limiting.** Use a proxy.
- **Wyoming has no auth/encryption either.** Keep port `10300` on a trusted
  network; it is meant for Home Assistant, not the public internet.
- **Wyoming is final-transcript only.** No interim `transcript-chunk` streaming,
  and no Wyoming translate event.
- **SSE is best-effort OpenAI parity.** The server emits its own event names and
  has not been validated against every third-party SSE consumer.
- **`STT_REQUEST_TIMEOUT_S` cannot cancel CTranslate2.** It only abandons
  the wait; the worker keeps running until the decode finishes.
- **Duration probing is best-effort.** If PyAV cannot determine container
  duration, the duration guard is skipped rather than rejecting valid audio.
- **Package layout.** Both Dockerfiles copy `app.py` and `stt_api/`; adding
  another runtime package requires updating both.
- **No model hot-swap.** The served model is fixed at process start by
  `STT_MODEL_NAME`.

---

## License

MIT.
