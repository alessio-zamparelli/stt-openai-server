# AGENTS.md

## Project shape

- Installable FastAPI package in `stt_api/` plus a compatibility launcher `app.py`.
- `stt_api/main.py` owns the FastAPI app, lifespan, and CLI (`run()`/`main()`).
- `stt_api/routers/` contains the HTTP surface:
  - `transcriptions.py` — `POST /v1/audio/transcriptions`
  - `translations.py` — `POST /v1/audio/translations`
  - `system.py` — `GET /v1/models`, `GET /health`
- `stt_api/services/` contains shared logic:
  - `model_store.py` — `ModelStore`, idle eviction, single-flight reload
  - `audio.py` — limits, PyAV duration probing, prefix decode
  - `formatting.py` — JSON/text/SRT/VTT response helpers
  - `language.py` — language validation/resolution
  - `inference.py` — decoding kwargs, batching, collection, SSE
  - `wyoming.py` — optional Wyoming ASR TCP listener + zeroconf
- `stt_api/runtime.py` owns process-wide mutable state: `settings`, `store`,
  `_inference_lock`, the bounded executor, and rolling stage timings.
- `stt_api/config.py` defines `Settings`, `load_settings()`, and env resolution.
- `app.py` only re-exports `app`/`run` so `python app.py` and
  `python app.py --healthcheck` keep working. Do not put application logic there.
- CPU-only by design: faster-whisper + CTranslate2, default `base` / `int8`.
  Do not add GPU/torch paths.
- `README.md` is the user-facing API/env reference; `docs/PLAN-*.md` hold design
  rationale and measured performance/robustness numbers.

## Naming

- Canonical env prefix is `STT_*`. `WHISPER_*` is a deprecated alias resolved
  generically in `stt_api/config.py`; `STT_*` wins when both are set and the
  legacy name logs one warning.
- The engine remains faster-whisper; do not rename the dependency, model
  identifiers (`base`, `small`, `large-v3`), or `owned_by: "faster-whisper"`.
- Project/distribution: `stt-api`; package: `stt_api`; Docker image/service:
  `stt-api` / `stt`; volume: `stt-data`.

## Setup and run

```bash
uv sync --locked                 # repo-local .venv (includes dev deps + package)
uv run python app.py             # dev server on :8080
uv run uvicorn stt_api.main:app  # ASGI entrypoint
uv run stt-api                   # console script
uv run python -c "import stt_api.main"  # import smoke check (no model load)
```

- Local dev and CI use `.python-version` (3.14), matching the Docker image
  (`python:3.14-slim`).
- Docker: `docker compose up -d --build`, or
  `docker build -f deploy/Dockerfile -t stt-api .`. The first startup downloads
  weights into `/data/hf`.

## Tests — use `python -m pytest`

**Use `uv run python -m pytest ...`, not plain `uv run pytest ...`.** With the
current pytest version, `uv run pytest` does not put the repo root on
`sys.path`, so tests can fail during collection.

```bash
# Fast unit suite (fake model, no network/model download)
uv run python -m pytest tests/test_*.py -q

# Focused test
uv run python -m pytest tests/test_tuning.py::test_route_forwards_beam_and_schedule_to_model -q
```

- Unit tests: `tests/test_config.py`, `test_idle_unload.py`, `test_language.py`,
  `test_plan_docs.py`, `test_robustness.py`, `test_timings.py`, `test_tuning.py`,
  `test_wyoming.py`. They swap `stt_api.runtime.settings` / `.store` and use
  `FakeModel` + `TestClient`.
- E2E tests: `tests/e2e_test.py`, `e2e_language_test.py`,
  `e2e_robustness_test.py`, `e2e_wyoming_test.py`, `e2e_performance_test.py`.
  Each boots a real `app.py` subprocess on random ports and runs real inference;
  first run downloads `base` (~140 MB) and `jfk.flac` (cached under `/tmp`).
  Server startup timeout is 180 s.
- `tests/e2e_performance_test.py` timing assertions are skipped unless
  `PERF_WALLTIME=1`; run only on quiet hardware.
- `tests/test_plan_docs.py` is a documentation drift guard. Adding/renaming any
  `STT_*` knob means updating `stt_api/config.py`, the `Settings` model (and
  `/health` when the knob is reported there), the README configuration table,
  and `docker-compose.yml` together — the test scans tokens anywhere in the
  package, the README config section, and compose, including comments. A new
  preset must also appear in the README preset table. `WHISPER_*` aliases are
  generic, so they need no per-knob docs.
- No lint, formatter, or typecheck task is configured. `pyrightconfig.json`
  points at `/workspace/.venv` and a Python 3.14 site-packages path, not the
  repo-local `.venv`; adjust it if you run pyright.

## Runtime invariants

- Env vars are read **once at import** into `stt_api.runtime.settings` via
  `stt_api.config.load_settings()`. Changing `os.environ` after import has no
  effect; tests should construct `Settings(...)` and swap
  `runtime.settings`/`runtime.store`.
- Services/routers must read `runtime.settings` and `runtime.store` at call
  time (`import stt_api.runtime as runtime`), never
  `from stt_api.runtime import settings`. Late binding is what makes tests,
  idle eviction, and dependency overrides work.
- Inference uses one shared model. `transcribe()` is not safe concurrently on
  one instance, so calls are serialized with `runtime._inference_lock`;
  `STT_MAX_CONCURRENT` bounds the queue, not parallel decodes. Preserve the lock
  and `ModelStore` single-flight load/eviction behavior.
- `/health` deliberately does **not** call `store.touch()`; this prevents
  monitoring traffic from defeating idle eviction. Do not "fix" it. Wyoming
  `describe` also must not touch the store or load the model; `transcribe` and
  `audio-stop` do touch it.
- The request `model` field is accepted for OpenAI compatibility but ignored;
  the served model is always `runtime.settings.model_name`. The Wyoming
  `transcribe.name` field is ignored for the same reason.
- Uploads are buffered by Starlette into an `UploadFile` (`SpooledTemporaryFile`)
  before the route runs; routes pass that file-like object straight to
  PyAV/faster-whisper (no second temp copy). Size/duration limits are enforced
  before inference, and the upload is closed on both streaming and
  non-streaming paths. Keep cleanup symmetric when editing routes.
- Wyoming accumulates PCM into a `SpooledTemporaryFile` + WAV header, converts
  to 16 kHz mono s16, enforces the same audio limits, sends one `transcript`,
  and closes the connection. It shares `runtime._offload` /
  `runtime._inference_lock` with HTTP.
- Startup fails if the Wyoming port cannot bind; zeroconf registration failures
  are warnings only (manual host/port still works).

## Docker and CI

- The runtime image copies **`app.py` and `stt_api/`** and runs
  `uv sync --locked --no-install-project`. If you add runtime modules or data
  files outside `stt_api/`, update `deploy/Dockerfile` or the image will fail at
  startup.
- Docker pins uv `0.13.1` and Python `3.14`; a different local uv may be in use.
  If dependencies change, regenerate `uv.lock` intentionally (`uv lock` /
  `uv lock --upgrade`) and keep the Dockerfile's uv pin compatible, otherwise
  `--locked` builds can fail. `.dockerignore` excludes tests and the local
  `.venv`.
- `.github/workflows/docker-publish.yml` runs only an import/OpenAPI smoke test,
  then publishes to GHCR on pushes to `main` and `v*.*.*` tags. There is no test
  job and no PR trigger, so run the unit/e2e suites locally before pushing.
