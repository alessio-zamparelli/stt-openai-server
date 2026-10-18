# AGENTS.md

## Project shape

- Single-module OpenAI-compatible Whisper server: all runtime code lives in `app.py` (FastAPI app `app`, module-global `settings`, `ModelStore` for model lifecycle). There is no `src/` or installable package (`pyproject.toml` sets `package = false`).
- CPU-only by design: faster-whisper + CTranslate2, default `base` / `int8`. Do not add GPU/torch paths.
- `README.md` is the user-facing API/env reference; `docs/PLAN-*.md` hold the design rationale and measured performance/robustness numbers.

## Setup and run

```bash
uv sync --locked              # creates the repo-local .venv (includes dev deps)
uv run python app.py          # dev server on :8080
uv run python -c "import app" # import smoke check (does not load the model)
```

- The app is not installed into the venv, so imports only work when the repo root is on `sys.path`.
- Docker: `docker compose up -d --build`, or `docker build -f deploy/Dockerfile -t whisper-api .`. The first startup downloads weights into `/data/hf`.

## Tests — use `python -m pytest`

**Use `uv run python -m pytest ...`, not plain `uv run pytest ...`.** With the current pytest version, `uv run pytest` does not put the repo root on `sys.path`, so unit tests fail during collection with `ModuleNotFoundError: No module named 'app'`. `uv run python -m pytest` works.

```bash
# Fast unit suite (fake model, no network/model download)
uv run python -m pytest tests/test_*.py -q

# Focused test
uv run python -m pytest tests/test_tuning.py::test_route_forwards_beam_and_schedule_to_model -q
```

- Unit tests: `tests/test_idle_unload.py`, `test_language.py`, `test_plan_docs.py`, `test_robustness.py`, `test_timings.py`, `test_tuning.py`. They monkeypatch `app.settings` / `app.store` and use `FakeModel` + `TestClient`.
- E2E tests: `tests/e2e_test.py`, `e2e_language_test.py`, `e2e_robustness_test.py`, `e2e_performance_test.py`. Each boots a real `app.py` subprocess on a random port and runs real inference; first run downloads the `base` model (~140 MB) and `jfk.flac` (cached under `/tmp`). Server startup timeout is 180 s. Prefer `-k`/node IDs, but each file still pays that startup cost.
- `tests/e2e_performance_test.py` timing assertions are skipped unless `PERF_WALLTIME=1`; run only on quiet hardware.
- `tests/test_plan_docs.py` is a documentation drift guard. Adding/renaming any `WHISPER_*` knob means updating `app.py`, the `Settings` model (and `/health` when the knob is reported there), the README configuration table, and `docker-compose.yml` together — the test scans tokens anywhere in `app.py`, the README config section, and compose, including comments. A new preset must also appear in the README preset table.
- No lint, formatter, or typecheck task is configured. `pyrightconfig.json` points at `/workspace/.venv` and a Python 3.14 site-packages path, not the repo-local `.venv`; adjust it if you run pyright.

## Runtime invariants

- Env vars are read **once at import** into the global `settings`. Changing `os.environ` after import has no effect; tests should construct `Settings(...)` / swap module globals.
- Inference uses one shared model. `transcribe()` is not safe concurrently on one instance, so calls are serialized with `_inference_lock`; `WHISPER_MAX_CONCURRENT` bounds the queue, not parallel decodes. Preserve the lock and `ModelStore` single-flight load/eviction behavior.
- `/health` deliberately does **not** call `store.touch()`; this prevents monitoring traffic from defeating idle eviction. Do not "fix" it.
- The request `model` field is accepted for OpenAI compatibility but ignored; the served model is always `settings.model_name`.
- Uploads are buffered by Starlette into an `UploadFile` (`SpooledTemporaryFile`) before the route runs; routes pass that file-like object straight to PyAV/faster-whisper (no second temp copy). Size/duration limits are enforced before inference, and the upload is closed on both streaming and non-streaming paths. Keep cleanup symmetric when editing routes.

## Docker and CI

- The runtime image copies **only `app.py`** and runs `uv sync --locked --no-install-project`. If you add runtime modules or data files, update `deploy/Dockerfile` or the image will fail at startup.
- Docker pins uv `0.12.13` and Python `3.14`; a newer local uv may be in use. If dependencies change, regenerate `uv.lock` intentionally (`uv lock` / `uv lock --upgrade`) and keep the Dockerfile's uv pin compatible, otherwise `--locked` builds can fail. `.dockerignore` excludes tests and the local `.venv`.
- `.github/workflows/docker-publish.yml` runs only an import/OpenAPI smoke test, then publishes to GHCR on pushes to `main` and `v*.*.*` tags. There is no test job and no PR trigger, so run the unit/e2e suites locally before pushing.
