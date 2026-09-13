# PLAN — RAM footprint: idle model eviction (whisper-api)

> **Goal:** when the API has had no transcription/translation request for a
> configurable window (default 5 min), drop the in-RAM faster-whisper model so
> RSS drops from ~0.5 GB to ~0.1 GB. The next request **blocks until the model
> reloads** (warm disk cache ≈ 0.5 s; cold/network path can hold for seconds).
>
> This is a direct port of the proven M5.5 idle-unload design already shipped in
> `pocket-tts-openai-server` (`PLAN-idle-unload.md`, `engine.py`, `server.py`,
> `tests/test_idle_unload.py`), adapted to whisper's single-file `app.py` and
> faster-whisper's `WhisperModel`.

## Measured baseline (this machine, CPU, CTranslate2, /proc VmRSS)

| State | RSS |
| --- | --- |
| empty python | ~13 MB |
| faster-whisper imported (ctranslate2 pulled in) | ~63 MB |
| `base` model loaded (int8) | ~448 MB |
| after one transcription (jfk.flac) | ~507 MB |
| after dropping model + `gc.collect()` + `malloc_trim(0)` | ~108 MB |
| warm reload (model from HF cache) | ~0.51 s → ~448 MB |

**Takeaway:** ~400 MB (≈80%) is reclaimable by evicting the in-process model.
The hard floor is ~63 MB (loaded libraries) → ~108 MB with allocator slack;
below that requires a full process restart (~13 MB), out of scope (drops the
hot event loop, cold start, needs a supervisor). **Note:** the numbers are even
better than the TTS reference (which reclaimed ~60% / 600 MB) because
faster-whisper has no per-voice state and a smaller resident floor.

## Design: in-process eviction + lazy reload (recommended)

Keep the process alive; evict only the heavy object. The model holds no other
state besides the audio pipeline, so drop = `_model = None` + `gc` + `trim`.

### 1. Introduce a `ModelManager` (encapsulates today's module-global `model`)

Today `model` is a module global mutated by `_load_model` / `_ensure_model`.
Replace it so eviction/reload is testable and single-flight:

```python
class ModelStore:
    """Holder for the faster-whisper model with idle-eviction + lazy reload."""

    def __init__(self, config: Settings):
        self._config = config
        self._model = None            # WhisperModel | None
        self._loader_calls = 0        # observability
        self._last_activity = time.monotonic()
        self._lock = threading.Lock() # serializes load + eviction
        self.loaded = False

    def touch(self) -> None:            # called per API request (never by /health)
        self._last_activity = time.monotonic()

    def get(self) -> "WhisperModel":    # block-until-reloaded wake
        with self._lock:
            if self._model is None:
                self._model = _load_model()
                self._loader_calls += 1
                self.loaded = True
            return self._model

    def maybe_unload(self, now=None) -> bool:  # watchdog calls this; non-blocking
        ...
```

- `_load_model()` keeps its current signature (builds `WhisperModel(...)`).
- `get()` is the only way routes obtain the model; it does a **single-flight**
  reload under `_lock`, so N concurrent waiters after an eviction trigger **one**
  rebuild (no thundering herd).
- `maybe_unload(now)`: if `settings.idle_unload_s > 0` and `_model` is set and
  `now - _last_activity >= idle_unload_s`, then under `_lock` set `_model=None`,
  `loaded=False`, `gc.collect()`, then best-effort
  `ctypes.CDLL("libc.so.6").malloc_trim(0)` (reclaims ~25–50 MB slack; measure).
  Guarded so it **never blocks**: `_lock.acquire(blocking=False)` and skip the
  pass if busy (a transcription holds the lock; watchdog retries next tick).
- `last_request_age_s = time.monotonic() - _last_activity` for `/health`.

> **Whisper-specific note:** our routes are `async def` but call the *blocking*
> `wm.transcribe()` directly on the event loop. That means two requests can't
> be mid-transcribe concurrently in the single-worker default, which already
> serializes eviction vs. generation. But keep the lock-based design anyway:
> it's cheap, matches the reference, and stays correct if we later move
> inference to `run_in_executor`/anyio workers or run multiple uvicorn workers.

### 2. Capture-the-reference rule on route code (critical)

`_ensure_model()` becomes `store.get()`. Each route must capture the returned
model into a local **before** any await, and use that local for the whole
request — so a concurrent eviction (watchdog thread) can null `store._model`
without tearing down the object this request is using:

```python
wm = store.get()          # returns WhisperModel (reloads if evicted)
# ... all transcribe / segment work uses `wm`, never store._model
```

- `transcribe()` returns a lazy `segments` generator consumed *after* the call.
  Capturing `wm` once up front and iterating it while holding the local
  reference keeps the model alive for the full request even if the store's
  pointer is cleared. No route-level lock needed for this race because we never
  touch the store again mid-request.
- `touch()` is called at the **top** of `transcribe_audio` and
  `translate_audio`, right after `_validate_response_format` (before the temp
  file write), so the idle window reflects real API demand. Do **NOT** touch in
  `/health` — continuous health probes would reset the timer and defeat
  eviction (same trap the reference explicitly called out).

### 3. Watchdog in `lifespan` (mirrors reference `_idle_watchdog`)

- `Settings` gains `idle_unload_s: int = 300` (`WHISPER_IDLE_UNLOAD_S`, `0`=
  off) and `idle_poll_s: int = 30` (`WHISPER_IDLE_POLL_S`).
- In `lifespan`, when `settings.idle_unload_s > 0`, spawn a **daemon** thread
  (`threading.Event` stop signal) that loops `stop.wait(poll_s)` →
  `store.maybe_unload()`. `poll_s = min(settings.idle_poll_s,
  max(5, settings.idle_unload_s // 2))` (5 min ⇒ poll 30 s).
- On shutdown, `stop.set()` (daemon thread, so it dies with the process anyway;
  the set just shortens the wait).
- Only the *watchdog* may evict. `get()` on the request path only ever loads;
  no background reload thread is needed (unlike lazy_load's eager start, which
  stays as-is since loading the model eagerly at startup is the default).

### 4. `/health` observability

Extend the existing endpoint to expose the eviction state (mirrors reference):

```json
{ "status": "ok", "model": "base", "compute_type": "int8",
  "loaded": true|false,
  "idle_unload_s": 300,
  "last_request_age_s": 120.5,
  "unloads": 3, "reloads": 4 }
```

- `status` stays `"ok"` after an eviction (next request wakes it) — 503 only
  exists in the pre-first-load path (engine absent), which we keep unchanged.
- Add counters `unloads` / `reloads` to `ModelStore` (incremented in
  `maybe_unload` / `get`).

### 5. Config plumbing (`Settings`)

- Add `idle_unload_s` / `idle_poll_s` fields with the same explicit
  `_env_int` binding pattern already used for `WHISPER_PORT`:
  `idle_unload_s=_env_int("WHISPER_IDLE_UNLOAD_S", 300)`,
  `idle_poll_s=_env_int("WHISPER_IDLE_POLL_S", 30)`.
- `docker-compose.yml`: add the two commented env knobs to the existing
  "Optional knobs" block. Dockerfile needs **no change** (runtime config only).
- README: add both vars to the Configuration table; note `/data` volume keeps
  the HF cache warm so a reload is ~0.5 s and needs no re-download.

### 6. Tests (keep 18 green + new, deterministic — no sleeps)

Port the reference's `tests/test_idle_unload.py` shape, but for our app:

- **Injectable loader:** `ModelStore` gets an optional `loader: Callable[[], Any]`
  kwarg defaulting to `_load_model`; fakes pass a stub and are fast/offline.
- **Explicit clock:** `maybe_unload(now=...)` and direct `_last_activity`
  mutation make all idle timers deterministic.
- New test module `tests/test_idle_unload.py`:
  - config: `WHISPER_IDLE_UNLOAD_S` parse; `0` disables / no watchdog thread.
  - store: not loaded before first `get`; `get` loads once (single-flight under
    6 threads ⇒ loader called once); `maybe_unload` evicts when idle, no-ops
    when recent or when `idle_unload_s == 0`; `malloc_trim` wrapped in try/except.
  - route wake: e2e via `TestClient` — POST transcribe with a **stub model
    store** (loaded via fake loader), simulate eviction (`maybe_unload(now=...)`),
    then POST again and assert 200 + text + exactly one reload.
  - health: reports `loaded`/`unloads`/`reloads`/`last_request_age_s`; and the
    **health probe does not advance/reset the idle window** (call `/health`
    repeatedly, assert `last_activity` unchanged).
  - asserts no eviction while a "transcribe" (holding `_lock`) is in flight.
- The existing 18 e2e tests must keep passing (they cover response formats,
  error paths, translation). The suite will stay booting a real server — the
  new tests use a fake store via the TestClient injection path (add a small
  `create_app(store=...)` hook if needed, or monkeypatch the module-global
  store).

### 7. Edge cases / rejected alternatives

- **Process restart on idle** → ~13 MB but kills open connections, cold
  start, needs an orchestrator. Not worth it; 108 MB floor is fine.
- **Lazy-import faster-whisper:** already lazy in the route path (import
  happens inside `_load_model`) but the 63 MB floor after import is acceptable.
- **Touching the wedge on `/health`:** rejected — health probes reset the idle
  timer and defeat eviction. The watchdog must key off **API request**
  timestamps only (this is the reference's explicit design decision).
- **`whisper` vs CTranslate2's own cache:** CTranslate2 pins some arenas;
  `malloc_trim` gets most back (measured 507→108). If eviction still shows
  slack, attempt `wm.model.unload()` (ctranslate2 model has no public unload
  in our version) — dropping the Python ref + gc + trim is sufficient (measured).
- **`WHISPER_LAZY_LOAD` interaction:** independent. `lazy_load` controls the
  *initial* load timing (startup vs. first request); idle-unload controls
  *subsequent* drops. They compose: both on ⇒ model loads on request, evicts
  when idle, reloads on next request.

## Milestone

M5.5 (whisper) · idle unload. Files: `app.py` (Store + lifecycle + routes +
health), `docker-compose.yml`, `README.md`, `tests/e2e_test.py` (if the app
factory changes), `tests/test_idle_unload.py` (new). No Dockerfile change.

> Reference to mirror: `pocket-tts-openai-server` — `PLAN-idle-unload.md`,
> `src/pocket_tts_openai/engine.py` (TTSEngine.maybe_unload/ensure_loaded),
> `src/pocket_tts_openai/server.py` (_idle_watchdog), `config.py`,
> `routes_speech.py` (health), `tests/test_idle_unload.py`.
