# PLAN — Robustness: event-loop offload, resource limits, upload hygiene

> **Goal:** make the server safe to run behind a real load balancer / on shared
> hardware. Today inference runs *on the event loop* (serializing all other
> requests, including `/health`, behind one transcription), there is no
> upper bound on upload size or audio duration (a hostile or malformed upload
> can OOM the process or tie up the worker indefinitely), uploads are buffered
> fully in RAM, and a few smaller hygiene gaps exist. This plan fixes the six
> reported items, phased so each phase keeps the suite green and ships as its
> own commit.

## Reported issues → severity

| # | Finding | Where today | Risk | Priority |
| --- | --- | --- | --- | --- |
| 1 | Blocking `transcribe()` inside `async def` endpoints blocks the event loop → all requests + `/health` serialize behind one transcription | `app.py` `transcribe_audio` / `translate_audio` | High (multi-user) | **P0** |
| 2 | No upload-size / audio-duration limit → OOM or unbounded worker tie-up | none | High (security) | **P0** size / **P1** duration |
| 3 | `await file.read()` buffers whole upload in RAM before disk write → ~2× peak RAM for large files | both routes (`content = await file.read()`) | Medium | **P0** |
| 4 | Temp file always suffixed `.wav` regardless of real container | both routes (`suffix=".wav"`) | Low (PyAV sniffs; downstream tools may trust suffix) | P0 (cheap) |
| 5 | No inference timeout → pathological file hangs a worker indefinitely | none | Medium | P1 |
| 6 | No `stream=true` SSE for transcriptions (OpenAI compatibility gap) | none | Low ("may not matter") | P2 |

## Measured evidence (this machine)

- **GIL is released during CTranslate2 inference**: 5.10 s inference in a worker
  thread while the main thread progressed ~15 M busy-step iterations.
  ⇒ A `ThreadPoolExecutor`/`run_in_threadpool` cleanly parallelizes inference
  with the event loop; **no `ProcessPoolExecutor` needed** (processes would
  also multiply the ~450 MB model footprint per worker).
- Latency floor is dominated by fixed per-request CT2 cost (≈1 s quiet, ≈3 s
  under contention) and scales in 30 s whisper windows — so a long upload is
  exactly the case that strands `/health` today.

---

## Phase 1 (P0) — one commit

### A. Offload blocking work off the event loop

Token pipeline per request: `store.get()` (model load/reload: 0.5–10 s),
`_resolve_language()` (constrained detection decodes + runs the detector),
`wm.transcribe() + segment iteration` (the long pole). **All three are blocking
and currently run inline in `async def` handlers.**

- Keep `async def` endpoints; wrap the three blocking chunks in
  `from starlette.concurrency import run_in_threadpool` and `await` them.
  (Alternative — converting endpoints to sync `def` — also works via Starlette's
  default 40-thread pool, but we want explicit control of the pool + bound.)
- **Bounded concurrency:** `Settings` gains `max_concurrent: int = 2`
  (`WHISPER_MAX_CONCURRENT`). A `threading.Semaphore` held *only* around the
  inference chunk (not around `/health`, not around the upload write) caps
  simultaneous transcriptions so CPU stays responsive and latency predictable.
- **Thread-safety verification step:** confirm two concurrent `transcribe()` on
  one `WhisperModel` are safe (faster-whisper constructs per-call state; CT2
  executors are multi-thread capable). If not, fall back to a per-store
  `inference_lock` that **serializes** inference — the wait now happens inside
  the threadpool, so `/health` and uploads still don't stall on the event loop.
- **Idle-eviction interplay is already safe**: routes capture the model ref
  before awaiting; watchdog uses a non-blocking lock; an in-flight thread keeps
  `wm` alive regardless of `store._model` being nulled.

Proof-of-concern test (real server, e2e): start a slow transcription (a
concatenated multi-minute clip), immediately `/health`; assert `< ~1 s` return
while inference is still running.

### B. Stream the upload to disk + enforce a size cap → 413

Replace full-buffer read with a chunked copy that *stops at the limit*:

```python
async def _drain_upload(file: UploadFile, tmp, limit: int) -> int:
    written = 0
    while chunk := await file.read(1024 * 1024):        # 1 MB chunks
        written += len(chunk)
        if written > limit:
            raise HTTPException(413, f"Upload too large (max {limit} bytes)")
        tmp.write(chunk)
    return written
```

- Peak extra RAM drops from *entire file* to ~1 MB; oversized uploads are
  aborted **mid-stream** (no wasted disk/CPU, temp file unlinked in `finally`).
- Fast path: if `Content-Length` is present and exceeds the cap, return 413
  before reading the body.
- New knob `WHISPER_MAX_UPLOAD_MB: int = 100` (conservative vs. OpenAI's 25 MB;
  a 30 s window keeps per-request memory bounded regardless of byte size).
- Status is **413** (payload too large) with an OpenAI-style detail body; the
  existing "Abandon Bad Input" 400/422 tests keep passing.

### C. Preserve the original suffix (hygiene)

New helper `_safe_suffix(filename) -> str`:

```python
_ALLOWED = {".wav",".mp3",".flac",".m4a",".ogg",".opus",".aac",".wma",
            ".webm",".mp4",".avi",".mov",".mkv",".amr",".3gp",".aiff"}
def _safe_suffix(filename: str | None) -> str:
    ext = (Path(filename or "").suffix or "").lower()
    return ext if ext in _ALLOWED else ".bin"      # was unconditional ".wav"
```

- PyAV/libavformat sniffs the real container from bytes, so this is hygiene for
  any tool that trusts the extension; `.bin` (not `.wav`) avoids implying a codec.
- Temp name: `NamedTemporaryFile(delete=False, prefix="whisper-", suffix=_safe_suffix(...))`.

### Phase-1 tests

- unit: `_safe_suffix` (valid/absent/weird/uppercase); size-cap abort mid-stream
  using a fake `UploadFile`; `Content-Length` fast-path 413.
- e2e (real server): upload > cap → **413**; valid 11 s jfk still 200; `/health`
  returns < 1 s while a long transcription runs; a `.mp3`-suffixed upload (real
  MP3, encoded in-test via PyAV) still transcribes.

---

## Phase 2 (P1) — second commit

### D. Max-audio-duration guard

- New knob `WHISPER_MAX_AUDIO_SECONDS: int = 3600` (0 = off). After the temp
  write, read container duration cheaply via PyAV metadata
  (`av.open(path).streams.audio[0].duration / time_base`) — **no model load, ~
  ms**. Unknown duration (non-seekable/odd containers) → log and skip the guard.
- Over-limit → **400** `"Audio file is too long (max N s)"` (matches OpenAI's
  "too long" 400 rather than a byte-based 413).
- Test: generated >limit wav → 400; under-limit → 200.

### E. Inference request timeout (abandon-the-call semantics)

- New knob `WHISPER_REQUEST_TIMEOUT_S: int = 300`. Run the whole inference chunk
  in the threadpool and await it with `asyncio.wait_for`; on timeout: return
  **504**, unlink the temp, and let the orphaned thread *drain in the threadpool*
  (CTranslate2 cannot be preempted from Python). `WHISPER_MAX_CONCURRENT` bounds
  how many such drained threads can stack, so this cannot grow unboundedly.
- Documented as a backstop: the duration guard (D) already rejects the realistic
  pathological cases before the model even runs.
- Test: monkeypatched slow model (sleeps > timeout) → 504 while the server
  remains healthy (`/health` 200).

---

## Phase 3 (P2, optional) — streaming response

OpenAI supports `stream=true` on transcriptions (SSE deltas). faster-whisper
yields segments lazily, so the data source is already a stream. MVP scope:

- `stream=true` + `response_format=json|verbose_json` only (OpenAI streams
  structured JSON, not text/srt/vtt).
- Incremental segment iteration → per-segment SSE events, then a final JSON
  frame — replaces today's `list(segments)` assembly on this path.
- **Explicitly deferred**: "may not matter for most users"; not worth the
  OpenAI-spec risk until P0/P1 are shipped and there is a real client to test
  against.

---

## Config surface (all new, all optional)

| Env | Default | Meaning | Phase |
| --- | --- | --- | --- |
| `WHISPER_MAX_CONCURRENT` | `2` | Concurrent transcriptions cap (0 = unbounded) | 1 |
| `WHISPER_MAX_UPLOAD_MB` | `100` | Upload size cap → 413 | 1 |
| `WHISPER_MAX_AUDIO_SECONDS` | `3600` | Audio-duration cap → 400 (0 = off) | 2 |
| `WHISPER_REQUEST_TIMEOUT_S` | `300` | Inference timeout → 504 (abandon-call) | 2 |

Each phase updates: `app.py` (+`/health` reporting where useful), the README env
table + troubleshooting, `docker-compose.yml` commented knobs, and the suites
(`tests/` unit + `tests/e2e_*_test.py`).

## Risks / open questions

- **Concurrent `transcribe()` on one `WhisperModel`** — not yet verified safe
  upstream; Phase-1 step A resolves it (fallback: serialize inference under a
  lock, which still fixes `/health`).
- **Timed-out orphan threads** briefly hold a model + worker slot; bounded by
  `WHISPER_MAX_CONCURRENT`; acceptable backstop semantics (documented).
- **413 (bytes) vs 400 (duration)** deliberately split; note the divergence from
  OpenAI (which 400s too-long audio) in the README.
- `stream=true` needs a real SSE-consuming client for spec verification before
  it can be called compatible.
