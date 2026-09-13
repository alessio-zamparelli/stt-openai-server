"""
OpenAI-compatible Whisper API server backed by faster-whisper.

Exposes /v1/audio/transcriptions and /v1/audio/translations
with defaults matching the OpenAI Whisper API, using the
'base' model with INT8 quantization.
"""

import asyncio
import ctypes
import gc
import json
import logging
import os
import queue
import tempfile
import threading
import time
from concurrent import futures
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Callable, List, Optional

from fastapi import FastAPI, File, Form, UploadFile, HTTPException, Request
from fastapi.responses import PlainTextResponse, StreamingResponse
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

class Settings(BaseModel):
    """Server / model configuration — overridable via WHISPER_* env vars."""

    model_name: str = "base"
    device: str = "cpu"
    compute_type: str = "int8"
    host: str = "0.0.0.0"
    port: int = 8080
    lazy_load: bool = False
    idle_unload_s: int = 300  # evict the model to free RAM after this much idle; 0 = off
    idle_poll_s: int = 30  # watchdog cadence (only when idle_unload_s > 0)
    # Language handling. Priority per request: client `language` field, then
    # `default_language`, then (with a multi-code `allowed_languages`) a
    # detection constrained to the allowlist, then faster-whisper full auto.
    default_language: Optional[str] = None  # WHISPER_LANGUAGE
    allowed_languages: Optional[str] = None  # WHISPER_LANGUAGES  e.g. "it,en"
    initial_prompt: Optional[str] = None  # WHISPER_INITIAL_PROMPT
    # Robustness / resource limits (docs/PLAN-robustness-concurrency.md).
    max_concurrent: int = 2  # WHISPER_MAX_CONCURRENT (0 = unbounded threadpool)
    max_upload_mb: int = 100  # WHISPER_MAX_UPLOAD_MB (0 = unlimited)
    max_audio_seconds: int = 3600  # WHISPER_MAX_AUDIO_SECONDS (0 = off)
    request_timeout_s: float = 300.0  # WHISPER_REQUEST_TIMEOUT_S (0 = off)
    # Inference tuning (docs/PLAN-performance.md Phase 1). Accuracy-first
    # defaults; the knobs exist for latency-sensitive deployments.
    beam_size: int = 5  # WHISPER_BEAM_SIZE
    best_of: int = 5  # WHISPER_BEST_OF
    temperature_schedule: str = ""  # WHISPER_TEMPERATURES e.g. "0,0.2,0.4" ("" = fw default)
    cpu_threads: int = 0  # WHISPER_CPU_THREADS (0 = faster-whisper default)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _env_int(name: str, default: int) -> int:
    """Parse an integer env var, falling back to the default on bad input."""
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    """Parse a float env var, falling back to the default on bad input."""
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


settings = Settings(
    model_name=os.getenv("WHISPER_MODEL_NAME", "base"),
    device=os.getenv("WHISPER_DEVICE", "cpu"),
    compute_type=os.getenv("WHISPER_COMPUTE_TYPE", "int8"),
    host=os.getenv("WHISPER_HOST", "0.0.0.0"),
    port=_env_int("WHISPER_PORT", 8080),
    lazy_load=_env_bool("WHISPER_LAZY_LOAD", False),
    idle_unload_s=_env_int("WHISPER_IDLE_UNLOAD_S", 300),
    idle_poll_s=_env_int("WHISPER_IDLE_POLL_S", 30),
    default_language=os.getenv("WHISPER_LANGUAGE") or None,
    allowed_languages=os.getenv("WHISPER_LANGUAGES") or None,
    initial_prompt=os.getenv("WHISPER_INITIAL_PROMPT") or None,
    max_concurrent=_env_int("WHISPER_MAX_CONCURRENT", 2),
    max_upload_mb=_env_int("WHISPER_MAX_UPLOAD_MB", 100),
    max_audio_seconds=_env_int("WHISPER_MAX_AUDIO_SECONDS", 3600),
    request_timeout_s=_env_float("WHISPER_REQUEST_TIMEOUT_S", 300.0),
    beam_size=_env_int("WHISPER_BEAM_SIZE", 5),
    best_of=_env_int("WHISPER_BEST_OF", 5),
    temperature_schedule=os.getenv("WHISPER_TEMPERATURES", "").strip(),
    cpu_threads=_env_int("WHISPER_CPU_THREADS", 0),
)


# ---------------------------------------------------------------------------
# Global state: ModelStore (idle-eviction + lazy reload)
# ---------------------------------------------------------------------------
# See docs/PLAN-idle-unload.md — port of pocket-tts-openai-server M5.5.
# The store owns the faster-whisper model; after `idle_unload_s` without API
# requests the watchdog drops it so RSS returns toward the process floor
# (measured ~507 MB -> ~108 MB for `base`/int8), and the next request blocks
# until it reloads (warm HF cache ~0.5 s).


def _load_model():
    from faster_whisper import WhisperModel

    return WhisperModel(
        settings.model_name,
        device=settings.device,
        compute_type=settings.compute_type,
        cpu_threads=settings.cpu_threads,
    )


class ModelStore:
    """Holder for the faster-whisper model with idle-eviction + lazy reload.

    ``loader`` is injectable for tests (defaults to ``_load_model``); with
    ``idle_unload_s == 0`` eviction is disabled entirely (fakes, tests that
    need a resident model, or operators who want it always-hot).
    """

    def __init__(self, config: Settings, loader: Callable[[], Any] | None = None):
        self._config = config
        self._loader = loader if loader is not None else _load_model
        self._model: Any = None
        self._last_activity = time.monotonic()
        self._lock = threading.Lock()  # serializes load + eviction
        self.loaded = False
        self.unloads = 0  # idle-eviction events (model dropped)
        self.reloads = 0  # model rebuilds (initial load + post-eviction)

    @property
    def eviction_enabled(self) -> bool:
        return self._config.idle_unload_s > 0

    def touch(self) -> None:
        """Record an API request — resets the idle window. NEVER called by
        /health (continuous health probes would defeat eviction)."""
        self._last_activity = time.monotonic()

    def last_request_age(self) -> float:
        return time.monotonic() - self._last_activity

    def get(self) -> Any:
        """Return the resident model, building it single-flight if absent/evicted.

        Concurrent callers block on ``_lock`` and share the first rebuild (no
        thundering herd). Callers must capture the returned model into a local
        and use that reference for the whole request — a concurrent eviction
        can then clear the store's pointer without tearing down the model the
        request is using.
        """
        with self._lock:
            if self._model is None:
                self._model = self._loader()
                self.loaded = True
                self.reloads += 1
                if self.reloads == 1:
                    logger.info(
                        "model loaded: '%s' on %s (%s)",
                        self._config.model_name,
                        self._config.device,
                        self._config.compute_type,
                    )
                else:
                    logger.info("model reloaded after eviction (loaded %d times)", self.reloads)
            return self._model

    def maybe_unload(self, now: float | None = None) -> bool:
        """Evict the resident model if idle past ``idle_unload_s``.

        Returns True if the model was dropped. Non-blocking on ``_lock`` so an
        in-flight transcription is never evicted under it (watchdog retries
        next tick). Best-effort returns freed heap to the OS via gc + trim.
        """
        if not self.eviction_enabled or self._model is None:
            return False
        if now is None:
            now = time.monotonic()
        if now - self._last_activity < self._config.idle_unload_s:
            return False
        if not self._lock.acquire(blocking=False):
            return False
        try:
            # Re-check under the lock; a request may have landed since we looked.
            if (
                self._model is not None
                and now - self._last_activity >= self._config.idle_unload_s
            ):
                self._model = None
                self.loaded = False
                self.unloads += 1
                gc.collect()
                try:
                    ctypes.CDLL("libc.so.6").malloc_trim(0)
                except Exception:  # pragma: no cover - non-glibc
                    logger.debug("malloc_trim unavailable on this platform")
                    logger.info(
                    "idle %.0fs: evicted model to free RAM (unloads=%d)",
                    self._config.idle_unload_s,
                    self.unloads,
                )
                return True
            return False
        finally:
            self._lock.release()


store = ModelStore(settings)


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not settings.lazy_load:
        store.get()  # eager pre-load at startup
        print(
            f"[server] faster-whisper model '{settings.model_name}' "
            f"on {settings.device} ({settings.compute_type})"
        )

    # Eagerly create the bounded inference pool before any request can hit it
    # (docs/PLAN-performance.md Phase 0: the lazy path can never race).
    _init_inference_pool()

    # RAM-reclamation watchdog: evict the idle model (PLAN-idle-unload.md).
    # Only spawned when eviction is enabled (idle_unload_s > 0). Health probes
    # never call store.touch(), so they don't reset the idle window.
    stop = threading.Event()
    app.state._idle_stop = stop
    if store.eviction_enabled:
        threading.Thread(target=_idle_watchdog, args=(stop,), daemon=True).start()

    try:
        yield
    finally:
        stop.set()
        _reset_inference_pool()


def _idle_watchdog(stop: threading.Event) -> None:
    """Periodically evict the model after ``idle_unload_s`` without API requests."""
    interval = min(settings.idle_poll_s, max(5, settings.idle_unload_s // 2))
    while not stop.wait(interval):
        try:
            store.maybe_unload()
        except Exception:  # pragma: no cover - defensive
            logger.exception("idle-unload watchdog failed")


# ---------------------------------------------------------------------------
# App creation
# ---------------------------------------------------------------------------

app = FastAPI(
    title="OpenAI-Compatible Whisper API",
    description=(
        "faster-whisper backend exposing OpenAI-compatible "
        "/v1/audio/transcriptions and /v1/audio/translations endpoints."
    ),
    version="0.1.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Pydantic models (OpenAI-compatible request/response shapes)
# ---------------------------------------------------------------------------

class TranscriptionCreateRequest(BaseModel):
    """Mirrors OpenAI's /v1/audio/transcriptions request body."""

    # When sending via multipart/form-data, these come from form fields.
    # They are defined here for documentation and for JSON fallback.
    model: Optional[str] = Field(
        default=None, description="ID of the model to use (ignored — uses server default)."
    )
    language: Optional[str] = Field(
        default=None, description="Language of the input audio (e.g. 'en')."
    )
    prompt: Optional[str] = Field(
        default=None,
        description="An optional prompt to guide the model's behaviour.",
    )
    response_format: Optional[str] = Field(
        default="json",
        description="Format of the output: json, text, srt, verbose_json, vtt.",
    )
    temperature: Optional[float] = Field(
        default=0.0, description="Sampling temperature (0 = deterministic).", ge=0.0, le=1.0
    )
    timestamps: Optional[bool] = Field(
        default=True, description="Include timestamped segments in the response."
    )


class TranslationCreateRequest(BaseModel):
    """Mirrors OpenAI's /v1/audio/translations request body."""

    model: Optional[str] = Field(default=None, description="Ignored — uses server default.")
    prompt: Optional[str] = Field(
        default=None,
        description="An optional prompt to guide the model's behaviour.",
    )
    response_format: Optional[str] = Field(
        default="json",
        description="Format of the output: json, text, srt, verbose_json, vtt.",
    )
    temperature: Optional[float] = Field(
        default=0.0, description="Sampling temperature (0 = deterministic).", ge=0.0, le=1.0
    )
    timestamps: Optional[bool] = Field(
        default=True, description="Include timestamped segments in the response."
    )


class TranscriptionResponse(BaseModel):
    """Mirrors OpenAI's transcription response."""

    text: str
    task: str = "transcribe"
    language: Optional[str] = None
    segments: Optional[List[dict]] = None


class TranslationResponse(BaseModel):
    """Mirrors OpenAI's translation response."""

    text: str
    task: str = "translate"
    language: Optional[str] = None
    segments: Optional[List[dict]] = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

FORMAT_MAP = {
    "json": "json",
    "text": "txt",
    "srt": "srt",
    "verbose_json": "json",
    "vtt": "vtt",
}


def _format_response(
    segments,
    text: str,
    language: str,
    response_format: str,
    task: str,
) -> dict:
    """Format faster-whisper output into an OpenAI-compatible dict."""

    if response_format in ("text", "txt", "vtt", "srt"):
        # For text-like formats, faster-whisper returns a plain string from
        # the file written to disk.  We handle those in the endpoint.
        return {"text": text}

    # json / verbose_json
    return {
        "task": task,
        "language": language,
        "segments": segments if segments else [],
        "text": text,
    }


def _ts_srt(seconds: float) -> str:
    """Seconds -> SRT timestamp (HH:MM:SS,mmm)."""
    try:
        ms = max(0, int(round(seconds * 1000)))
    except (TypeError, ValueError, OverflowError):
        ms = 0
    h, rem = divmod(ms, 3600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _ts_vtt(seconds: float) -> str:
    """Seconds -> WebVTT timestamp (HH:MM:SS.mmm)."""
    return _ts_srt(seconds).replace(",", ".")


def _to_srt(segments) -> str:
    blocks = []
    for i, seg in enumerate(segments, start=1):
        blocks.append(
            f"{i}\n{_ts_srt(seg['start'])} --> {_ts_srt(seg['end'])}\n{seg['text'].strip()}\n"
        )
    return "\n".join(blocks)


def _to_vtt(segments) -> str:
    blocks = ["WEBVTT\n"]
    for seg in segments:
        blocks.append(
            f"\n{_ts_vtt(seg['start'])} --> {_ts_vtt(seg['end'])}\n{seg['text'].strip()}\n"
        )
    return "".join(blocks)


def _validate_response_format(response_format: Optional[str]) -> str:
    """Resolve response_format or raise 400, matching OpenAI behaviour."""
    fmt = FORMAT_MAP.get(response_format or "json")
    if fmt is None:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Invalid response_format: {response_format!r}. "
                f"Expected one of: {', '.join(sorted(FORMAT_MAP))}."
            ),
        )
    return fmt


def _split_allowed_languages(raw: Optional[str]) -> List[str]:
    """Parse a WHISPER_LANGUAGES value like 'it, en' -> ['it', 'en']."""
    if not raw:
        return []
    return [code.strip().lower() for code in raw.split(",") if code.strip()]


def _supported_language_codes(model: Any) -> Optional[set]:
    """Whisper's supported ISO codes for a model (None when unavailable)."""
    codes = getattr(model, "supported_languages", None)
    return set(codes) if codes else None


def _check_language(code: Optional[str], supported: Optional[set]) -> Optional[str]:
    """Normalize/validate an ISO code, raising a clear 400 on bad values."""
    if not code:
        return None
    code = code.strip().lower()
    if not code:  # whitespace-only counts as "no preference", like an empty field
        return None
    if supported and code not in supported:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported language {code!r}. Supported: {sorted(supported)}. "
                "Pass an ISO-639-1 code such as 'en' or 'it'."
            ),
        )
    return code


def _resolve_language(model: Any, audio_path: str, requested: Optional[str]) -> Optional[str]:
    """Resolve the `language` hint forwarded to faster-whisper.

    Priority:
      1. The client's `language` field (explicit override; an empty or
         whitespace-only value is treated as absent and falls through).
      2. Server default  WHISPER_LANGUAGE.
      3. WHISPER_LANGUAGES allowlist:
         - a single code is used verbatim;
         - several codes enable *constrained* detection: run faster-whisper's
           language detector, then pick the highest-probability *allowed* code
           (plain auto-detect ignores the allowlist entirely).
      4. None -> faster-whisper full auto-detect.
    """
    supported = _supported_language_codes(model)

    if requested is not None:
        checked = _check_language(requested, supported)
        if checked:  # empty/whitespace-only -> absent, fall through to defaults
            return checked
    if settings.default_language:
        return _check_language(settings.default_language, supported)

    allowed = _split_allowed_languages(settings.allowed_languages)
    if len(allowed) == 1:
        return _check_language(allowed[0], supported)
    if len(allowed) > 1:
        # Validate the whole allowlist up front for a clear 400.
        for code in allowed:
            _check_language(code, supported)
        # Constrained detection needs decoded audio; faster-whisper returns the
        # full ranked list, so we can ignore every code outside the allowlist.
        from faster_whisper.audio import decode_audio

        audio = decode_audio(audio_path)
        _code, _prob, all_scores = model.detect_language(audio)
        best: Optional[tuple] = None
        for code, prob in all_scores:
            candidate = code.strip().lower()
            if candidate in allowed and (best is None or prob > best[1]):
                best = (candidate, prob)
        return best[0] if best else None
    return None


def _text_response(fmt: str, segment_list, full_text: str) -> PlainTextResponse:
    """Return plain-text / subtitle responses, OpenAI style."""
    if fmt == "txt":
        return PlainTextResponse(full_text)
    if fmt == "srt":
        return PlainTextResponse(_to_srt(segment_list), media_type="application/x-subrip")
    return PlainTextResponse(_to_vtt(segment_list), media_type="text/vtt")


# ---------------------------------------------------------------------------
# Robustness helpers (docs/PLAN-robustness-concurrency.md)
# ---------------------------------------------------------------------------

# Temp-file suffixes we pass through. PyAV probes the container from the bytes
# (never trusts the extension); mapping real suffixes is hygiene for anything
# downstream that does trust it.
_ALLOWED_SUFFIXES = {
    ".wav", ".mp3", ".flac", ".m4a", ".ogg", ".opus", ".aac", ".wma",
    ".webm", ".mp4", ".avi", ".mov", ".mkv", ".amr", ".3gp", ".aiff",
}


def _safe_suffix(filename: Optional[str]) -> str:
    """Sanitized, whitelisted extension for the temp copy of an upload."""
    ext = (Path(filename or "").suffix or "").lower()
    return ext if ext in _ALLOWED_SUFFIXES else ".bin"


_inference_executor: Optional[futures.ThreadPoolExecutor] = None
_inference_pool_lock = threading.Lock()


def _build_inference_pool_locked() -> None:
    """Create the pool from the current settings. Caller must hold the lock."""
    global _inference_executor
    if _inference_executor is None and settings.max_concurrent > 0:
        _inference_executor = futures.ThreadPoolExecutor(
            max_workers=settings.max_concurrent,
            thread_name_prefix="whisper-infer",
        )


def _init_inference_pool() -> None:
    """Eagerly create the bounded pool at startup (event loop is single-threaded)."""
    with _inference_pool_lock:
        _build_inference_pool_locked()


def _reset_inference_pool() -> None:
    """Close and drop the pool so it can be rebuilt at a new max_concurrent."""
    global _inference_executor
    with _inference_pool_lock:
        if _inference_executor is not None:
            _inference_executor.shutdown(wait=False)
            _inference_executor = None


def _inference_pool() -> Optional[futures.ThreadPoolExecutor]:
    """Bounded worker pool for inference (None -> anyio pool, unbounded).

    Double-checked locking: safe even if a future caller invokes this off the
    event loop (docs/PLAN-performance.md Phase 0). ``_init_inference_pool()``
    warms it at startup, so the lock path is effectively never taken in
    production.
    """
    if _inference_executor is None:
        with _inference_pool_lock:
            _build_inference_pool_locked()
    return _inference_executor


async def _offload(fn: Callable, *args) -> Any:
    """Run a blocking chunk off the event loop (CTranslate2 releases the GIL).

    Uses the bounded inference pool when WHISPER_MAX_CONCURRENT > 0 (backpressures
    excess requests by queueing), else starlette's anyio pool for an unbounded
    thread-per-request model.
    """
    pool = _inference_pool()
    if pool is None:
        return await run_in_threadpool(fn, *args)
    return await asyncio.get_running_loop().run_in_executor(pool, fn, *args)


def _upload_limit() -> Optional[int]:
    """Upload cap in bytes (None = unlimited)."""
    mb = settings.max_upload_mb
    return mb * 1024 * 1024 if mb > 0 else None


def _reject_too_large(size: int, limit: Optional[int]) -> None:
    if limit is not None and size > limit:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({size} bytes > {limit}-byte limit).",
        )


def _check_content_length(header: Optional[str], limit: Optional[int]) -> None:
    """Fast-path 413 from Content-Length before reading the body."""
    if limit is None or not header:
        return
    try:
        declared = int(header)
    except (TypeError, ValueError):
        return
    _reject_too_large(declared, limit)


async def _write_upload(file: UploadFile, tmp: Any, limit: Optional[int]) -> int:
    """Stream the upload to disk in 1 MB chunks, 413-ing mid-stream past the cap.

    Returns the number of bytes written. Peak RAM is bounded to ~1 MB per
    request instead of buffering the whole file.
    """
    written = 0
    while chunk := await file.read(1024 * 1024):
        written += len(chunk)
        _reject_too_large(written, limit)
        tmp.write(chunk)
    return written


def _audio_duration_s(path: str) -> Optional[float]:
    """Container duration via PyAV metadata (fast; no model load). None if unknown."""
    try:
        import av

        container = av.open(path)
        try:
            stream = container.streams.audio[0]
            if stream.duration and stream.time_base:
                # duration * time_base is a Fraction; coerce to float seconds.
                return float(stream.duration * stream.time_base)
        finally:
            container.close()
    except Exception:
        logger.debug("duration probe failed for %s", path, exc_info=True)
    return None


async def _guard_duration_ok(path: str) -> None:
    """Reject audio longer than WHISPER_MAX_AUDIO_SECONDS (OpenAI-style 400)."""
    max_s = settings.max_audio_seconds
    if max_s and max_s > 0:
        duration = await _offload(_audio_duration_s, path)
        if duration is not None and duration > max_s:
            raise HTTPException(
                status_code=400,
                detail=f"Audio file is too long ({duration:.1f}s > {max_s}s).",
            )


async def _maybe_timeout(awaitable: Any, timeout_s: float) -> Any:
    """Abandon-the-call inference timeout (CTranslate2 cannot be preempted).

    On expiry the scheduler 504s and the worker thread drains the orphaned
    inference in the bounded pool — no way to kill it mid-flight, so we only
    stop waiting.
    """
    if timeout_s and timeout_s > 0:
        try:
            return await asyncio.wait_for(awaitable, timeout=timeout_s)
        except asyncio.TimeoutError:
            raise HTTPException(
                status_code=504,
                detail=f"Transcription timed out after {timeout_s:g}s.",
            )
    return await awaitable


def _segment_dict(seg) -> dict:
    return {
        "id": seg.id,
        "seek": seg.seek,
        "start": seg.start,
        "end": seg.end,
        "text": seg.text,
        "tokens": seg.tokens,
        "temperature": seg.temperature,
        "avg_logprob": seg.avg_logprob,
        "compression_ratio": seg.compression_ratio,
        "no_speech_prob": seg.no_speech_prob,
    }


def _parse_temperatures(raw: Optional[str]) -> Optional[list]:
    """Parse WHISPER_TEMPERATURES ('0,0.2,0.4') -> [0.0, 0.2, 0.4]; None = default.

    Bad input falls back to the faster-whisper default schedule rather than
    failing startup or reverting to a single pass.
    """
    raw = (raw or "").strip()
    if not raw:
        return None
    try:
        vals = [float(v) for v in raw.split(",") if v.strip()]
    except ValueError:
        logger.warning("ignoring bad WHISPER_TEMPERATURES=%r", raw)
        return None
    return vals or None


def _build_kwargs(
    language: Optional[str],
    prompt: Optional[str],
    temperature: float,
    task: Optional[str] = None,
) -> dict:
    """Shared transcription kwargs (single source of truth for both endpoints)."""
    kwargs: dict = {
        "beam_size": settings.beam_size,
        "best_of": settings.best_of,
        "condition_on_previous_text": False,
        "temperature": temperature if temperature is not None else 0.0,
    }
    # A server-wide fallback schedule (WHISPER_TEMPERATURES) replaces the
    # per-request scalar when set — documented tradeoff, PLAN-performance 1B.
    schedule = _parse_temperatures(settings.temperature_schedule)
    if schedule:
        kwargs["temperature"] = schedule
    if task:
        kwargs["task"] = task
    if language:
        kwargs["language"] = language
    if prompt:
        kwargs["initial_prompt"] = prompt
    return kwargs


def _transcribe_collect(wm: Any, path: str, kwargs: dict) -> tuple:
    """Blocking unit: transcribe + aggregate segments/text (worker thread)."""
    segments, info = wm.transcribe(path, **kwargs)
    segment_list = [_segment_dict(seg) for seg in segments]
    full_text = "".join(seg["text"] for seg in segment_list).strip()
    return segment_list, info, full_text


# -- OpenAI-style streaming (stream=true) ----------------------------------
#
# faster-whisper yields segments lazily, so a producer thread pushes each as it
# is produced and the SSE generator forwards it. The wire format is best-effort
# OpenAI-shaped (transcript deltas, transcript.done, [DONE]); verify against a
# real client before promising strict parity.


def _stream_worker(wm: Any, path: str, kwargs: dict, q: "queue.SimpleQueue") -> None:
    """Producer: ('segment', dict) / ('done', (info, text)) / ('error', msg) / ('end', None)."""
    try:
        segments, info = wm.transcribe(path, **kwargs)
        parts = []
        for seg in segments:
            d = _segment_dict(seg)
            parts.append(d["text"])
            q.put(("segment", d))
        q.put(("done", (info, "".join(parts).strip())))
    except Exception as e:  # surfaced to the client as an SSE error event
        q.put(("error", repr(e)))
    finally:
        q.put(("end", None))


async def _sse_stream(wm: Any, path: str, kwargs: dict, task: str):
    """SSE generator: per-segment deltas, then transcript.done + [DONE]."""
    q: "queue.SimpleQueue" = queue.SimpleQueue()
    loop = asyncio.get_running_loop()
    fut = loop.run_in_executor(_inference_pool(), _stream_worker, wm, path, kwargs, q)
    try:
        while True:
            kind, payload = await run_in_threadpool(q.get)
            if kind == "segment":
                seg = payload
                yield (
                    "event: transcript\n"
                    f"data: {json.dumps({'delta': seg['text'], 'timestamp': {'start': seg['start'], 'end': seg['end']}})}\n\n"
                )
            elif kind == "done":
                info, text = payload
                yield (
                    "event: transcript.done\n"
                    f"data: {json.dumps({'text': text, 'language': info.language, 'task': task})}\n\n"
                    "event: done\ndata: [DONE]\n\n"
                )
                break
            elif kind == "error":
                yield f"event: error\ndata: {json.dumps({'error': payload})}\n\n"
                break
            else:  # 'end' sentinel (defensive; done/error break out first)
                break
    finally:
        # Drain the producer before closing so the pool slot is freed, then
        # release the temp file the route relinquished to us for streaming.
        if not fut.done():
            await fut
        try:
            os.unlink(path)
        except OSError:
            logger.debug("temp file already gone on stream drain: %s", path)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/v1/models")
async def list_models():
    """List available models (OpenAI-compatible)."""
    try:
        created_at = int(time.time())
    except Exception:
        created_at = 0
    return {
        "object": "list",
        "data": [
            {
                "id": settings.model_name,
                "object": "model",
                "created": created_at,
                "owned_by": "faster-whisper",
            }
        ],
    }


@app.get("/health")
async def health():
    """Liveness + model/eviction visibility. `loaded` is False after an idle
    eviction (the next request wakes the model). Health probes deliberately do
    NOT touch the idle timer, so they never reset the eviction window."""
    return {
        "status": "ok",
        "model": settings.model_name,
        "compute_type": settings.compute_type,
        "loaded": store.loaded,
        "idle_unload_s": settings.idle_unload_s,
        "last_request_age_s": round(store.last_request_age(), 1),
        "unloads": store.unloads,
        "reloads": store.reloads,
        "default_language": settings.default_language,
        "allowed_languages": settings.allowed_languages,
        "max_concurrent": settings.max_concurrent,
        "max_upload_mb": settings.max_upload_mb,
        "max_audio_seconds": settings.max_audio_seconds,
        "request_timeout_s": settings.request_timeout_s,
        "beam_size": settings.beam_size,
        "best_of": settings.best_of,
        "temperature_schedule": settings.temperature_schedule or None,
        "cpu_threads": settings.cpu_threads,
    }


@app.post("/v1/audio/transcriptions")
async def transcribe_audio(
    request: Request,
    file: UploadFile = File(...),
    # Scalar params must be declared as Form fields — without Form(), FastAPI
    # binds them as query params and silently ignores multipart form values.
    model: Optional[str] = Form(None),
    language: Optional[str] = Form(None),
    prompt: Optional[str] = Form(None),
    response_format: Optional[str] = Form("json"),
    temperature: Optional[float] = Form(0.0),
    timestamps: Optional[bool] = Form(True),
    stream: Optional[bool] = Form(False),
):
    """
    Transcribe audio into the input language (OpenAI-compatible).

    Send the audio as multipart/form-data with the field name `file`.
    """
    # Fail fast on unsupported formats, before touching the model (OpenAI: 400).
    fmt = _validate_response_format(response_format)
    limit = _upload_limit()
    _check_content_length(request.headers.get("content-length"), limit)

    # Record real API demand (never /health), then capture the model into a
    # local ref so a concurrent idle-eviction can't tear it down mid-request.
    store.touch()
    wm = await _offload(store.get)

    tmp_path: Optional[str] = None
    try:
        # 1. Stream the upload to disk (bounded RAM; 413 mid-stream past cap)
        #    under a sanitized extension that mirrors the real upload. PyAV
        #    sniffs the bytes, so the extension is hygiene, not logic.
        suffix = _safe_suffix(file.filename)
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            await _write_upload(file, tmp, limit)
            tmp_path = tmp.name

        # 2. Reject audio longer than the cap before touching the model.
        await _guard_duration_ok(tmp_path)

        # 3. Build transcription kwargs. Client `language` wins; otherwise
        #    server defaults/allowlist apply. Client `prompt` wins; otherwise
        #    a server-wide initial prompt applies. faster-whisper names the
        #    prompt kwarg `initial_prompt` (OpenAI calls it `prompt`).
        resolved_language = await _offload(_resolve_language, wm, tmp_path, language)
        resolved_prompt = prompt or settings.initial_prompt
        kwargs = _build_kwargs(
            language=resolved_language,
            prompt=resolved_prompt,
            temperature=temperature if temperature is not None else 0.0,
        )

        # 4. Optional OpenAI-style streaming (SSE); otherwise run inference
        #    off the event loop with the configured abandon-timeout.
        if stream:
            return StreamingResponse(
                _sse_stream(wm, tmp_path, kwargs, task="transcribe"),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache"},
            )

        segment_list, info, full_text = await _maybe_timeout(
            _offload(_transcribe_collect, wm, tmp_path, kwargs),
            settings.request_timeout_s,
        )

        # 5. Plain-text / subtitle formats return the document itself (OpenAI style)
        if fmt in ("txt", "srt", "vtt"):
            return _text_response(fmt, segment_list, full_text)

        return TranscriptionResponse(
            text=full_text,
            task="transcribe",
            language=info.language,
            segments=segment_list if timestamps else None,
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        # For stream=true the temp file is owned by _sse_stream (which unlinks it
        # after draining); here we only clean up the non-streaming path.
        if tmp_path and not stream:
            try:
                os.unlink(tmp_path)
            except OSError:
                logger.debug("temp file already gone (transcribe): %s", tmp_path)


@app.post("/v1/audio/translations")
async def translate_audio(
    request: Request,
    file: UploadFile = File(...),
    model: Optional[str] = Form(None),
    prompt: Optional[str] = Form(None),
    response_format: Optional[str] = Form("json"),
    temperature: Optional[float] = Form(0.0),
    timestamps: Optional[bool] = Form(True),
    stream: Optional[bool] = Form(False),
):
    """
    Translate audio into English (OpenAI-compatible).

    Send the audio as multipart/form-data with the field name `file`.
    """
    fmt = _validate_response_format(response_format)
    limit = _upload_limit()
    _check_content_length(request.headers.get("content-length"), limit)

    # Record real API demand (never /health), then capture the model into a
    # local ref so a concurrent idle-eviction can't tear it down mid-request.
    store.touch()
    wm = await _offload(store.get)

    tmp_path: Optional[str] = None
    try:
        suffix = _safe_suffix(file.filename)
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            await _write_upload(file, tmp, limit)
            tmp_path = tmp.name

        await _guard_duration_ok(tmp_path)

        # Server defaults/allowlist steer source-language detection
        # (OpenAI's translations endpoint always translates *to* English).
        resolved_language = await _offload(_resolve_language, wm, tmp_path, None)
        resolved_prompt = prompt or settings.initial_prompt
        kwargs = _build_kwargs(
            language=resolved_language,
            prompt=resolved_prompt,
            temperature=temperature if temperature is not None else 0.0,
            task="translate",
        )

        if stream:
            return StreamingResponse(
                _sse_stream(wm, tmp_path, kwargs, task="translate"),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache"},
            )

        segment_list, info, full_text = await _maybe_timeout(
            _offload(_transcribe_collect, wm, tmp_path, kwargs),
            settings.request_timeout_s,
        )

        if fmt in ("txt", "srt", "vtt"):
            return _text_response(fmt, segment_list, full_text)

        return TranslationResponse(
            text=full_text,
            task="translate",
            language=info.language,
            segments=segment_list if timestamps else None,
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        # stream=true -> _sse_stream owns cleanup of the temp file.
        if tmp_path and not stream:
            try:
                os.unlink(tmp_path)
            except OSError:
                logger.debug("temp file already gone (translate): %s", tmp_path)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    import uvicorn

    print(
        f"[server] Starting OpenAI-compatible Whisper server\n"
        f"[server] Model: {settings.model_name} | Device: {settings.device} | "
        f"Compute: {settings.compute_type}\n"
        f"[server] Endpoints:\n"
        f"  POST /v1/audio/transcriptions\n"
        f"  POST /v1/audio/translations\n"
        f"  GET  /v1/models\n"
        f"  GET  /health\n"
    )
    uvicorn.run(app, host=settings.host, port=settings.port)


if __name__ == "__main__":
    main()
