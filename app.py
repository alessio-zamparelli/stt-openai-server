"""
OpenAI-compatible Whisper API server backed by faster-whisper.

Exposes /v1/audio/transcriptions and /v1/audio/translations
with defaults matching the OpenAI Whisper API, using the
'base' model with INT8 quantization.
"""

import ctypes
import gc
import logging
import os
import tempfile
import threading
import time
from contextlib import asynccontextmanager
from typing import Any, Callable, List, Optional

from fastapi import FastAPI, File, Form, UploadFile, HTTPException, Request
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field

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


settings = Settings(
    model_name=os.getenv("WHISPER_MODEL_NAME", "base"),
    device=os.getenv("WHISPER_DEVICE", "cpu"),
    compute_type=os.getenv("WHISPER_COMPUTE_TYPE", "int8"),
    host=os.getenv("WHISPER_HOST", "0.0.0.0"),
    port=_env_int("WHISPER_PORT", 8080),
    lazy_load=_env_bool("WHISPER_LAZY_LOAD", False),
    idle_unload_s=_env_int("WHISPER_IDLE_UNLOAD_S", 300),
    idle_poll_s=_env_int("WHISPER_IDLE_POLL_S", 30),
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
                    pass
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


def _text_response(fmt: str, segment_list, full_text: str) -> PlainTextResponse:
    """Return plain-text / subtitle responses, OpenAI style."""
    if fmt == "txt":
        return PlainTextResponse(full_text)
    if fmt == "srt":
        return PlainTextResponse(_to_srt(segment_list), media_type="application/x-subrip")
    return PlainTextResponse(_to_vtt(segment_list), media_type="text/vtt")


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
    }


@app.post("/v1/audio/transcriptions", response_model=TranscriptionResponse)
async def transcribe_audio(
    file: UploadFile = File(...),
    # Scalar params must be declared as Form fields — without Form(), FastAPI
    # binds them as query params and silently ignores multipart form values.
    model: Optional[str] = Form(None),
    language: Optional[str] = Form(None),
    prompt: Optional[str] = Form(None),
    response_format: Optional[str] = Form("json"),
    temperature: Optional[float] = Form(0.0),
    timestamps: Optional[bool] = Form(True),
):
    """
    Transcribe audio into the input language (OpenAI-compatible).

    Send the audio as multipart/form-data with the field name `file`.
    """
    # Fail fast on unsupported formats, before touching the model (OpenAI: 400).
    fmt = _validate_response_format(response_format)

    # Record real API demand (never /health), then capture the model into a
    # local ref so a concurrent idle-eviction can't tear it down mid-request.
    store.touch()
    wm = store.get()

    # 1. Save uploaded file to a temp location
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        # 2. Build transcription kwargs
        kwargs: dict = {
            "beam_size": 5,
            "best_of": 5,
            "condition_on_previous_text": False,
            "temperature": temperature if temperature is not None else 0.0,
        }
        if language:
            kwargs["language"] = language
        if prompt:
            kwargs["prompt"] = prompt

        # 3. Run inference
        segments, info = wm.transcribe(tmp_path, **kwargs)

        # 4. Collect segments
        segment_list = []
        full_text_parts = []
        for seg in segments:
            segment_dict = {
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
            segment_list.append(segment_dict)
            full_text_parts.append(seg.text)

        full_text = "".join(full_text_parts).strip()

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
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


@app.post("/v1/audio/translations", response_model=TranslationResponse)
async def translate_audio(
    file: UploadFile = File(...),
    model: Optional[str] = Form(None),
    prompt: Optional[str] = Form(None),
    response_format: Optional[str] = Form("json"),
    temperature: Optional[float] = Form(0.0),
    timestamps: Optional[bool] = Form(True),
):
    """
    Translate audio into English (OpenAI-compatible).

    Send the audio as multipart/form-data with the field name `file`.
    """
    fmt = _validate_response_format(response_format)

    # Record real API demand (never /health), then capture the model into a
    # local ref so a concurrent idle-eviction can't tear it down mid-request.
    store.touch()
    wm = store.get()

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        kwargs: dict = {
            "beam_size": 5,
            "best_of": 5,
            "condition_on_previous_text": False,
            "temperature": temperature if temperature is not None else 0.0,
            "task": "translate",
        }
        if prompt:
            kwargs["prompt"] = prompt

        segments, info = wm.transcribe(tmp_path, **kwargs)

        segment_list = []
        full_text_parts = []
        for seg in segments:
            segment_dict = {
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
            segment_list.append(segment_dict)
            full_text_parts.append(seg.text)

        full_text = "".join(full_text_parts).strip()

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
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


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
