"""POST /v1/audio/transcriptions."""

from __future__ import annotations

import logging
import time
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import StreamingResponse

from .. import runtime
from ..config import Settings
from ..dependencies import get_settings, get_store
from ..schemas import TranscriptionResponse
from ..services.audio import _check_content_length, _guard_duration_ok, _reject_too_large, _upload_limit
from ..services.formatting import _text_response, _validate_response_format
from ..services.inference import (
    _build_kwargs,
    _build_runner,
    _sse_stream,
    _transcribe_collect,
    _use_batched,
)
from ..services.language import _resolve_language
from ..services.model_store import ModelStore

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/audio", tags=["audio"])


@router.post("/transcriptions")
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
    settings: Settings = Depends(get_settings),
    store: ModelStore = Depends(get_store),
):
    """
    Transcribe audio into the input language (OpenAI-compatible).

    Send the audio as multipart/form-data with the field name `file`.
    """
    # Fail fast on unsupported formats, before touching the model (OpenAI: 400).
    fmt = _validate_response_format(response_format)
    limit = _upload_limit()
    _check_content_length(request.headers.get("content-length"), limit)
    _t_route = time.perf_counter()

    # Record real API demand (never /health), then capture the model into a
    # local ref so a concurrent idle-eviction can't tear it down mid-request.
    store.touch()
    wm = await runtime._offload(store.get)

    stream_owned = False
    try:
        # 1. Validate the upload. Starlette has already parsed the multipart
        #    body into file.file (a SpooledTemporaryFile), so the route hands
        #    that to PyAV/faster-whisper directly instead of copying it again.
        _t_up = time.perf_counter()
        _reject_too_large(file.size or 0, limit)
        runtime._record_stage("upload", time.perf_counter() - _t_up)

        # 2. Probe the duration: enforce the cap and, when adaptive batching is
        #    configured, pick the runner. Both use one PyAV metadata read.
        _t_dur = time.perf_counter()
        needs_duration = settings.batch_size > 0 and settings.batch_min_seconds > 0
        duration = await _guard_duration_ok(file.file, probe=needs_duration)
        runtime._record_stage("duration", time.perf_counter() - _t_dur)

        # 3. Build transcription kwargs. Client `language` wins; otherwise
        #    server defaults/allowlist apply. Client `prompt` wins; otherwise
        #    a server-wide initial prompt applies. faster-whisper names the
        #    prompt kwarg `initial_prompt` (OpenAI calls it `prompt`).
        _t_lang = time.perf_counter()
        resolved_language = await runtime._offload(
            _resolve_language, wm, file.file, language
        )
        runtime._record_stage("language", time.perf_counter() - _t_lang)
        resolved_prompt = prompt or settings.initial_prompt
        # Text-only requests skip timestamp-token prediction and segment dicts.
        needs_segments = fmt in ("srt", "vtt") or (fmt == "json" and bool(timestamps))
        without_timestamps = not needs_segments and not stream
        batched = _use_batched(duration)
        runner = _build_runner(wm, duration)
        kwargs = _build_kwargs(
            language=resolved_language,
            prompt=resolved_prompt,
            temperature=temperature if temperature is not None else 0.0,
            batched=batched,
            without_timestamps=without_timestamps,
        )

        # 4. Optional OpenAI-style streaming (SSE); otherwise run inference
        #    off the event loop with the configured abandon-timeout.
        if stream:
            response = StreamingResponse(
                _sse_stream(runner, file.file, kwargs, task="transcribe", upload=file),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache"},
            )
            stream_owned = True
            return response

        _t_inf = time.perf_counter()
        segment_list, info, full_text = await runtime._maybe_timeout(
            runtime._offload(_transcribe_collect, runner, file.file, kwargs, needs_segments),
            settings.request_timeout_s,
        )
        runtime._record_stage("inference", time.perf_counter() - _t_inf)

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
        # Rolling p50/p95 stage timings; stream paths record setup stages only
        # (inference finishes after the streaming response returns).
        runtime._record_stage("total", time.perf_counter() - _t_route)
        # For stream=true _sse_stream owns the upload and closes it after
        # draining; otherwise close it here (FastAPI also auto-closes).
        if not stream_owned:
            try:
                await file.close()
            except Exception:  # pragma: no cover - defensive
                logger.debug("upload close failed (transcribe)", exc_info=True)
