"""POST /v1/audio/translations."""

from __future__ import annotations

import logging
import time
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import StreamingResponse

from .. import runtime
from ..config import Settings
from ..dependencies import get_settings, get_store
from ..schemas import TranslationResponse
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


@router.post("/translations")
async def translate_audio(
    request: Request,
    file: UploadFile = File(...),
    model: Optional[str] = Form(None),
    prompt: Optional[str] = Form(None),
    response_format: Optional[str] = Form("json"),
    temperature: Optional[float] = Form(0.0),
    timestamps: Optional[bool] = Form(True),
    stream: Optional[bool] = Form(False),
    settings: Settings = Depends(get_settings),
    store: ModelStore = Depends(get_store),
):
    """
    Translate audio into English (OpenAI-compatible).

    Send the audio as multipart/form-data with the field name `file`.
    """
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
        _t_up = time.perf_counter()
        _reject_too_large(file.size or 0, limit)
        runtime._record_stage("upload", time.perf_counter() - _t_up)

        _t_dur = time.perf_counter()
        needs_duration = settings.batch_size > 0 and settings.batch_min_seconds > 0
        duration = await _guard_duration_ok(file.file, probe=needs_duration)
        runtime._record_stage("duration", time.perf_counter() - _t_dur)

        # Server defaults/allowlist steer source-language detection
        # (OpenAI's translations endpoint always translates *to* English).
        _t_lang = time.perf_counter()
        resolved_language = await runtime._offload(
            _resolve_language, wm, file.file, None
        )
        runtime._record_stage("language", time.perf_counter() - _t_lang)
        resolved_prompt = prompt or settings.initial_prompt
        needs_segments = fmt in ("srt", "vtt") or (fmt == "json" and bool(timestamps))
        without_timestamps = not needs_segments and not stream
        batched = _use_batched(duration)
        runner = _build_runner(wm, duration)
        kwargs = _build_kwargs(
            language=resolved_language,
            prompt=resolved_prompt,
            temperature=temperature if temperature is not None else 0.0,
            task="translate",
            batched=batched,
            without_timestamps=without_timestamps,
        )

        if stream:
            response = StreamingResponse(
                _sse_stream(runner, file.file, kwargs, task="translate", upload=file),
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
        # Rolling p50/p95 stage timings; stream paths record setup stages only.
        runtime._record_stage("total", time.perf_counter() - _t_route)
        # stream=true -> _sse_stream owns (and closes) the buffered upload.
        if not stream_owned:
            try:
                await file.close()
            except Exception:  # pragma: no cover - defensive
                logger.debug("upload close failed (translate)", exc_info=True)
