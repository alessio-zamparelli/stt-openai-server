"""Decoding kwargs, batched-runner selection, collection, and SSE streaming."""

from __future__ import annotations

import asyncio
import json
import logging
import queue
from typing import Any, Optional

from starlette.concurrency import run_in_threadpool

from .. import runtime

logger = logging.getLogger(__name__)


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
    """Parse STT_TEMPERATURES ('0,0.2,0.4') -> [0.0, 0.2, 0.4]; None = default.

    Bad input falls back to the faster-whisper default schedule rather than
    failing startup or reverting to a single pass.
    """
    raw = (raw or "").strip()
    if not raw:
        return None
    try:
        vals = [float(v) for v in raw.split(",") if v.strip()]
    except ValueError:
        logger.warning("ignoring bad STT_TEMPERATURES=%r", raw)
        return None
    return vals or None


def _build_kwargs(
    language: Optional[str],
    prompt: Optional[str],
    temperature: float,
    task: Optional[str] = None,
    batched: bool = False,
    without_timestamps: bool = False,
) -> dict:
    """Shared transcription kwargs (single source of truth for both endpoints)."""
    settings = runtime.settings
    kwargs: dict = {
        "beam_size": settings.beam_size,
        "best_of": settings.best_of,
        "condition_on_previous_text": False,
        "temperature": temperature if temperature is not None else 0.0,
        # Text-only requests (text / timestamps=false) skip timestamp-token
        # prediction. Batched mode defaults this to true, so pass it explicitly.
        "without_timestamps": without_timestamps,
    }
    # A server-wide fallback schedule (STT_TEMPERATURES) replaces the
    # per-request scalar when set — documented tradeoff, PLAN-performance 1B.
    schedule = _parse_temperatures(settings.temperature_schedule)
    if schedule:
        kwargs["temperature"] = schedule
    if batched:
        kwargs["batch_size"] = settings.batch_size
        # BatchedInferencePipeline needs VAD (or clip_timestamps) to find speech
        # chunks; without it transcribe() raises. Its default is True, so pass it
        # explicitly: STT_VAD only controls the sequential path.
        kwargs["vad_filter"] = True
    elif settings.vad_filter:
        kwargs["vad_filter"] = True
    if task:
        kwargs["task"] = task
    if language:
        kwargs["language"] = language
    if prompt:
        kwargs["initial_prompt"] = prompt
    return kwargs


def _use_batched(duration: Optional[float]) -> bool:
    """Whether this request should use BatchedInferencePipeline.

    ``STT_BATCH_SIZE > 0`` enables batching; ``STT_BATCH_MIN_SECONDS``
    (``> 0``) restricts it to clips at least that long. Unknown duration falls
    back to the sequential path (docs/PLAN-performance.md Phase 2).
    """
    settings = runtime.settings
    if settings.batch_size <= 0:
        return False
    if settings.batch_min_seconds <= 0:
        return True
    return duration is not None and duration >= settings.batch_min_seconds


def _build_runner(wm: Any, duration: Optional[float] = None) -> Any:
    """Model or opt-in BatchedInferencePipeline wrapper.

    Selected when ``STT_BATCH_SIZE > 0`` and, if ``STT_BATCH_MIN_SECONDS > 0``,
    the clip is at least that long. Batched mode chunks the ~30 s windows itself
    and does not support condition_on_previous_text, so it can change
    segmentation/casing — strict opt-in (docs/PLAN-performance.md Phase 2).
    Either form exposes .transcribe, so the collection/streaming units are
    runner-agnostic.
    """
    if _use_batched(duration):
        from faster_whisper import BatchedInferencePipeline

        return BatchedInferencePipeline(model=wm)
    return wm


def _transcribe_collect(
    runner: Any,
    audio: Any,
    kwargs: dict,
    collect_segments: bool = True,
) -> tuple:
    """Blocking unit: transcribe + aggregate segments/text (worker thread).

    Serialized on the shared model — ``runner.transcribe`` is lazy, so the
    lock covers the segment iteration too (that is where the decode runs).
    With ``collect_segments=False`` (text output / timestamps=false) segment
    dicts and token lists are never built.
    """
    from .audio import _seek_start

    with runtime._inference_lock:
        _seek_start(audio)
        segments, info = runner.transcribe(audio, **kwargs)
        if collect_segments:
            segment_list = [_segment_dict(seg) for seg in segments]
            parts = [seg["text"] for seg in segment_list]
        else:
            segment_list = []
            parts = [seg.text for seg in segments]
    full_text = "".join(parts).strip()
    return segment_list, info, full_text


# -- OpenAI-style streaming (stream=true) ----------------------------------
#
# faster-whisper yields segments lazily, so a producer thread pushes each as it
# is produced and the SSE generator forwards it. The wire format is best-effort
# OpenAI-shaped (transcript deltas, transcript.done, [DONE]); verify against a
# real client before promising strict parity.


def _stream_worker(runner: Any, audio: Any, kwargs: dict, q: "queue.SimpleQueue") -> None:
    """Producer: ('segment', dict) / ('done', (info, text)) / ('error', msg) / ('end', None).

    Serialized like ``_transcribe_collect`` (shared model, lazy decode).
    """
    from .audio import _seek_start

    try:
        with runtime._inference_lock:
            _seek_start(audio)
            segments, info = runner.transcribe(audio, **kwargs)
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


async def _sse_stream(runner: Any, audio: Any, kwargs: dict, task: str, upload) -> Any:
    """SSE generator: per-segment deltas, then transcript.done + [DONE]."""
    q: "queue.SimpleQueue" = queue.SimpleQueue()
    loop = asyncio.get_running_loop()
    fut = loop.run_in_executor(runtime._inference_pool(), _stream_worker, runner, audio, kwargs, q)
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
        # release the buffered upload the route relinquished to us.
        if not fut.done():
            await fut
        try:
            await upload.close()
        except Exception:  # pragma: no cover - defensive
            logger.debug("upload already closed on stream drain", exc_info=True)
