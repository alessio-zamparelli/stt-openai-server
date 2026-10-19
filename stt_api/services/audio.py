"""Audio probing, upload/duration limits, and prefix decoding."""

from __future__ import annotations

import gc
import logging
from typing import Any, Optional

from fastapi import HTTPException

from .. import runtime

logger = logging.getLogger(__name__)


def _seek_start(audio: Any) -> None:
    """Rewind a file-like upload before a blocking probe/decode (no-op for paths)."""
    seek = getattr(audio, "seek", None)
    if seek is not None:
        seek(0)


def _upload_limit() -> Optional[int]:
    """Upload cap in bytes (None = unlimited)."""
    mb = runtime.settings.max_upload_mb
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


def _audio_duration_s(audio: Any) -> Optional[float]:
    """Container duration via PyAV metadata (fast; no model load). None if unknown."""
    try:
        import av

        _seek_start(audio)
        container = av.open(audio, mode="r", metadata_errors="ignore")
        try:
            stream = container.streams.audio[0]
            if stream.duration and stream.time_base:
                # duration * time_base is a Fraction; coerce to float seconds.
                return float(stream.duration * stream.time_base)
        finally:
            container.close()
    except Exception:
        logger.debug("duration probe failed for %s", audio, exc_info=True)
    return None


async def _guard_duration_ok(audio: Any, probe: bool = False) -> Optional[float]:
    """Enforce STT_MAX_AUDIO_SECONDS and return the probed duration.

    ``probe`` requests the duration even when the guard is disabled (adaptive
    batching needs it). Returns None when not probed or PyAV cannot tell.
    """
    max_s = runtime.settings.max_audio_seconds
    if not (max_s and max_s > 0) and not probe:
        return None
    duration = await runtime._offload(_audio_duration_s, audio)
    if max_s and max_s > 0 and duration is not None and duration > max_s:
        raise HTTPException(
            status_code=400,
            detail=f"Audio file is too long ({duration:.1f}s > {max_s}s).",
        )
    return duration


def _decode_audio_prefix(
    audio: Any, max_seconds: float = 30.0
) -> Optional[Any]:
    """Decode only the prefix needed for language detection (mono f32 @16 kHz).

    ``detect_language`` reads at most one 30 s window, but
    ``faster_whisper.audio.decode_audio`` decodes the whole upload. This mirrors
    its PyAV resampling and stops as soon as enough samples are buffered, so a
    long file no longer pays a second full decode before inference.

    Returns None when no audio could be decoded (caller falls back to
    faster-whisper's own detection during ``transcribe``).
    """
    import av
    import numpy as np

    sampling_rate = 16000
    max_samples = int(max_seconds * sampling_rate)
    resampler = av.audio.resampler.AudioResampler(
        format="s16", layout="mono", rate=sampling_rate
    )
    chunks = []
    seen = 0
    try:
        _seek_start(audio)
        with av.open(audio, mode="r", metadata_errors="ignore") as container:
            frames = iter(container.decode(audio=0))
            while seen < max_samples:
                try:
                    frame = next(frames)
                except StopIteration:
                    break
                except av.error.InvalidDataError:  # mirror faster-whisper
                    continue
                frame.pts = None  # ignore timestamp checks
                for resampled in resampler.resample(frame):
                    array = resampled.to_ndarray()
                    chunks.append(array)
                    seen += array.size
    except Exception:
        logger.debug("language prefix decode failed", exc_info=True)
        return None
    finally:
        # Release resampler internals without waiting for a full GC pass
        # (same workaround as faster_whisper.audio.decode_audio).
        del resampler
        gc.collect()

    if not chunks:
        return None
    samples = np.concatenate([chunk.ravel() for chunk in chunks])[:max_samples]
    return samples.astype(np.float32) / 32768.0
