"""Process-wide runtime state: settings, model store, inference pool, timings.

Services and routers import this module (``from . import runtime``) and read
``runtime.settings`` / ``runtime.store`` at call time. Do not re-bind those
names with ``from .runtime import settings`` — tests and the idle watchdog rely
on late binding.
"""

from __future__ import annotations

import asyncio
import threading
from collections import defaultdict, deque
from concurrent import futures
from typing import Any, Callable, Optional

from starlette.concurrency import run_in_threadpool

from .config import load_settings
from .services.model_store import ModelStore

# ---------------------------------------------------------------------------
# Singletons
# ---------------------------------------------------------------------------

settings = load_settings()
store = ModelStore(settings)

# Set by the lifespan when the Wyoming listener is enabled.
wyoming_service: Any = None

# ---------------------------------------------------------------------------
# Inference pool + shared-model serialization
# ---------------------------------------------------------------------------

_inference_executor: Optional[futures.ThreadPoolExecutor] = None
_inference_pool_lock = threading.Lock()
# Serializer for the single shared model. faster-whisper's WhisperModel is NOT
# safe for concurrent transcribe() on one instance, and the decode is lazy (it
# runs on segment iteration), so the lock wraps BOTH the call and the full
# iteration. Concurrent requests (bounded upstream by STT_MAX_CONCURRENT)
# queue here instead of racing the model — docs/PLAN-robustness-concurrency.md.
_inference_lock = threading.Lock()


def _build_inference_pool_locked() -> None:
    """Create the pool from the current settings. Caller must hold the lock."""
    global _inference_executor
    if _inference_executor is None and settings.max_concurrent > 0:
        _inference_executor = futures.ThreadPoolExecutor(
            max_workers=settings.max_concurrent,
            thread_name_prefix="stt-infer",
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

    Uses the bounded inference pool when STT_MAX_CONCURRENT > 0 (backpressures
    excess requests by queueing), else starlette's anyio pool for an unbounded
    thread-per-request model.
    """
    pool = _inference_pool()
    if pool is None:
        return await run_in_threadpool(fn, *args)
    return await asyncio.get_running_loop().run_in_executor(pool, fn, *args)


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
            from fastapi import HTTPException

            raise HTTPException(
                status_code=504,
                detail=f"Transcription timed out after {timeout_s:g}s.",
            )
    return await awaitable


# ---------------------------------------------------------------------------
# Rolling stage timings (upload / duration / language / inference / total)
# ---------------------------------------------------------------------------
# Circular buffers of millisecond samples per stage; /health reports rolling
# p50/p95 so operators can see where latency goes without a metrics agent.
# Thread-safe: route coroutines and worker threads may record concurrently.
_STAGE_WINDOW = 200
_stage_samples: dict = defaultdict(deque)
_timings_lock = threading.Lock()


def _record_stage(stage: str, seconds: float) -> None:
    with _timings_lock:
        buf = _stage_samples[stage]
        buf.append(seconds * 1000.0)
        if len(buf) > _STAGE_WINDOW:
            buf.popleft()


def _percentile_sorted_ms(vals: list, pct: float) -> float:
    """Nearest-rank percentile of an already-sorted sample list (ms)."""
    if not vals:
        return 0.0
    idx = round((len(vals) - 1) * pct)  # round() returns an int index
    return round(vals[idx], 1)


def _stage_stats() -> dict:
    with _timings_lock:
        out = {}
        for stage, buf in _stage_samples.items():
            vals = sorted(buf)
            out[stage] = {
                "n": len(vals),
                "p50_ms": _percentile_sorted_ms(vals, 0.5),
                "p95_ms": _percentile_sorted_ms(vals, 0.95),
            }
        return out
