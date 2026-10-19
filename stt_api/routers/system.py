"""GET /v1/models and GET /health."""

from __future__ import annotations

import time

from fastapi import APIRouter, Depends

from .. import runtime
from ..config import Settings
from ..dependencies import get_settings, get_store
from ..services.model_store import ModelStore

router = APIRouter(tags=["system"])


@router.get("/v1/models")
async def list_models(settings: Settings = Depends(get_settings)):
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


@router.get("/health")
async def health(
    settings: Settings = Depends(get_settings),
    store: ModelStore = Depends(get_store),
):
    """Liveness + model/eviction visibility. `loaded` is False after an idle
    eviction (the next request wakes the model). Health probes deliberately do
    NOT touch the idle timer, so they never reset the eviction window."""
    service = runtime.wyoming_service
    if service is not None:
        wyoming = service.stats()
    else:
        wyoming = {
            "enabled": settings.wyoming,
            "host": settings.wyoming_host,
            "port": settings.wyoming_port,
            "zeroconf": settings.wyoming_zeroconf,
            "name": settings.wyoming_name,
            "requests": 0,
            "errors": 0,
            "active": 0,
        }
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
        "batch_size": settings.batch_size,
        "batch_min_seconds": settings.batch_min_seconds,
        "vad_filter": settings.vad_filter,
        "hf_offline": settings.hf_offline,
        "wyoming": wyoming,
        "latency_ms": runtime._stage_stats(),
    }
