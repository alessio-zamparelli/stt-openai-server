"""Server / model configuration.

Canonical environment variables use the ``STT_*`` prefix. The historical
``WHISPER_*`` names are still accepted as deprecated aliases (``STT_*`` wins
when both are set) so existing deployments keep working after the rename.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

from pydantic import BaseModel

logger = logging.getLogger(__name__)

CANONICAL_PREFIX = "STT_"
LEGACY_PREFIX = "WHISPER_"

_warned_legacy: set[str] = set()


class Settings(BaseModel):
    """Server / model configuration — overridable via ``STT_*`` env vars."""

    model_name: str = "base"  # STT_MODEL_NAME
    device: str = "cpu"  # STT_DEVICE
    compute_type: str = "int8"  # STT_COMPUTE_TYPE
    host: str = "0.0.0.0"  # STT_HOST
    port: int = 8080  # STT_PORT
    lazy_load: bool = False  # STT_LAZY_LOAD
    idle_unload_s: int = 300  # STT_IDLE_UNLOAD_S; evict after this much idle; 0 = off
    idle_poll_s: int = 30  # STT_IDLE_POLL_S (watchdog cadence; only when idle_unload_s > 0)
    # Language handling. Priority per request: client `language` field, then
    # `default_language`, then (with a multi-code `allowed_languages`) a
    # detection constrained to the allowlist, then faster-whisper full auto.
    default_language: Optional[str] = None  # STT_LANGUAGE
    allowed_languages: Optional[str] = None  # STT_LANGUAGES  e.g. "it,en"
    initial_prompt: Optional[str] = None  # STT_INITIAL_PROMPT
    # Robustness / resource limits (docs/PLAN-robustness-concurrency.md).
    max_concurrent: int = 2  # STT_MAX_CONCURRENT (0 = unbounded threadpool)
    max_upload_mb: int = 100  # STT_MAX_UPLOAD_MB (0 = unlimited)
    max_audio_seconds: int = 3600  # STT_MAX_AUDIO_SECONDS (0 = off)
    request_timeout_s: float = 300.0  # STT_REQUEST_TIMEOUT_S (0 = off)
    # Inference tuning (docs/PLAN-performance.md Phase 1). Accuracy-first
    # defaults; the knobs exist for latency-sensitive deployments.
    beam_size: int = 5  # STT_BEAM_SIZE
    best_of: int = 5  # STT_BEST_OF
    temperature_schedule: str = ""  # STT_TEMPERATURES e.g. "0,0.2,0.4" ("" = fw default)
    cpu_threads: int = 0  # STT_CPU_THREADS (0 = faster-whisper default)
    batch_size: int = 0  # STT_BATCH_SIZE (0 = sequential; >0 = batched pipeline)
    # With batch_size > 0, batch only clips at least this long (0 = always batch).
    # Unknown duration falls back to sequential (docs/PLAN-performance.md Phase 2).
    batch_min_seconds: int = 0  # STT_BATCH_MIN_SECONDS
    vad_filter: bool = False  # STT_VAD (enable VAD pre-filtering at transcribe time)
    hf_offline: bool = False  # STT_HF_OFFLINE (local_files_only on model load)
    # Wyoming ASR protocol (Home Assistant). Opt-in; see services/wyoming.py.
    wyoming: bool = False  # STT_WYOMING
    wyoming_host: str = "0.0.0.0"  # STT_WYOMING_HOST
    wyoming_port: int = 10300  # STT_WYOMING_PORT
    wyoming_zeroconf: bool = False  # STT_WYOMING_ZEROCONF
    wyoming_name: Optional[str] = None  # STT_WYOMING_NAME


def _raw_env(suffix: str) -> Optional[str]:
    """Read ``STT_<suffix>``, falling back to the legacy ``WHISPER_<suffix>``.

    The fallback is logged once per variable so operators can migrate without
    flooding logs on every request/startup helper call.
    """
    value = os.getenv(CANONICAL_PREFIX + suffix)
    if value is not None:
        return value

    legacy_name = LEGACY_PREFIX + suffix
    value = os.getenv(legacy_name)
    if value is not None and legacy_name not in _warned_legacy:
        _warned_legacy.add(legacy_name)
        logger.warning(
            "%s is deprecated; use %s%s",
            legacy_name,
            CANONICAL_PREFIX,
            suffix,
        )
    return value


def _env_str(suffix: str, default: str) -> str:
    value = _raw_env(suffix)
    return default if value is None else value


def _env_bool(suffix: str, default: bool) -> bool:
    raw = _raw_env(suffix)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _env_int(suffix: str, default: int) -> int:
    """Parse an integer env var, falling back to the default on bad input."""
    try:
        return int(_raw_env(suffix) or default)
    except (TypeError, ValueError):
        return default


def _env_float(suffix: str, default: float) -> float:
    """Parse a float env var, falling back to the default on bad input."""
    try:
        return float(_raw_env(suffix) or default)
    except (TypeError, ValueError):
        return default


def load_settings() -> Settings:
    """Build a Settings instance from the current environment."""
    return Settings(
        model_name=_env_str("MODEL_NAME", "base"),
        device=_env_str("DEVICE", "cpu"),
        compute_type=_env_str("COMPUTE_TYPE", "int8"),
        host=_env_str("HOST", "0.0.0.0"),
        port=_env_int("PORT", 8080),
        lazy_load=_env_bool("LAZY_LOAD", False),
        idle_unload_s=_env_int("IDLE_UNLOAD_S", 300),
        idle_poll_s=_env_int("IDLE_POLL_S", 30),
        default_language=_raw_env("LANGUAGE") or None,
        allowed_languages=_raw_env("LANGUAGES") or None,
        initial_prompt=_raw_env("INITIAL_PROMPT") or None,
        max_concurrent=_env_int("MAX_CONCURRENT", 2),
        max_upload_mb=_env_int("MAX_UPLOAD_MB", 100),
        max_audio_seconds=_env_int("MAX_AUDIO_SECONDS", 3600),
        request_timeout_s=_env_float("REQUEST_TIMEOUT_S", 300.0),
        beam_size=_env_int("BEAM_SIZE", 5),
        best_of=_env_int("BEST_OF", 5),
        temperature_schedule=(_raw_env("TEMPERATURES") or "").strip(),
        cpu_threads=_env_int("CPU_THREADS", 0),
        batch_size=_env_int("BATCH_SIZE", 0),
        batch_min_seconds=_env_int("BATCH_MIN_SECONDS", 0),
        vad_filter=_env_bool("VAD", False),
        hf_offline=_env_bool("HF_OFFLINE", False),
        wyoming=_env_bool("WYOMING", False),
        wyoming_host=_env_str("WYOMING_HOST", "0.0.0.0"),
        wyoming_port=_env_int("WYOMING_PORT", 10300),
        wyoming_zeroconf=_env_bool("WYOMING_ZEROCONF", False),
        wyoming_name=_raw_env("WYOMING_NAME") or None,
    )
