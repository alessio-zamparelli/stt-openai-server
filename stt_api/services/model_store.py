"""faster-whisper model lifecycle: idle eviction + single-flight lazy reload."""

from __future__ import annotations

import ctypes
import gc
import logging
import threading
import time
from typing import Any, Callable, Optional

from ..config import Settings

logger = logging.getLogger(__name__)


def _load_model(config: Optional[Settings] = None):
    """Build the faster-whisper model.

    ``config`` defaults to the process-wide runtime settings, read lazily to
    avoid an import cycle between this module and ``stt_api.runtime``.
    """
    from faster_whisper import WhisperModel

    if config is None:
        from .. import runtime

        config = runtime.settings

    return WhisperModel(
        config.model_name,
        device=config.device,
        compute_type=config.compute_type,
        cpu_threads=config.cpu_threads,
        local_files_only=config.hf_offline,  # offline: never hit the HF hub
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

    def peek(self) -> Any:
        """Return the resident model without triggering a load (may be None)."""
        return self._model

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
