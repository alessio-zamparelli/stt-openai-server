"""FastAPI application, lifespan, and CLI entrypoint."""

from __future__ import annotations

import logging
import sys
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI

from . import __version__, runtime
from .routers import system, transcriptions, translations

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not runtime.settings.lazy_load:
        runtime.store.get()  # eager pre-load at startup
        print(
            f"[server] faster-whisper model '{runtime.settings.model_name}' "
            f"on {runtime.settings.device} ({runtime.settings.compute_type})"
        )

    # Eagerly create the bounded inference pool before any request can hit it
    # (docs/PLAN-performance.md Phase 0: the lazy path can never race).
    runtime._init_inference_pool()

    # RAM-reclamation watchdog: evict the idle model (PLAN-idle-unload.md).
    # Only spawned when eviction is enabled (idle_unload_s > 0). Health probes
    # never call store.touch(), so they don't reset the idle window.
    stop = threading.Event()
    app.state._idle_stop = stop
    if runtime.store.eviction_enabled:
        threading.Thread(target=_idle_watchdog, args=(stop,), daemon=True).start()

    # Optional Wyoming ASR listener (Home Assistant). Started after the pool so
    # its inference calls share the same bounded executor.
    if runtime.settings.wyoming:
        from .services.wyoming import WyomingService

        runtime.wyoming_service = WyomingService()
        await runtime.wyoming_service.start()
        print(
            f"[server] Wyoming ASR on tcp://{runtime.settings.wyoming_host}:"
            f"{runtime.settings.wyoming_port} "
            f"(zeroconf={'on' if runtime.settings.wyoming_zeroconf else 'off'})"
        )

    try:
        yield
    finally:
        if runtime.wyoming_service is not None:
            try:
                await runtime.wyoming_service.stop()
            except Exception:  # pragma: no cover - defensive
                logger.exception("Wyoming shutdown failed")
            finally:
                runtime.wyoming_service = None
        stop.set()
        runtime._reset_inference_pool()


def _idle_watchdog(stop: threading.Event) -> None:
    """Periodically evict the model after ``idle_unload_s`` without API requests."""
    settings = runtime.settings
    interval = min(settings.idle_poll_s, max(5, settings.idle_unload_s // 2))
    while not stop.wait(interval):
        try:
            runtime.store.maybe_unload()
        except Exception:  # pragma: no cover - defensive
            logger.exception("idle-unload watchdog failed")


app = FastAPI(
    title="OpenAI-Compatible STT API",
    description=(
        "faster-whisper backend exposing OpenAI-compatible "
        "/v1/audio/transcriptions and /v1/audio/translations endpoints."
    ),
    version=__version__,
    lifespan=lifespan,
)

app.include_router(transcriptions.router)
app.include_router(translations.router)
app.include_router(system.router)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def healthcheck() -> int:
    """One-shot probe used by the container HEALTHCHECK (`app.py --healthcheck`).

    Uses ``http.client`` instead of ``urllib`` so a loopback probe is never
    routed through an HTTP(S) proxy from the container environment, honors
    ``STT_PORT``/``STT_HOST`` (including a 0.0.0.0 bind), and prints a single
    line instead of a full traceback while the model is still downloading
    during Docker's start period.
    """
    import http.client

    settings = runtime.settings
    host = "127.0.0.1" if settings.host in ("", "0.0.0.0", "::") else settings.host
    conn = None
    try:
        conn = http.client.HTTPConnection(host, settings.port, timeout=4)
        conn.request("GET", "/health")
        response = conn.getresponse()
        response.read()
        return 0 if response.status == 200 else 1
    except Exception as exc:  # ConnectionRefusedError while uvicorn is not bound yet
        print(
            f"healthcheck: {host}:{settings.port} not ready "
            f"({type(exc).__name__}: {exc})",
            file=sys.stderr,
        )
        return 1
    finally:
        if conn is not None:
            conn.close()


def main() -> None:
    import uvicorn

    settings = runtime.settings
    if settings.wyoming:
        wyoming_line = (
            f"  tcp://{settings.wyoming_host}:{settings.wyoming_port} "
            "(Wyoming ASR)"
        )
    else:
        wyoming_line = "  Wyoming ASR disabled (STT_WYOMING=true to enable)"
    print(
        f"[server] Starting OpenAI-compatible STT server\n"
        f"[server] Model: {settings.model_name} | Device: {settings.device} | "
        f"Compute: {settings.compute_type}\n"
        f"[server] Endpoints:\n"
        f"  POST /v1/audio/transcriptions\n"
        f"  POST /v1/audio/translations\n"
        f"  GET  /v1/models\n"
        f"  GET  /health\n"
        f"{wyoming_line}\n"
    )
    uvicorn.run(app, host=settings.host, port=settings.port)


def run() -> None:
    """CLI entrypoint used by ``app.py`` and the ``stt-api`` console script."""
    if "--healthcheck" in sys.argv[1:]:
        raise SystemExit(healthcheck())
    main()
