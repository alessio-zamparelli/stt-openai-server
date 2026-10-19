"""Wyoming ASR protocol server for Home Assistant.

The listener is started from ``stt_api.main.lifespan`` when ``STT_WYOMING=true``.
Each connection handles one utterance: ``transcribe`` + ``audio-start`` +
``audio-chunk*`` + ``audio-stop`` -> ``transcript``, then the server closes the
connection (matching wyoming-faster-whisper).
"""

from __future__ import annotations

import logging
import socket
import tempfile
import time
import uuid
import wave
from functools import partial
from typing import Any, Optional

from fastapi import HTTPException
from wyoming.asr import Transcribe, Transcript
from wyoming.audio import AudioChunk, AudioChunkConverter, AudioStart, AudioStop
from wyoming.event import Event
from wyoming.info import AsrModel, AsrProgram, Attribution, Describe, Info
from wyoming.server import AsyncEventHandler, AsyncTcpServer

from .. import __version__, runtime
from . import inference
from . import language as language_helpers

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
SAMPLE_WIDTH = 2
CHANNELS = 1
BYTES_PER_SECOND = SAMPLE_RATE * SAMPLE_WIDTH * CHANNELS
_SPOOL_MAX = 1024 * 1024  # keep short utterances in memory, long ones on disk
_PROGRAM_NAME = "stt-api"
_WYOMING_SERVICE_TYPE = "_wyoming._tcp.local."


def _model_languages() -> list[str]:
    """Languages advertised to Home Assistant.

    Prefer the resident model (it knows whether an ``.en`` checkpoint is
    monolingual) without triggering a load just for ``describe``.
    """
    model = runtime.store.peek()
    if model is not None:
        codes = language_helpers._supported_language_codes(model)
        if codes:
            return sorted(codes)
    if runtime.settings.model_name.endswith(".en"):
        return ["en"]
    try:
        from faster_whisper.tokenizer import _LANGUAGE_CODES

        return sorted(_LANGUAGE_CODES)
    except Exception:  # pragma: no cover - defensive
        return ["en"]


def build_info() -> Info:
    settings = runtime.settings
    try:
        import faster_whisper

        engine_version = getattr(faster_whisper, "__version__", None)
    except Exception:  # pragma: no cover - defensive
        engine_version = None

    return Info(
        asr=[
            AsrProgram(
                name=_PROGRAM_NAME,
                description="OpenAI-compatible STT server backed by faster-whisper",
                attribution=Attribution(
                    name="SYSTRAN",
                    url="https://github.com/SYSTRAN/faster-whisper",
                ),
                installed=True,
                version=__version__,
                models=[
                    AsrModel(
                        name=settings.model_name,
                        description=(
                            f"{settings.model_name} "
                            f"({settings.compute_type}, {settings.device})"
                        ),
                        attribution=Attribution(
                            name="Systran",
                            url="https://huggingface.co/Systran",
                        ),
                        installed=True,
                        version=engine_version,
                        languages=_model_languages(),
                    )
                ],
                requires_external_vad=True,
                supports_transcript_streaming=False,
            )
        ]
    )


def _normalize_language(value: Optional[str]) -> Optional[str]:
    """Map Home Assistant's language hints to an ISO-639-1 code."""
    if not value:
        return None
    code = value.strip().lower()
    if not code or code == "auto":
        return None
    # Accept region-qualified tags (en-US, zh_HK) and let validation/fallback
    # handle anything that still is not in the model's language set.
    return code.split("-")[0].split("_")[0]


class WyomingAsrHandler(AsyncEventHandler):
    """One Wyoming connection (one utterance)."""

    def __init__(self, service: "WyomingService", reader, writer) -> None:
        super().__init__(reader, writer)
        self._service = service
        self._converter = self._new_converter()
        self._spool = None
        self._wav = None
        self._pcm_bytes = 0
        self._language: Optional[str] = None
        self._overflow = False
        self._got_audio = False
        self._finished = False
        self._counted = False
        service.active += 1

    @staticmethod
    def _new_converter() -> AudioChunkConverter:
        return AudioChunkConverter(rate=SAMPLE_RATE, width=SAMPLE_WIDTH, channels=CHANNELS)

    async def handle_event(self, event: Event) -> bool:
        try:
            if Describe.is_type(event.type):
                await self.write_event(build_info().event())
                return True

            if Transcribe.is_type(event.type):
                transcribe = Transcribe.from_event(event)
                self._language = _normalize_language(transcribe.language)
                # A real transcription request is API demand; describe is not.
                runtime.store.touch()
                return True

            if AudioStart.is_type(event.type):
                self._reset_audio()
                return True

            if AudioChunk.is_type(event.type):
                chunk = self._converter.convert(AudioChunk.from_event(event))
                self._append(chunk.audio)
                return True

            if AudioStop.is_type(event.type):
                await self._finish()
                return False  # one utterance per connection

            # select-program and unknown/future events are ignored.
            return True
        except Exception:
            logger.exception("Wyoming event handling failed")
            self._service.errors += 1
            return False

    async def disconnect(self) -> None:
        self._reset_audio()
        self._decrement_active()

    def _decrement_active(self) -> None:
        if self._counted:
            return
        self._counted = True
        self._service.active = max(0, self._service.active - 1)

    def _close_audio(self) -> None:
        wav, self._wav = self._wav, None
        if wav is not None:
            try:
                wav.close()
            except Exception:  # pragma: no cover - defensive
                logger.debug("WAV finalize failed", exc_info=True)
        spool, self._spool = self._spool, None
        if spool is not None:
            try:
                spool.close()
            except Exception:  # pragma: no cover - defensive
                logger.debug("audio buffer close failed", exc_info=True)
        self._pcm_bytes = 0
        self._got_audio = False

    def _reset_audio(self) -> None:
        self._close_audio()
        self._overflow = False
        self._finished = False
        self._converter = self._new_converter()

    def _append(self, pcm: bytes) -> None:
        if not pcm or self._overflow:
            return

        self._got_audio = True
        self._pcm_bytes += len(pcm)

        settings = runtime.settings
        limits = []
        if settings.max_audio_seconds > 0:
            limits.append(BYTES_PER_SECOND * settings.max_audio_seconds)
        if settings.max_upload_mb > 0:
            limits.append(settings.max_upload_mb * 1024 * 1024)
        limit = min(limits) if limits else None
        if limit is not None and self._pcm_bytes > limit:
            logger.warning(
                "Wyoming utterance exceeds the configured audio limit (%d bytes); "
                "returning an empty transcript",
                self._pcm_bytes,
            )
            self._overflow = True
            self._service.errors += 1
            self._close_audio()
            return

        if self._wav is None:
            self._spool = tempfile.SpooledTemporaryFile(max_size=_SPOOL_MAX, mode="w+b")
            self._wav = wave.open(self._spool, "wb")
            self._wav.setnchannels(CHANNELS)
            self._wav.setsampwidth(SAMPLE_WIDTH)
            self._wav.setframerate(SAMPLE_RATE)
        self._wav.writeframes(pcm)

    async def _finish(self) -> None:
        if self._finished:
            return
        self._finished = True
        self._service.requests += 1
        runtime.store.touch()

        if self._overflow or not self._got_audio or self._wav is None:
            await self._send_transcript("")
            self._reset_audio()
            return

        # Finalize the RIFF header and hand the seekable buffer to the model.
        try:
            self._wav.close()
        except Exception:  # pragma: no cover - defensive
            logger.exception("failed to finalize Wyoming audio")
            await self._send_transcript("")
            self._reset_audio()
            return
        self._wav = None
        spool = self._spool
        self._spool = None
        if spool is None:  # pragma: no cover - defensive
            await self._send_transcript("")
            self._reset_audio()
            return

        duration = self._pcm_bytes / BYTES_PER_SECOND
        started = time.perf_counter()
        try:
            spool.seek(0)
            wm = await runtime._offload(runtime.store.get)

            lang_started = time.perf_counter()
            try:
                resolved_language = await runtime._offload(
                    language_helpers._resolve_language, wm, spool, self._language
                )
            except HTTPException:
                # Wyoming has no error event; fall back to server defaults/auto.
                logger.warning(
                    "unsupported Wyoming language %r; falling back to server defaults",
                    self._language,
                )
                resolved_language = None
            runtime._record_stage("language", time.perf_counter() - lang_started)

            spool.seek(0)
            batched = inference._use_batched(duration)
            runner = inference._build_runner(wm, duration)
            kwargs = inference._build_kwargs(
                language=resolved_language,
                prompt=runtime.settings.initial_prompt,
                temperature=0.0,
                batched=batched,
                without_timestamps=True,
            )
            inf_started = time.perf_counter()
            _segments, info, text = await runtime._maybe_timeout(
                runtime._offload(
                    inference._transcribe_collect, runner, spool, kwargs, False
                ),
                runtime.settings.request_timeout_s,
            )
            runtime._record_stage("inference", time.perf_counter() - inf_started)
            await self._send_transcript(text, getattr(info, "language", None))
        except Exception:
            logger.exception("Wyoming transcription failed")
            self._service.errors += 1
            await self._send_transcript("")
        finally:
            try:
                spool.close()
            except Exception:  # pragma: no cover - defensive
                pass
            runtime._record_stage("wyoming", time.perf_counter() - started)
            self._reset_audio()

    async def _send_transcript(self, text: str, language: Optional[str] = None) -> None:
        try:
            await self.write_event(Transcript(text=text, language=language).event())
        except Exception:  # pragma: no cover - client disconnected during shutdown
            logger.debug("could not write Wyoming transcript", exc_info=True)


class _ZeroconfRegistration:
    """Register/unregister ``_wyoming._tcp`` for Home Assistant discovery."""

    def __init__(self, name: Optional[str], port: int, host: Optional[str]) -> None:
        self.name = name or _mac_name()
        self.port = port
        self.host = _advertised_host(host)
        self._zeroconf: Any = None

    async def start(self) -> None:
        from zeroconf.asyncio import AsyncServiceInfo, AsyncZeroconf

        self._zeroconf = AsyncZeroconf()
        info = AsyncServiceInfo(
            _WYOMING_SERVICE_TYPE,
            f"{self.name}.{_WYOMING_SERVICE_TYPE}",
            addresses=[socket.inet_aton(self.host)],
            port=self.port,
        )
        await self._zeroconf.async_register_service(info)
        logger.info("zeroconf: registered %s on %s:%d", self.name, self.host, self.port)

    async def stop(self) -> None:
        if self._zeroconf is None:
            return
        try:
            await self._zeroconf.async_unregister_all_services()
        finally:
            await self._zeroconf.async_close()
            self._zeroconf = None


def _advertised_host(bind_host: Optional[str]) -> str:
    """Pick the address to publish over mDNS (IPv4)."""
    if bind_host and bind_host not in ("0.0.0.0", "::"):
        return bind_host
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect(("224.0.0.251", 1))
        return sock.getsockname()[0]
    finally:
        sock.close()


def _mac_name() -> str:
    return "".join(
        f"{(uuid.getnode() >> shift) & 0xFF:02x}" for shift in range(40, -1, -8)
    )


class WyomingService:
    """Owns the Wyoming TCP listener and optional mDNS registration."""

    def __init__(self) -> None:
        self.requests = 0
        self.errors = 0
        self.active = 0
        self._server: Optional[AsyncTcpServer] = None
        self._zeroconf: Optional[_ZeroconfRegistration] = None

    async def start(self) -> None:
        settings = runtime.settings
        self._server = AsyncTcpServer(settings.wyoming_host, settings.wyoming_port)
        await self._server.start(partial(WyomingAsrHandler, self))
        logger.info(
            "Wyoming ASR listening on tcp://%s:%d",
            settings.wyoming_host,
            settings.wyoming_port,
        )

        if settings.wyoming_zeroconf:
            registration = _ZeroconfRegistration(
                settings.wyoming_name,
                settings.wyoming_port,
                settings.wyoming_host,
            )
            try:
                await registration.start()
            except Exception:
                # Discovery is optional; manual host/port still works.
                logger.exception("zeroconf registration failed; continuing without discovery")
            else:
                self._zeroconf = registration

    async def stop(self) -> None:
        if self._zeroconf is not None:
            try:
                await self._zeroconf.stop()
            except Exception:  # pragma: no cover - defensive
                logger.exception("zeroconf unregister failed")
            finally:
                self._zeroconf = None

        if self._server is not None:
            try:
                await self._server.stop()
            finally:
                self._server = None

    def stats(self) -> dict:
        settings = runtime.settings
        return {
            "enabled": settings.wyoming,
            "host": settings.wyoming_host,
            "port": settings.wyoming_port,
            "zeroconf": settings.wyoming_zeroconf,
            "name": settings.wyoming_name,
            "requests": self.requests,
            "errors": self.errors,
            "active": self.active,
        }
