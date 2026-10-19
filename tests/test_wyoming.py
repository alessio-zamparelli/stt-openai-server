"""Unit/integration tests for the Wyoming ASR protocol server.

These start a real Wyoming TCP listener on a free loopback port and talk to it
with the official ``wyoming`` client. The model is a fake, so the suite is fast
and offline.
"""

from __future__ import annotations

import asyncio
import io
import math
import socket
import wave
from types import SimpleNamespace
from typing import List, Optional

import numpy as np

import stt_api.runtime as runtime
import stt_api.services.wyoming as wyoming
from stt_api.config import Settings
from stt_api.services.model_store import ModelStore
from wyoming.asr import Transcribe, Transcript
from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.client import AsyncTcpClient
from wyoming.event import Event
from wyoming.info import Describe, Info

WYOMING_TIMEOUT = 5.0


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _pcm(seconds: float, rate: int = 16000) -> bytes:
    n = int(seconds * rate)
    samples = (np.sin(2 * math.pi * 440 * np.arange(n) / rate) * 12000).astype(np.int16)
    return samples.tobytes()


class FakeModel:
    def __init__(self, text: str = " hello world") -> None:
        self.text = text
        self.supported_languages = {"en", "it", "de"}
        self.transcribe_kwargs: List[dict] = []
        self.params: Optional[tuple] = None

    def transcribe(self, audio, **kwargs):
        self.transcribe_kwargs.append(dict(kwargs))
        # Capture the WAV format before the handler closes the buffer.
        try:
            audio.seek(0)
            with wave.open(audio, "rb") as wav:
                self.params = (
                    wav.getframerate(),
                    wav.getsampwidth(),
                    wav.getnchannels(),
                )
        except Exception:
            self.params = None
        return iter([SimpleNamespace(text=self.text)]), SimpleInfo("en")


class SimpleInfo:
    def __init__(self, language: str) -> None:
        self.language = language


def _settings(port: int, **overrides) -> Settings:
    return Settings(
        idle_unload_s=0,
        max_concurrent=2,
        wyoming=True,
        wyoming_host="127.0.0.1",
        wyoming_port=port,
        **overrides,
    )


async def _run_with_service(settings: Settings, model: FakeModel, scenario) -> None:
    """Start a Wyoming service with the fake model, run ``scenario``, clean up."""
    original_store, original_settings = runtime.store, runtime.settings
    runtime.store = ModelStore(settings, loader=lambda: model)
    runtime.settings = settings
    runtime._reset_inference_pool()
    service = wyoming.WyomingService()
    await service.start()
    try:
        await scenario(service, model)
    finally:
        await service.stop()
        runtime.store, runtime.settings = original_store, original_settings
        runtime._reset_inference_pool()


def _run(coro) -> None:
    asyncio.run(coro)


async def _read(client) -> Optional[Event]:
    return await asyncio.wait_for(client.read_event(), timeout=WYOMING_TIMEOUT)


async def _send_audio(client, pcm: bytes, rate: int = 16000, chunk: int = 3200) -> None:
    await client.write_event(
        AudioStart(rate=rate, width=2, channels=1).event()
    )
    for offset in range(0, len(pcm), chunk):
        await client.write_event(
            AudioChunk(
                rate=rate,
                width=2,
                channels=1,
                audio=pcm[offset : offset + chunk],
            ).event()
        )
    await client.write_event(AudioStop().event())


# ---------------------------------------------------------------------------
# Info / discovery
# ---------------------------------------------------------------------------


def test_describe_returns_asr_info():
    port = _free_port()
    model = FakeModel()

    async def scenario(service, _model):
        async with AsyncTcpClient("127.0.0.1", port) as client:
            await client.write_event(Describe().event())
            event = await _read(client)
            assert event is not None
            info = Info.from_event(event)
            assert len(info.asr) == 1
            program = info.asr[0]
            assert program.name == "stt-api"
            assert program.requires_external_vad is True
            assert program.supports_transcript_streaming is False
            assert program.models[0].name == "base"
            assert program.models[0].languages  # non-empty for HA
            assert "en" in program.models[0].languages

    _run(_run_with_service(_settings(port), model, scenario))


def test_describe_does_not_load_the_model():
    port = _free_port()
    model = FakeModel()

    async def scenario(service, _model):
        assert runtime.store.peek() is None
        async with AsyncTcpClient("127.0.0.1", port) as client:
            await client.write_event(Describe().event())
            await _read(client)
        assert runtime.store.peek() is None
        assert runtime.store.reloads == 0

    _run(_run_with_service(_settings(port), model, scenario))


def test_unknown_events_are_ignored():
    port = _free_port()
    model = FakeModel()

    async def scenario(service, _model):
        async with AsyncTcpClient("127.0.0.1", port) as client:
            await client.write_event(
                Event(type="select-program", data={"name": "stt-api"})
            )
            await client.write_event(Event(type="future-event", data={}))
            await client.write_event(Describe().event())
            event = await _read(client)
            assert event is not None and event.type == "info"

    _run(_run_with_service(_settings(port), model, scenario))


# ---------------------------------------------------------------------------
# Transcription
# ---------------------------------------------------------------------------


def test_transcribe_roundtrip_and_close():
    port = _free_port()
    model = FakeModel(" hello world")

    async def scenario(service, _model):
        async with AsyncTcpClient("127.0.0.1", port) as client:
            await client.write_event(Transcribe(language="en").event())
            await _send_audio(client, _pcm(0.3))
            event = await _read(client)
            assert event is not None
            transcript = Transcript.from_event(event)
            assert transcript.text == "hello world"
            assert transcript.language == "en"
            # One utterance per connection: the server closes it.
            assert await _read(client) is None
        assert service.requests == 1
        assert service.errors == 0
        assert runtime.store.reloads == 1

    _run(_run_with_service(_settings(port), model, scenario))


def test_transcribe_ignores_model_name_and_normalizes_language():
    port = _free_port()
    model = FakeModel("ok")

    async def scenario(service, _model):
        async with AsyncTcpClient("127.0.0.1", port) as client:
            await client.write_event(
                Transcribe(name="not-the-server-model", language="it-IT").event()
            )
            await _send_audio(client, _pcm(0.2))
            event = await _read(client)
            assert event is not None
            assert Transcript.from_event(event).text == "ok"
        assert model.transcribe_kwargs[-1]["language"] == "it"

    _run(_run_with_service(_settings(port), model, scenario))


def test_unsupported_language_falls_back_instead_of_failing():
    port = _free_port()
    model = FakeModel("fallback")

    async def scenario(service, _model):
        async with AsyncTcpClient("127.0.0.1", port) as client:
            await client.write_event(Transcribe(language="klingon").event())
            await _send_audio(client, _pcm(0.2))
            event = await _read(client)
            assert event is not None
            assert Transcript.from_event(event).text == "fallback"
        assert "language" not in model.transcribe_kwargs[-1]
        assert service.errors == 0

    _run(_run_with_service(_settings(port), model, scenario))


def test_non_16k_audio_is_converted_to_16k_mono():
    port = _free_port()
    model = FakeModel("converted")

    async def scenario(service, _model):
        async with AsyncTcpClient("127.0.0.1", port) as client:
            await client.write_event(Transcribe(language="en").event())
            await _send_audio(client, _pcm(0.3, rate=8000), rate=8000)
            event = await _read(client)
            assert event is not None
            assert Transcript.from_event(event).text == "converted"
        assert model.params == (16000, 2, 1)

    _run(_run_with_service(_settings(port), model, scenario))


def test_empty_audio_returns_empty_transcript():
    port = _free_port()
    model = FakeModel()

    async def scenario(service, _model):
        async with AsyncTcpClient("127.0.0.1", port) as client:
            await client.write_event(Transcribe(language="en").event())
            await client.write_event(
                AudioStart(rate=16000, width=2, channels=1).event()
            )
            await client.write_event(AudioStop().event())
            event = await _read(client)
            assert event is not None
            assert Transcript.from_event(event).text == ""
        assert model.transcribe_kwargs == []

    _run(_run_with_service(_settings(port), model, scenario))


def test_audio_over_limit_returns_empty_transcript():
    port = _free_port()
    model = FakeModel()
    settings = _settings(port, max_audio_seconds=1)

    async def scenario(service, _model):
        async with AsyncTcpClient("127.0.0.1", port) as client:
            await client.write_event(Transcribe(language="en").event())
            await _send_audio(client, _pcm(2.0))
            event = await _read(client)
            assert event is not None
            assert Transcript.from_event(event).text == ""
        assert model.transcribe_kwargs == []
        assert service.errors == 1
        assert runtime.store.reloads == 0  # never touched the model

    _run(_run_with_service(settings, model, scenario))


def test_wyoming_requests_mark_store_activity():
    port = _free_port()
    model = FakeModel()

    async def scenario(service, _model):
        # Pretend the store has been idle for a while.
        runtime.store._last_activity -= 100.0
        assert runtime.store.last_request_age() > 50.0
        async with AsyncTcpClient("127.0.0.1", port) as client:
            await client.write_event(Transcribe(language="en").event())
            # A describe round-trip guarantees the transcribe event was handled.
            await client.write_event(Describe().event())
            assert await _read(client) is not None
        # Transcribe itself is demand even before audio arrives.
        assert runtime.store.last_request_age() < 50.0

    _run(_run_with_service(_settings(port), model, scenario))


# ---------------------------------------------------------------------------
# Zeroconf (no real mDNS traffic)
# ---------------------------------------------------------------------------


def test_zeroconf_registration_uses_wyoming_service_type(monkeypatch):
    captured = {}

    class FakeInfo:
        def __init__(self, service_type, name, addresses=None, port=None):
            captured["service_type"] = service_type
            captured["name"] = name
            captured["addresses"] = addresses
            captured["port"] = port

    class FakeZeroconf:
        def __init__(self):
            self.registered = None
            self.unregistered = False
            self.closed = False

        async def async_register_service(self, info):
            self.registered = info

        async def async_unregister_all_services(self):
            self.unregistered = True

        async def async_close(self):
            self.closed = True

    monkeypatch.setattr("zeroconf.asyncio.AsyncServiceInfo", FakeInfo)
    monkeypatch.setattr("zeroconf.asyncio.AsyncZeroconf", FakeZeroconf)

    registration = wyoming._ZeroconfRegistration("test-stt", 10300, "127.0.0.1")

    async def scenario():
        await registration.start()
        assert captured["service_type"] == "_wyoming._tcp.local."
        assert captured["name"] == "test-stt._wyoming._tcp.local."
        assert captured["port"] == 10300
        await registration.stop()
        assert registration._zeroconf is None

    asyncio.run(scenario())
