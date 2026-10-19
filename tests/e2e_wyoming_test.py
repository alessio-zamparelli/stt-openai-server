"""End-to-end test for the Wyoming ASR listener (real model, real inference).

Boots the real server with ``STT_WYOMING=true`` on random HTTP/Wyoming ports,
streams OpenAI's jfk.flac as 16 kHz PCM chunks, and asserts the transcript.

Run:  uv run python -m pytest tests/e2e_wyoming_test.py -v
"""

import asyncio
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx
import numpy as np
import pytest

from wyoming.asr import Transcribe, Transcript
from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.client import AsyncTcpClient
from wyoming.info import Describe, Info

REPO_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_URL = "https://github.com/openai/whisper/raw/main/tests/jfk.flac"
EXPECTED_PHRASE = "my fellow americans"
STARTUP_TIMEOUT_S = 180


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.fixture(scope="session")
def server(tmp_path_factory):
    http_port = _free_port()
    wyoming_port = _free_port()
    log_path = tmp_path_factory.mktemp("wyoming-server") / "server.log"
    env = os.environ.copy()
    env.update(
        STT_HOST="127.0.0.1",
        STT_PORT=str(http_port),
        STT_MODEL_NAME="base",
        STT_COMPUTE_TYPE="int8",
        STT_IDLE_UNLOAD_S="0",
        STT_WYOMING="true",
        STT_WYOMING_HOST="127.0.0.1",
        STT_WYOMING_PORT=str(wyoming_port),
        PYTHONUNBUFFERED="1",
    )
    with open(log_path, "wb") as log:
        proc = subprocess.Popen(
            [sys.executable, "app.py"],
            cwd=REPO_ROOT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    base_url = f"http://127.0.0.1:{http_port}"
    deadline = time.monotonic() + STARTUP_TIMEOUT_S
    last_error = None
    try:
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                raise RuntimeError(
                    f"server exited early (code {proc.returncode}):\n"
                    f"{log_path.read_text(errors='replace')[-2000:]}"
                )
            try:
                response = httpx.get(f"{base_url}/health", timeout=5)
                if response.status_code == 200:
                    last_error = None
                    break
            except httpx.HTTPError as exc:
                last_error = exc
            time.sleep(1)
        else:
            raise TimeoutError(
                f"server not healthy after {STARTUP_TIMEOUT_S}s: {last_error}"
            )
        yield base_url, wyoming_port
    finally:
        if proc.poll() is None:
            proc.send_signal(signal.SIGTERM)
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=10)


@pytest.fixture(scope="session")
def speech_pcm(tmp_path_factory):
    """jfk.flac decoded to 16 kHz mono s16 PCM (or a tone fallback)."""
    from faster_whisper.audio import decode_audio

    data_dir = tmp_path_factory.mktemp("wyoming-audio")
    sample = data_dir / "jfk.flac"
    real = True
    try:
        sample.write_bytes(
            httpx.get(SAMPLE_URL, timeout=60, follow_redirects=True).content
        )
    except httpx.HTTPError:
        real = False
        sample = None

    if sample is not None:
        audio = decode_audio(str(sample))
    else:
        n = 16000 * 2
        audio = (np.sin(2 * np.pi * 440 * np.arange(n) / 16000) * 8000).astype(
            np.float32
        )
    pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16).tobytes()
    return pcm, real


def _run_wyoming(port: int, pcm: bytes) -> tuple[str, str]:
    async def scenario():
        async with AsyncTcpClient("127.0.0.1", port) as client:
            await client.write_event(Describe().event())
            info_event = await asyncio.wait_for(client.read_event(), timeout=10)
            assert info_event is not None
            info = Info.from_event(info_event)
            assert info.asr and info.asr[0].name == "stt-api"
            program_name = info.asr[0].name

        async with AsyncTcpClient("127.0.0.1", port) as client:
            await client.write_event(Transcribe(language="en").event())
            await client.write_event(
                AudioStart(rate=16000, width=2, channels=1).event()
            )
            for offset in range(0, len(pcm), 3200):
                await client.write_event(
                    AudioChunk(
                        rate=16000,
                        width=2,
                        channels=1,
                        audio=pcm[offset : offset + 3200],
                    ).event()
                )
            await client.write_event(AudioStop().event())
            event = await asyncio.wait_for(client.read_event(), timeout=120)
            assert event is not None
            transcript = Transcript.from_event(event)
            return program_name, transcript.text

    return asyncio.run(scenario())


def test_wyoming_roundtrip(server, speech_pcm):
    base_url, wyoming_port = server
    pcm, real = speech_pcm

    program_name, text = _run_wyoming(wyoming_port, pcm)
    assert program_name == "stt-api"

    if real:
        assert EXPECTED_PHRASE in text.lower()
    else:
        pytest.skip("speech sample download failed; plumbing-only run")

    health = httpx.get(f"{base_url}/health", timeout=10).json()
    assert health["wyoming"]["enabled"] is True
    assert health["wyoming"]["port"] == wyoming_port
    assert health["wyoming"]["requests"] >= 1
    assert health["wyoming"]["errors"] == 0
    assert "wyoming" in health["latency_ms"]
