"""
End-to-end tests for the robustness + concurrency features on the REAL model
(docs/PLAN-robustness-concurrency.md).

Boots real faster-whisper servers as subprocesses (no mocks) and verifies:

  - Phase 1: 413 on oversized uploads; /health stays responsive (event loop
             not blocked) during a long transcription; unexpected file suffixes
             still decode (PyAV sniffs bytes).
  - Phase 2: 400 on audio longer than WHISPER_MAX_AUDIO_SECONDS.
  - Phase 3: real OpenAI-style SSE streaming (stream=true).

Run:  uv run pytest tests/e2e_robustness_test.py -v
"""

import contextlib
import io
import json
import math
import os
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import wave
from pathlib import Path

import httpx
# pi-lens-ignore: reportMissingImports
import numpy as np
# pi-lens-ignore: reportMissingImports
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_URL = "https://github.com/openai/whisper/raw/main/tests/jfk.flac"
EXPECTED_PHRASE = "my fellow americans"
STARTUP_TIMEOUT_S = 180
SAMPLE_CACHE = Path(tempfile.gettempdir()) / "whisper_api_test_jfk.flac"


# ---------------------------------------------------------------------------
# Server booting (real app.py subprocess, WHISPER_* env) — shared shape
# ---------------------------------------------------------------------------

def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@contextlib.contextmanager
def _server(tmp_path_factory, extra_env: dict):
    port = _free_port()
    log_path = tmp_path_factory.mktemp("server") / "server.log"
    env = os.environ.copy()
    env.update(
        WHISPER_HOST="127.0.0.1",
        WHISPER_PORT=str(port),
        WHISPER_MODEL_NAME="base",
        WHISPER_COMPUTE_TYPE="int8",
        WHISPER_IDLE_UNLOAD_S="0",
        PYTHONUNBUFFERED="1",
        **extra_env,
    )
    with open(log_path, "wb") as log:
        proc = subprocess.Popen(
            [sys.executable, "app.py"], cwd=REPO_ROOT, env=env,
            stdout=log, stderr=subprocess.STDOUT,
        )
    base_url = f"http://127.0.0.1:{port}"
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
                r = httpx.get(f"{base_url}/health", timeout=5)
                if r.status_code == 200:
                    last_error = None
                    break
            except httpx.HTTPError as e:
                last_error = e
            time.sleep(1)
        else:
            raise TimeoutError(f"server not healthy after {STARTUP_TIMEOUT_S}s: {last_error}")
        yield base_url
    finally:
        if proc.poll() is None:
            proc.send_signal(signal.SIGTERM)
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=10)


# ---------------------------------------------------------------------------
# Fixtures: two servers + real speech sample
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def speech_sample():
    """jfk.flac (real English speech), cached under /tmp; tone fallback."""
    try:
        if not SAMPLE_CACHE.exists():
            SAMPLE_CACHE.write_bytes(
                httpx.get(SAMPLE_URL, timeout=60, follow_redirects=True).content
            )
        return SAMPLE_CACHE, True
    except httpx.HTTPError:
        tone = Path(tempfile.gettempdir()) / "whisper_api_test_tone.wav"
        with wave.open(str(tone), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(16000)
            frames = b"".join(
                b"\x00\x00" if int(i * 440 / 16000 * 2) % 2 else b"\x39\x05"
                for i in range(16000 * 2)
            )
            w.writeframes(frames)
        return tone, False


@pytest.fixture(scope="session")
def server_limits(tmp_path_factory):
    """WHISPER_MAX_UPLOAD_MB=1 and WHISPER_MAX_AUDIO_SECONDS=5."""
    with _server(
        tmp_path_factory,
        {"WHISPER_MAX_UPLOAD_MB": "1", "WHISPER_MAX_AUDIO_SECONDS": "5"},
    ) as url:
        yield url


@pytest.fixture(scope="session")
def server_responsive(tmp_path_factory):
    """Default robustness knobs (100 MB upload / 3600 s audio / timeout on)."""
    with _server(tmp_path_factory, {"WHISPER_MAX_CONCURRENT": "2"}) as url:
        yield url


@pytest.fixture(scope="session")
def client_limits(server_limits):
    with httpx.Client(base_url=server_limits, timeout=300.0) as c:
        yield c


@pytest.fixture(scope="session")
def client_responsive(server_responsive):
    with httpx.Client(base_url=server_responsive, timeout=300.0) as c:
        yield c


@pytest.fixture(scope="session")
def long_wav_bytes(speech_sample, tmp_path_factory):
    """jfk concatenated x8 (~88 s of speech) as a WAV, or None if no sample."""
    sample, is_speech = speech_sample
    if not is_speech:
        return None
    cached = Path(tempfile.gettempdir()) / "whisper_api_test_long.wav"
    if not cached.exists():
        from faster_whisper.audio import decode_audio

        samples = decode_audio(str(sample))  # float32 mono @16k
        big = np.tile(samples, 8)
        pcm = np.clip(big * 32767, -32767, 32767).astype(np.int16)
        with wave.open(str(cached), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(16000)
            w.writeframes(pcm.tobytes())
    return cached.read_bytes()


def _mk_wav(seconds: float, rate: int) -> bytes:
    n = int(seconds * rate)
    pcm = (np.sin(2 * math.pi * 440 * np.arange(n) / rate) * 12000).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm.tobytes())
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Server A (limits): 413 size cap + 400 duration guard, real model
# ---------------------------------------------------------------------------

def test_oversized_upload_returns_413(client_limits):
    # ~3s @ 192 kHz mono int16 = ~1.15 MB > 1 MB cap (fails on size, not duration).
    r = client_limits.post(
        "/v1/audio/transcriptions",
        files={"file": ("big.wav", _mk_wav(3.0, rate=192000), "audio/wav")},
    )
    assert r.status_code == 413, r.text
    assert "too large" in r.json()["detail"].lower()


def test_audio_over_duration_limit_returns_400(client_limits):
    # 10s @ 8 kHz mono = 160 KB (< 1 MB size cap) but > 5 s duration cap -> 400.
    r = client_limits.post(
        "/v1/audio/transcriptions",
        files={"file": ("long.wav", _mk_wav(10.0, rate=8000), "audio/wav")},
    )
    assert r.status_code == 400, r.text
    assert "too long" in r.json()["detail"].lower()


def test_under_limits_ok(client_limits):
    # 2s @ 16 kHz = 64 KB (< 1 MB) and < 5 s -> success path reached the model.
    r = client_limits.post(
        "/v1/audio/transcriptions",
        files={"file": ("ok.wav", _mk_wav(2.0, rate=16000), "audio/wav")},
    )
    assert r.status_code == 200, r.text
    assert isinstance(r.json()["text"], str)


# ---------------------------------------------------------------------------
# Server B (responsive): event loop stays free, suffix sniffing, streaming
# ---------------------------------------------------------------------------

def test_health_responsive_during_long_transcription(client_responsive, long_wav_bytes):
    """/health must answer quickly while a multi-window transcription runs."""
    if long_wav_bytes is None:
        pytest.skip("speech sample unavailable for a long clip")
    result = {}

    def _do_post():
        result["r"] = client_responsive.post(
            "/v1/audio/transcriptions",
            files={"file": ("long.wav", long_wav_bytes, "audio/wav")},
        )

    th = threading.Thread(target=_do_post)
    th.start()
    time.sleep(1.0)  # let the model load + transcription start
    t0 = time.monotonic()
    h = client_responsive.get("/health")
    latency = time.monotonic() - t0
    assert h.status_code == 200
    assert latency < 2.0, f"/health blocked for {latency:.2f}s during inference"
    th.join(timeout=240)
    assert result["r"].status_code == 200, result["r"].text


def test_unexpected_suffix_still_decodes(client_responsive, speech_sample):
    """flac bytes uploaded as .mp3 must still decode (PyAV sniffs the bytes)."""
    sample, is_speech = speech_sample
    with open(sample, "rb") as f:
        r = client_responsive.post(
            "/v1/audio/transcriptions",
            files={"file": ("jfk.mp3", f, "audio/mpeg")},
        )
    assert r.status_code == 200, r.text
    if is_speech:
        assert EXPECTED_PHRASE in r.json()["text"].lower()


def test_streaming_real_model(client_responsive, speech_sample):
    """stream=true returns an SSE transcript over real inference."""
    sample, is_speech = speech_sample
    with open(sample, "rb") as f:
        with client_responsive.stream(
            "POST", "/v1/audio/transcriptions",
            files={"file": (sample.name, f, "audio/flac")},
            data={"stream": "true"},
        ) as r:
            assert r.status_code == 200
            body = "".join(r.iter_text())
    assert r.headers.get("content-type", "").startswith("text/event-stream")
    assert "event: transcript\n" in body
    assert "[DONE]" in body
    if is_speech:
        done_text = ""
        for line in body.splitlines():
            if line.startswith("data: ") and '"text"' in line:
                done_text = json.loads(line[len("data: "):])["text"].lower()
                break
        assert EXPECTED_PHRASE in done_text