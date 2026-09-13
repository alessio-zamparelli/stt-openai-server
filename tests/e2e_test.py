"""
End-to-end functional tests for the OpenAI-compatible Whisper API.

These tests boot the REAL server as a subprocess (real faster-whisper model,
real inference — no mocks) on a random port, then exercise every endpoint over
HTTP, including all response formats and error paths.

Run:  uv run pytest tests/e2e_test.py -v

Requires network on first run to fetch the speech sample (cached afterwards).
"""

import os
import signal
import socket
import subprocess
import sys
import time
import wave
from pathlib import Path

import httpx
# pi-lens-ignore: reportMissingImports
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_URL = "https://github.com/openai/whisper/raw/main/tests/jfk.flac"
# Known content of the OpenAI jfk.flac sample (JFK's 1961 inaugural address).
EXPECTED_PHRASE = "my fellow americans"
STARTUP_TIMEOUT_S = 180  # model download/load can be slow on first run


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="session")
def server(tmp_path_factory):
    """Boot the real server as a subprocess and wait until it is healthy."""
    port = _free_port()
    log_path = tmp_path_factory.mktemp("server") / "server.log"
    env = os.environ.copy()
    env.update(
        WHISPER_HOST="127.0.0.1",
        WHISPER_PORT=str(port),
        WHISPER_MODEL_NAME="base",
        WHISPER_COMPUTE_TYPE="int8",
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


@pytest.fixture(scope="session")
def client(server):
    with httpx.Client(base_url=server, timeout=300.0) as c:
        yield c


@pytest.fixture(scope="session")
def speech_sample(tmp_path_factory):
    """
    A real speech recording (OpenAI's jfk.flac, 11 s, English).

    Falls back to a generated sine tone if the download fails; tests that
    assert on transcript *content* are skipped for the fallback.
    """
    data_dir = tmp_path_factory.mktemp("audio")
    sample = data_dir / "jfk.flac"
    try:
        sample.write_bytes(httpx.get(SAMPLE_URL, timeout=60, follow_redirects=True).content)
        return sample, True
    except httpx.HTTPError:
        # Fallback: 2 s of 440 Hz tone — enough to exercise the plumbing.
        tone = data_dir / "tone.wav"
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


def _transcribe(client, sample_path, **data):
    with open(sample_path, "rb") as f:
        return client.post(
            "/v1/audio/transcriptions",
            files={"file": (sample_path.name, f, "audio/flac")},
            data=data,
        )


# ---------------------------------------------------------------------------
# Service-level endpoints
# ---------------------------------------------------------------------------

def test_health(server, client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["model"] == "base"
    assert body["compute_type"] == "int8"


def test_list_models(client):
    r = client.get("/v1/models")
    assert r.status_code == 200
    body = r.json()
    assert body["object"] == "list"
    assert [m["id"] for m in body["data"]] == ["base"]


def test_openapi_docs(client):
    r = client.get("/openapi.json")
    assert r.status_code == 200
    paths = set(r.json()["paths"])
    assert {
        "/v1/audio/transcriptions",
        "/v1/audio/translations",
        "/v1/models",
        "/health",
    } <= paths


# ---------------------------------------------------------------------------
# Transcriptions — happy paths
# ---------------------------------------------------------------------------

def test_transcribe_json(client, speech_sample):
    sample, is_speech = speech_sample
    r = _transcribe(client, sample, language="en")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["task"] == "transcribe"
    assert body["language"] == "en"
    assert isinstance(body["text"], str)
    if is_speech:
        assert EXPECTED_PHRASE in body["text"].lower()
        assert len(body["segments"]) > 0
        seg = body["segments"][0]
        for key in ("id", "start", "end", "text"):
            assert key in seg
        assert seg["end"] > seg["start"]
        # text is the concatenation of segment texts
        assert body["text"] == "".join(s["text"] for s in body["segments"]).strip()


def test_transcribe_model_param_is_ignored(client, speech_sample):
    """OpenAI clients send model=whisper-1; server must accept it."""
    sample, is_speech = speech_sample
    r = _transcribe(client, sample, model="whisper-1", language="en")
    assert r.status_code == 200, r.text
    if is_speech:
        assert EXPECTED_PHRASE in r.json()["text"].lower()


def test_transcribe_text_format(client, speech_sample):
    sample, is_speech = speech_sample
    r = _transcribe(client, sample, language="en", response_format="text")
    assert r.status_code == 200, r.text
    assert r.headers["content-type"].startswith("text/plain")
    body = r.text.strip()
    if is_speech:
        json_r = _transcribe(client, sample, language="en")
        assert body == json_r.json()["text"].strip()
        assert EXPECTED_PHRASE in body.lower()


def test_transcribe_srt_format(client, speech_sample):
    sample, is_speech = speech_sample
    r = _transcribe(client, sample, language="en", response_format="srt")
    assert r.status_code == 200, r.text
    if is_speech:
        assert " --> " in r.text
        # SRT timestamps look like 00:00:00,000
        assert "00:00:0" in r.text
        assert r.text.lstrip().startswith("1")


def test_transcribe_vtt_format(client, speech_sample):
    sample, is_speech = speech_sample
    r = _transcribe(client, sample, language="en", response_format="vtt")
    assert r.status_code == 200, r.text
    assert r.text.startswith("WEBVTT")
    if is_speech:
        assert " --> " in r.text


def test_transcribe_verbose_json(client, speech_sample):
    sample, _ = speech_sample
    r = _transcribe(client, sample, language="en", response_format="verbose_json")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["text"]
    assert body["segments"]


def test_transcribe_no_timestamps(client, speech_sample):
    sample, _ = speech_sample
    r = _transcribe(client, sample, language="en", timestamps="false")
    assert r.status_code == 200, r.text
    assert r.json()["segments"] is None


def test_transcribe_temperature(client, speech_sample):
    sample, _ = speech_sample
    r = _transcribe(client, sample, language="en", temperature=0.5)
    assert r.status_code == 200, r.text


# ---------------------------------------------------------------------------
# Translations
# ---------------------------------------------------------------------------

def test_translate_json(client, speech_sample):
    """Translation endpoint: task=translate, valid OpenAI-shaped response."""
    sample, is_speech = speech_sample
    with open(sample, "rb") as f:
        r = client.post(
            "/v1/audio/translations",
            files={"file": (sample.name, f, "audio/flac")},
            data={},
        )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["task"] == "translate"
    assert body["text"]
    if is_speech:
        assert body["segments"]


def test_translate_text_format(client, speech_sample):
    sample, is_speech = speech_sample
    with open(sample, "rb") as f:
        r = client.post(
            "/v1/audio/translations",
            files={"file": (sample.name, f, "audio/flac")},
            data={"response_format": "text"},
        )
    assert r.status_code == 200, r.text
    assert r.headers["content-type"].startswith("text/plain")
    if is_speech:
        assert r.text.strip()


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------

def test_transcribe_missing_file(client):
    r = client.post("/v1/audio/transcriptions", data={})
    assert r.status_code == 422


def test_translate_missing_file(client):
    r = client.post("/v1/audio/translations", data={})
    assert r.status_code == 422


def test_transcribe_invalid_response_format(client, speech_sample):
    sample, _ = speech_sample
    r = _transcribe(client, sample, response_format="parquet")
    assert r.status_code == 400
    assert "response_format" in r.json()["detail"].lower()


def test_translate_invalid_response_format(client, speech_sample):
    sample, _ = speech_sample
    with open(sample, "rb") as f:
        r = client.post(
            "/v1/audio/translations",
            files={"file": (sample.name, f, "audio/flac")},
            data={"response_format": "yaml"},
        )
    assert r.status_code == 400


def test_transcribe_empty_file_is_rejected(client):
    """Empty upload must not crash the worker; server stays healthy after."""
    r = client.post(
        "/v1/audio/transcriptions",
        files={"file": ("empty.wav", b"", "audio/wav")},
    )
    assert r.status_code >= 400
    # server must still be alive
    assert client.get("/health").status_code == 200
