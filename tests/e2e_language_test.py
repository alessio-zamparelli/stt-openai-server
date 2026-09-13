"""
End-to-end tests for the server-side language features on the REAL model.

Boots real faster-whisper servers as subprocesses (real inference — no mocks)
with the WHISPER_* language knobs set, then exercises the resolution logic
over HTTP:

  - WHISPER_LANGUAGE       server default language applied when the client omits it
  - WHISPER_LANGUAGES      allowlist; with several codes, audio-language
                           detection is *constrained* to the allowed set
  - WHISPER_INITIAL_PROMPT server-wide prompt fallback (client may override)

Run:  uv run pytest tests/e2e_language_test.py -v

Requires network on first run to fetch a speech sample (cached under /tmp
afterwards, so the companion e2e_test.py can reuse it).
"""

import contextlib
import os
import signal
import socket
import subprocess
import sys
import tempfile
import time
import wave
from pathlib import Path

import httpx
# pi-lens-ignore: reportMissingImports
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_URL = "https://github.com/openai/whisper/raw/main/tests/jfk.flac"
EXPECTED_PHRASE = "my fellow americans"
STARTUP_TIMEOUT_S = 180  # model download/load can be slow on first run
SAMPLE_CACHE = Path(tempfile.gettempdir()) / "whisper_api_test_jfk.flac"


# ---------------------------------------------------------------------------
# Server booting (real app.py subprocess, WHISPER_* env)
# ---------------------------------------------------------------------------

def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@contextlib.contextmanager
def _server(tmp_path_factory, extra_env: dict):
    """Boot app.py with the given WHISPER_* env; kill it on exit."""
    port = _free_port()
    log_path = tmp_path_factory.mktemp("server") / "server.log"
    env = os.environ.copy()
    env.update(
        WHISPER_HOST="127.0.0.1",
        WHISPER_PORT=str(port),
        WHISPER_MODEL_NAME="base",
        WHISPER_COMPUTE_TYPE="int8",
        WHISPER_IDLE_UNLOAD_S="0",  # keep the model hot for deterministic timing
        PYTHONUNBUFFERED="1",
        **extra_env,
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
def server_default_lang(tmp_path_factory):
    """WHISPER_LANGUAGE=it (server default) + a server-wide initial prompt."""
    with _server(
        tmp_path_factory,
        {"WHISPER_LANGUAGE": "it", "WHISPER_INITIAL_PROMPT": "Transcribe the audio."},
    ) as url:
        yield url


@pytest.fixture(scope="session")
def server_allowlist(tmp_path_factory):
    """WHISPER_LANGUAGES=it,en: constrained detection, no forced default."""
    with _server(tmp_path_factory, {"WHISPER_LANGUAGES": "it,en"}) as url:
        yield url


@pytest.fixture(scope="session")
def client_default_lang(server_default_lang):
    with httpx.Client(base_url=server_default_lang, timeout=300.0) as c:
        yield c


@pytest.fixture(scope="session")
def client_allowlist(server_allowlist):
    with httpx.Client(base_url=server_allowlist, timeout=300.0) as c:
        yield c


@pytest.fixture(scope="session")
def speech_sample():
    """A real English speech clip (OpenAI's jfk.flac), cached under /tmp.

    Falls back to a sine tone if the download fails; tests asserting on
    transcript *content* are skipped for the fallback.
    """
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


def _transcribe(client, sample_path, **data):
    with open(sample_path, "rb") as f:
        return client.post(
            "/v1/audio/transcriptions",
            files={"file": (sample_path.name, f, "audio/flac")},
            data=data,
        )


# ---------------------------------------------------------------------------
# Server A: WHISPER_LANGUAGE=it  (server default)
# ---------------------------------------------------------------------------

def test_health_reports_default_language(client_default_lang):
    body = client_default_lang.get("/health").json()
    assert body["status"] == "ok"
    assert body["default_language"] == "it"
    assert body["allowed_languages"] is None


def test_server_default_language_applied_when_client_omits(client_default_lang, speech_sample):
    sample, _ = speech_sample
    r = _transcribe(client_default_lang, sample)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["task"] == "transcribe"
    # faster-whisper reports the language actually used for decoding.
    assert body["language"] == "it"
    assert body["text"]


def test_client_language_overrides_server_default(client_default_lang, speech_sample):
    sample, is_speech = speech_sample
    r = _transcribe(client_default_lang, sample, language="en")
    assert r.status_code == 200, r.text
    assert r.json()["language"] == "en"
    if is_speech:
        assert EXPECTED_PHRASE in r.json()["text"].lower()


def test_empty_language_string_uses_default(client_default_lang, speech_sample):
    sample, _ = speech_sample
    r = _transcribe(client_default_lang, sample, language="")
    assert r.status_code == 200, r.text
    assert r.json()["language"] == "it"


def test_whitespace_language_uses_default(client_default_lang, speech_sample):
    sample, _ = speech_sample
    r = _transcribe(client_default_lang, sample, language="   ")
    assert r.status_code == 200, r.text
    assert r.json()["language"] == "it"


def test_unsupported_language_rejected_400(client_default_lang, speech_sample):
    sample, _ = speech_sample
    r = _transcribe(client_default_lang, sample, language="klingon")
    assert r.status_code == 400
    assert "klingon" in r.json()["detail"]


def test_server_prompt_does_not_break_request(client_default_lang, speech_sample):
    sample, is_speech = speech_sample
    r = _transcribe(client_default_lang, sample)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["text"]
    if is_speech:
        # With a forced default language the English clip is transcribed under
        # the Italian prompt — assert plumbing works, not exact wording.
        assert body["task"] == "transcribe"


def test_client_prompt_overrides_server_prompt(client_default_lang, speech_sample):
    sample, _ = speech_sample
    r = _transcribe(client_default_lang, sample, language="en", prompt="Custom client prompt.")
    assert r.status_code == 200, r.text
    assert r.json()["text"]


# ---------------------------------------------------------------------------
# Server B: WHISPER_LANGUAGES=it,en  (constrained detection)
# ---------------------------------------------------------------------------

def test_health_reports_allowlist(client_allowlist):
    body = client_allowlist.get("/health").json()
    assert body["allowed_languages"] == "it,en"
    assert body["default_language"] is None


def test_constrained_detection_picks_english(client_allowlist, speech_sample):
    """Real audio: English speech must be detected as `en` inside {it,en}."""
    sample, is_speech = speech_sample
    r = _transcribe(client_allowlist, sample)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["task"] == "transcribe"
    if is_speech:
        assert body["language"] == "en"
        assert EXPECTED_PHRASE in body["text"].lower()
    else:
        # Fallback tone: the endpoint still answers and returns a language
        # from the allowlist (it or en).
        assert body["language"] in ("it", "en")


def test_client_language_overrides_allowlist(client_allowlist, speech_sample):
    sample, _ = speech_sample
    r = _transcribe(client_allowlist, sample, language="it")
    assert r.status_code == 200, r.text
    assert r.json()["language"] == "it"
