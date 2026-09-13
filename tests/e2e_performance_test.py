"""End-to-end tests for the inference-tuning + performance work on the REAL
model (docs/PLAN-performance.md).

Phase 1 (this file's core today):
  - Tuned server (WHISPER_BEAM_SIZE=2, WHISPER_BEST_OF=2,
    WHISPER_TEMPERATURES=0,0.2,0.4) still transcribes jfk with the expected
    phrase and no repetition loop; /health reports the effective knobs.

Phase 2 adds: batched server < 0.75x sequential wall-time on the 88s clip;
             SSE still streams progressively on the batched server.
Phase 3 adds: offline (WHISPER_HF_OFFLINE + HF_HUB_OFFLINE) reload e2e and
             per-stage latency_ms at /health.

Run:  uv run pytest tests/e2e_performance_test.py -v
"""

import time

import httpx
# pi-lens-ignore: reportMissingImports
import pytest

from tests.e2e_robustness_test import _server, _mk_wav
from tests.e2e_robustness_test import server_responsive  # defaults server
from tests.e2e_robustness_test import client_responsive  # defaults server
from tests.e2e_robustness_test import long_wav_bytes  # noqa: F401 (Phase 2)
from tests.e2e_robustness_test import speech_sample  # noqa: F401 (Phase 3)

EXPECTED_PHRASE = "my fellow americans"


def _norm(s: str) -> str:
    return "".join(c for c in s.lower() if c.isalnum())


# ---------------------------------------------------------------------------
# Phase 1 — tuned inference knobs, real model
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def server_tuned(tmp_path_factory):
    """beam=2/best=2 + trimmed temperature schedule."""
    with _server(
        tmp_path_factory,
        {
            "WHISPER_BEAM_SIZE": "2",
            "WHISPER_BEST_OF": "2",
            "WHISPER_TEMPERATURES": "0,0.2,0.4",
        },
    ) as url:
        yield url


@pytest.fixture(scope="session")
def client_tuned(server_tuned):
    with httpx.Client(base_url=server_tuned, timeout=300.0) as c:
        yield c


def test_tuned_server_transcribes_jfk_without_loops(client_tuned, speech_sample):
    sample, is_speech = speech_sample
    with open(sample, "rb") as f:
        r = client_tuned.post(
            "/v1/audio/transcriptions",
            files={"file": ("jfk.flac", f, "audio/flac")},
            data={"model": "whisper-1"},
        )
    assert r.status_code == 200
    text = r.json()["text"]
    assert _norm(EXPECTED_PHRASE) in _norm(text)
    # No repetition loop: the full transcript must not be longer than ~4x the
    # source (120 chars-ish for the 11 s clip) and must not repeat a 4-gram 3x.
    assert len(text) < 400, f"suspicious length (possible loop): {text!r}"
    words = _norm(text).split()
    grams = [" ".join(words[i : i + 4]) for i in range(len(words) - 3)]
    assert not any(grams.count(g) >= 3 for g in set(grams)), f"loop: {text!r}"


def test_tuned_server_health_reports_knobs(client_tuned):
    body = client_tuned.get("/health").json()
    assert body["beam_size"] == 2
    assert body["best_of"] == 2
    assert body["temperature_schedule"] == "0,0.2,0.4"
    assert body["cpu_threads"] == 0


def test_default_health_keeps_accuracy_first_defaults(client_responsive):
    """No-env server must still report beam=5/best=5 (gate kept defaults)."""
    body = client_responsive.get("/health").json()
    assert body["beam_size"] == 5
    assert body["best_of"] == 5
    assert body["temperature_schedule"] is None
    assert body["cpu_threads"] == 0


def test_temperature_trim_preserves_phrase_on_tone(client_tuned):
    """Even a synthetic tone with the trimmed schedule returns 200 (no crash)."""
    r = client_tuned.post(
        "/v1/audio/transcriptions",
        files={"file": ("t.wav", _mk_wav(1.0, 16000), "audio/wav")},
        data={"model": "whisper-1", "language": "en"},
    )
    assert r.status_code == 200
