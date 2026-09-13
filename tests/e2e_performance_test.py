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

import os
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


# ---------------------------------------------------------------------------
# Phase 2 — opt-in batched inference (WHISPER_BATCH_SIZE)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def server_batched(tmp_path_factory):
    """WHISPER_BATCH_SIZE=4 — BatchedInferencePipeline opt-in mode."""
    with _server(tmp_path_factory, {"WHISPER_BATCH_SIZE": "4"}) as url:
        yield url


@pytest.fixture(scope="session")
def client_batched(server_batched):
    with httpx.Client(base_url=server_batched, timeout=600.0) as c:
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
    assert body["batch_size"] == 0


def test_batched_health_reports_batch_size(client_batched):
    body = client_batched.get("/health").json()
    assert body["batch_size"] == 4
    assert body["beam_size"] == 5  # tuning defaults unchanged, batched is independent


def test_batched_transcribes_long_clip(client_batched, long_wav_bytes):
    """Deterministic: batched server returns a non-empty transcript for the
    88 s clip (correctness is the always-on gate; timing is env-gated below)."""
    if long_wav_bytes is None:
        pytest.skip("no 88s speech clip available")
    files = {"file": ("long.wav", long_wav_bytes, "audio/wav")}
    r = client_batched.post(
        "/v1/audio/transcriptions", files=files, data={"model": "whisper-1"}
    )
    assert r.status_code == 200
    assert r.json()["text"].strip()


def test_batched_faster_than_sequential_walltime(
    client_responsive, client_batched, long_wav_bytes
):
    """PLAN Phase-2 gate: batched < 0.75x sequential wall-time on the 88 s clip.

    Opt-in (PERF_WALLTIME=1) because shared-host load noise can reverse the
    one-shot ratio (observed 0.56x..1.44x on this machine); run it on quiet /
    dedicated hardware. Best-of-2, interleaved, same bytes."""
    if os.environ.get("PERF_WALLTIME") != "1":
        pytest.skip("PERF_WALLTIME=1 (quiet/dedicated hardware) to run timing gate")
    if long_wav_bytes is None:
        pytest.skip("no 88s speech clip available")
    files = {"file": ("long.wav", long_wav_bytes, "audio/wav")}

    def transcribe(client):
        t0 = time.perf_counter()
        r = client.post(
            "/v1/audio/transcriptions", files=files, data={"model": "whisper-1"}
        )
        dt = time.perf_counter() - t0
        assert r.status_code == 200 and r.json()["text"].strip()
        return dt

    seq_times, bat_times = [], []
    for _ in range(2):
        seq_times.append(transcribe(client_responsive))
        bat_times.append(transcribe(client_batched))
    seq, bat = min(seq_times), min(bat_times)
    print(f"\n[walltime] seq={seq_times} bat={bat_times} best ratio={bat / seq:.3f}")
    assert bat < seq * 0.75, (
        f"batched {bat:.2f}s not < 0.75x sequential {seq:.2f}s ({seq_times} vs {bat_times})"
    )


def test_batched_stream_still_progressive(client_batched, speech_sample):
    """SSE on the batched server still emits transcript deltas + [DONE]."""
    sample, is_speech = speech_sample
    if not is_speech:
        pytest.skip("no real speech sample")
    with open(sample, "rb") as f:
        with client_batched.stream(
            "POST",
            "/v1/audio/transcriptions",
            files={"file": ("jfk.flac", f, "audio/flac")},
            data={"model": "whisper-1", "stream": "true"},
        ) as resp:
            assert resp.status_code == 200
            body = "".join(resp.iter_text())
    assert "event: transcript" in body or "event: transcript.done" in body
    assert "[DONE]" in body


def test_temperature_trim_preserves_phrase_on_tone(client_tuned):
    """Even a synthetic tone with the trimmed schedule returns 200 (no crash)."""
    r = client_tuned.post(
        "/v1/audio/transcriptions",
        files={"file": ("t.wav", _mk_wav(1.0, 16000), "audio/wav")},
        data={"model": "whisper-1", "language": "en"},
    )
    assert r.status_code == 200
