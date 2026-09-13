"""
Unit/integration tests for the robustness + concurrency features
(docs/PLAN-robustness-concurrency.md):

  - Phase 1: event-loop offload (bounded threadpool), streamed uploads with a
             413 size cap (+ Content-Length fast path), suffix preservation.
  - Phase 2: duration guard (400) and abandon-the-call inference timeout (504).
  - Phase 3: OpenAI-style SSE streaming (stream=true).

Run:  uv run pytest tests/test_robustness.py -v
"""

from __future__ import annotations

import io
import math
import queue
import threading
import time
import wave
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any, Iterator, List, Optional, Tuple

import numpy as np
import pytest
from fastapi.testclient import TestClient

import app as app_module
from app import ModelStore, Settings, _check_content_length, _safe_suffix


# --------------------------------------------------------------------------
# Fakes + fixtures
# --------------------------------------------------------------------------

class FakeModel:
    """Returns scripted segments/info; optionally lags (for timeout tests)."""

    def __init__(
        self,
        segments: Optional[List[Any]] = None,
        lag: float = 0.0,
    ):
        self.segments = segments or []
        self.lag = lag
        self.transcribe_kwargs: List[dict] = []
        self.supported_languages = {"en", "it", "de", "es", "fr", "pt"}

    def transcribe(self, audio, **kwargs):
        self.transcribe_kwargs.append(dict(kwargs))
        if self.lag:
            import time as _t
            _t.sleep(self.lag)
        return iter(self.segments), Info("en")

    def detect_language(self, audio) -> Tuple[str, float, List[Tuple[str, float]]]:
        return "en", 0.9, [("en", 0.9), ("it", 0.1)]


class Info:
    def __init__(self, language):
        self.language = language


def _seg(i: int, text: str, start: float = 0.0, end: float = 1.0) -> SimpleNamespace:
    return SimpleNamespace(
        id=i, seek=0, start=start, end=end, text=text, tokens=[1, 2, 3],
        temperature=0.0, avg_logprob=-0.1, compression_ratio=1.0, no_speech_prob=0.0,
    )


def _make_store(model) -> ModelStore:
    return ModelStore(Settings(idle_unload_s=0), loader=lambda: model)


@contextmanager
def _make_client(store: ModelStore, cfg: Settings):
    orig_store, orig_cfg = app_module.store, app_module.settings
    app_module.store, app_module.settings = store, cfg
    try:
        with TestClient(app_module.app) as client:
            yield client
    finally:
        app_module.store, app_module.settings = orig_store, orig_cfg


def _wav(seconds: float, rate: int = 16000, channels: int = 1, bits: int = 16) -> bytes:
    """(Possibly large) mono/stereo PCM WAV so decode / size limits apply."""
    n = int(seconds * rate)
    pcm = (np.sin(2 * math.pi * 440 * np.arange(n) / rate) * 12000).astype(
        np.int16 if bits == 16 else np.int32
    )
    if channels == 2:
        pcm = np.repeat(pcm.reshape(-1, 1), 2, axis=1)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(bits // 8)
        w.setframerate(rate)
        w.writeframes(pcm.tobytes())
    return buf.getvalue()


# --------------------------------------------------------------------------
# Phase 1 — suffix preservation + upload size cap (unit-level helpers)
# --------------------------------------------------------------------------

def test_safe_suffix_mapping():
    assert _safe_suffix("k.MP3") == ".mp3"
    assert _safe_suffix("k.wav") == ".wav"
    assert _safe_suffix("a.flac") == ".flac"
    assert _safe_suffix(None) == ".bin"
    assert _safe_suffix("noext") == ".bin"
    assert _safe_suffix("a.txt") == ".bin"  # not in the allowed codec set
    assert _safe_suffix("a.tar.gz") == ".bin"  # last suffix not allowed


def test_check_content_length_fast_path_413():
    with pytest.raises(Exception) as exc:
        _check_content_length("99999999", 100)
    assert getattr(exc.value, "status_code", None) == 413
    # no header / no limit -> no-op
    _check_content_length(None, 100)
    _check_content_length("500", None)
    _check_content_length("50", 100)


def test_route_mp3_suffix_accepts_upload():
    # The route forwards a safe .mp3 suffix to the temp file; with a fake
    # model there's no decode, but it proves the suffix path doesn't crash.
    # (Real byte-sniffing decode is covered by the live-server e2e.)
    model = FakeModel(segments=[_seg(0, "hello")])
    cfg = Settings(idle_unload_s=0)
    with _make_client(_make_store(model), cfg) as c:
        r = c.post(
            "/v1/audio/transcriptions",
            files={"file": ("garbage.mp3", _wav(0.3), "audio/mpeg")},
        )
    assert r.status_code == 200
    assert r.json()["text"] == "hello"


# --------------------------------------------------------------------------
# Phase 1 — 413 upload cap (mid-stream + content-length)
# --------------------------------------------------------------------------

def test_upload_exceeds_cap_returns_413():
    model = FakeModel()
    cfg = Settings(idle_unload_s=0, max_upload_mb=1)
    with _make_client(_make_store(model), cfg) as c:
        r = c.post(
            "/v1/audio/transcriptions",
            # ~3s at 192 kHz mono int16 = ~1.15 MB > 1 MB cap.
            files={"file": ("big.wav", _wav(3.0, rate=192000), "audio/wav")},
        )
    assert r.status_code == 413
    assert "too large" in r.json()["detail"].lower()
    assert model.transcribe_kwargs == []  # never reached the model


def test_upload_within_cap_ok_and_reaches_model():
    model = FakeModel(segments=[_seg(0, "ok")])
    cfg = Settings(idle_unload_s=0, max_upload_mb=100)
    with _make_client(_make_store(model), cfg) as c:
        r = c.post(
            "/v1/audio/transcriptions",
            files={"file": ("t.wav", _wav(0.3), "audio/wav")},
        )
    assert r.status_code == 200
    assert r.json()["text"] == "ok"
    assert model.transcribe_kwargs  # inference ran


def test_unlimited_upload_accepted():
    model = FakeModel(segments=[_seg(0, "ok")])
    cfg = Settings(idle_unload_s=0, max_upload_mb=0)  # 0 = unlimited
    with _make_client(_make_store(model), cfg) as c:
        r = c.post(
            "/v1/audio/transcriptions",
            files={"file": ("t.wav", _wav(0.3), "audio/wav")},
        )
    assert r.status_code == 200


# --------------------------------------------------------------------------
# Phase 2 — duration guard (400)
# --------------------------------------------------------------------------

def test_audio_too_long_returns_400():
    model = FakeModel()
    cfg = Settings(idle_unload_s=0, max_audio_seconds=2)
    with _make_client(_make_store(model), cfg) as c:
        r = c.post(
            "/v1/audio/transcriptions",
            files={"file": ("long.wav", _wav(5.0, rate=8000), "audio/wav")},
        )
    assert r.status_code == 400
    assert "too long" in r.json()["detail"].lower()
    assert model.transcribe_kwargs == []  # never reached the model


def test_audio_under_duration_limit_ok():
    model = FakeModel(segments=[_seg(0, "ok")])
    cfg = Settings(idle_unload_s=0, max_audio_seconds=3600)
    with _make_client(_make_store(model), cfg) as c:
        r = c.post(
            "/v1/audio/transcriptions",
            files={"file": ("t.wav", _wav(0.3), "audio/wav")},
        )
    assert r.status_code == 200
    assert r.json()["text"] == "ok"


def test_duration_guard_disabled_when_zero():
    model = FakeModel(segments=[_seg(0, "ok")])
    cfg = Settings(idle_unload_s=0, max_audio_seconds=0)  # 0 = off
    with _make_client(_make_store(model), cfg) as c:
        r = c.post(
            "/v1/audio/transcriptions",
            files={"file": ("long.wav", _wav(5.0, rate=8000), "audio/wav")},
        )
    assert r.status_code == 200


# --------------------------------------------------------------------------
# Phase 2 — inference timeout (504)
# --------------------------------------------------------------------------

def test_inference_timeout_returns_504_and_stays_healthy():
    model = FakeModel(lag=1.0)  # worker sleeps past the tiny timeout
    cfg = Settings(idle_unload_s=0, request_timeout_s=0.05)
    with _make_client(_make_store(model), cfg) as c:
        r = c.post(
            "/v1/audio/transcriptions",
            files={"file": ("t.wav", _wav(0.3), "audio/wav")},
        )
        assert r.status_code == 504
        assert "timed out" in r.json()["detail"].lower()
        # The event loop was never blocked, so the server stays responsive.
        assert c.get("/health").status_code == 200
        assert c.get("/health").json()["status"] == "ok"


def test_timeout_disabled_when_zero_completes():
    model = FakeModel(segments=[_seg(0, "done")], lag=0.01)
    cfg = Settings(idle_unload_s=0, request_timeout_s=0)  # 0 = off
    with _make_client(_make_store(model), cfg) as c:
        r = c.post(
            "/v1/audio/transcriptions",
            files={"file": ("t.wav", _wav(0.3), "audio/wav")},
        )
    assert r.status_code == 200
    assert r.json()["text"] == "done"


# --------------------------------------------------------------------------
# Phase 3 — OpenAI-style streaming (SSE)
# --------------------------------------------------------------------------

def _stream_body(
    c: TestClient, extra_data: Optional[dict] = None
) -> Tuple[int, str, str]:
    """POST with stream=true; returns (status, body, content_type)."""
    data = {"model": "whisper-1", "stream": "true"}
    if extra_data:
        data.update(extra_data)
    with c.stream(
        "POST", "/v1/audio/transcriptions",
        files={"file": ("t.wav", _wav(0.3), "audio/wav")},
        data=data,
    ) as r:
        return r.status_code, "".join(r.iter_text()), r.headers.get("content-type", "")


def test_streaming_emits_deltas_and_done():
    model = FakeModel(segments=[_seg(0, "Hello "), _seg(1, "world.")])
    cfg = Settings(idle_unload_s=0)
    with _make_client(_make_store(model), cfg) as c:
        status, body, ct = _stream_body(c)
    assert status == 200
    assert ct.startswith("text/event-stream")
    assert "event: transcript\n" in body

    # Every per-segment delta plus the final done frame.
    deltas = []
    for line in body.splitlines():
        if line.startswith("data: ") and '"delta"' in line:
            import json
            deltas.append(json.loads(line[len("data: "):])["delta"])
    assert "".join(deltas).strip() == "Hello world."
    assert "event: transcript.done\n" in body
    assert "event: done\ndata: [DONE]" in body


def test_streaming_with_no_segments_still_completes():
    model = FakeModel(segments=[])
    cfg = Settings(idle_unload_s=0)
    with _make_client(_make_store(model), cfg) as c:
        status, body, ct = _stream_body(c)
    assert status == 200
    assert ct.startswith("text/event-stream")
    assert "event: transcript.done\n" in body
    assert "[DONE]" in body


def test_nonstream_still_returns_json():
    model = FakeModel(segments=[_seg(0, "Hello "), _seg(1, "world.")])
    cfg = Settings(idle_unload_s=0)
    with _make_client(_make_store(model), cfg) as c:
        r = c.post(
            "/v1/audio/transcriptions",
            files={"file": ("t.wav", _wav(0.3), "audio/wav")},
            data={"model": "whisper-1"},
        )
    assert r.status_code == 200
    j = r.json()
    assert j["task"] == "transcribe"
    assert j["language"] == "en"
    assert j["text"] == "Hello world."


# --------------------------------------------------------------------------
# /health exposes the robustness knobs
# --------------------------------------------------------------------------

def test_health_reports_robustness_fields():
    model = FakeModel()
    cfg = Settings(
        idle_unload_s=0,
        max_concurrent=3,
        max_upload_mb=7,
        max_audio_seconds=90,
        request_timeout_s=12.5,
    )
    with _make_client(_make_store(model), cfg) as c:
        body = c.get("/health").json()
    assert body["max_concurrent"] == 3
    assert body["max_upload_mb"] == 7
    assert body["max_audio_seconds"] == 90
    assert body["request_timeout_s"] == 12.5

# --------------------------------------------------------------------------
# Phase 0 (PLAN-performance.md) — thread-safe inference-pool init
# --------------------------------------------------------------------------

def test_inference_pool_concurrent_init_returns_same_executor():
    """16 threads hammering the getter must all see ONE executor (race guard)."""
    orig = app_module.settings
    app_module.settings = Settings(idle_unload_s=0, max_concurrent=2)
    app_module._reset_inference_pool()
    try:
        results = []

        def worker():
            results.append(app_module._inference_pool())

        threads = [threading.Thread(target=worker) for _ in range(16)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert all(r is results[0] for r in results)
        assert results[0] is not None
        assert results[0]._max_workers == 2
    finally:
        app_module.settings = orig
        app_module._reset_inference_pool()


def test_inference_pool_rebuilt_after_settings_swap():
    orig = app_module.settings
    app_module.settings = Settings(idle_unload_s=0, max_concurrent=4)
    app_module._reset_inference_pool()
    try:
        pool = app_module._inference_pool()
        assert pool is not None
        assert pool._max_workers == 4
    finally:
        app_module.settings = orig
        app_module._reset_inference_pool()


def test_inference_pool_unbounded_returns_none():
    orig = app_module.settings
    app_module.settings = Settings(idle_unload_s=0, max_concurrent=0)
    app_module._reset_inference_pool()
    try:
        assert app_module._inference_pool() is None
    finally:
        app_module.settings = orig
        app_module._reset_inference_pool()


def test_lifespan_inits_and_resets_pool():
    model = FakeModel()
    cfg = Settings(idle_unload_s=0, max_concurrent=2)
    app_module._reset_inference_pool()
    with _make_client(_make_store(model), cfg):
        assert app_module._inference_executor is not None
        assert app_module._inference_executor._max_workers == 2
    assert app_module._inference_executor is None  # shutdown resets


# --------------------------------------------------------------------------
# Serialize-lock on the single shared model (concurrent transcribe() race)
# --------------------------------------------------------------------------

class _SerializeProbe:
    """Fake runner tracking peak concurrent inference calls, including the
    lazy decode during segment iteration.

    One enter/exit pair spans ``transcribe()`` AND the full iteration (the
    exit fires in the generator's ``finally`` when it is exhausted). With the
    serialize lock held across call + iteration two threads never overlap →
    ``max_active`` stays 1; a lock covering only the call (or none) would hit
    2 here because the 50 ms sleep forces the decode windows to intersect.
    """

    def __init__(self) -> None:
        self._active = 0
        self.max_active = 0
        self._mu = threading.Lock()

    def _enter(self) -> None:
        with self._mu:
            self._active += 1
            self.max_active = max(self.max_active, self._active)

    def _exit(self) -> None:
        with self._mu:
            self._active -= 1

    def transcribe(self, audio, **kwargs):
        self._enter()

        def gen():
            try:
                time.sleep(0.05)  # overlap window if the lock is missing
                yield _seg(0, "ok")
            finally:
                self._exit()  # call ends when the lazy decode is exhausted

        return gen(), Info("en")


def _run_concurrent(fn):
    """Run ``fn()`` in two threads released simultaneously; join both."""
    barrier = threading.Barrier(3)

    def run():
        barrier.wait()
        fn()

    threads = [threading.Thread(target=run) for _ in range(2)]
    for t in threads:
        t.start()
    barrier.wait()
    for t in threads:
        t.join()


def test_inference_serialized_across_concurrent_collects():
    """Two threads transcribing the shared model never overlap: the serialize
    lock covers the lazy decode, so max_active stays at 1 (never interleaved)."""
    probe = _SerializeProbe()
    results: List[Any] = []
    _run_concurrent(
        lambda: results.append(
            app_module._transcribe_collect(probe, "/tmp/probe.wav", {})
        )
    )
    assert len(results) == 2
    assert probe.max_active == 1  # no two callers overlapped at any point
    assert all(len(r[0]) == 1 and r[0][0]["text"] == "ok" for r in results)


def test_stream_worker_serialized_across_concurrent_calls():
    """The SSE producer also holds the serialize lock: two concurrent
    producers never overlap and both still emit done+end."""
    probe = _SerializeProbe()
    q: "queue.SimpleQueue" = queue.SimpleQueue()
    _run_concurrent(lambda: app_module._stream_worker(probe, "/tmp/probe.wav", {}, q))
    kinds = []
    while not q.empty():
        kinds.append(q.get())
    assert probe.max_active == 1
    assert kinds.count(("end", None)) == 2
    assert sum(1 for k in kinds if k[0] == "done") == 2
