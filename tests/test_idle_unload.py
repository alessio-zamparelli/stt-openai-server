"""
Unit tests for the idle-eviction feature (docs/PLAN-idle-unload.md).

Uses a real ``ModelStore`` around a fake model with a fake *loader* so the
store can rebuild itself after an eviction. All timers are explicit (no
sleeps): ``maybe_unload(now=...)`` takes an injected clock and tests reach into
``_last_activity`` to place the idle window deterministically.

Run:  uv run pytest tests/test_idle_unload.py -v
"""

from __future__ import annotations

import threading
from contextlib import contextmanager
from typing import Any, Iterator, List

import pytest
from fastapi.testclient import TestClient

import app as app_module
from app import ModelStore, Settings

DEFAULT_IDLE = Settings().idle_unload_s  # 300


class FakeModel:
    """Pseudo-faster-whisper model: returns canned segments from transcribe()."""

    def transcribe(self, audio, **kwargs):
        segs = [
            SimpleSegment("hello world", 0.0, 1.0),
            SimpleSegment("this is a test", 1.0, 2.0),
        ]
        return iter(segs), SimpleInfo("en")


class SimpleSegment:
    def __init__(self, text, start, end):
        self.id = 1
        self.seek = 0
        self.start = start
        self.end = end
        self.text = text
        self.tokens = [1]
        self.temperature = 0.0
        self.avg_logprob = 0.0
        self.compression_ratio = 1.0
        self.no_speech_prob = 0.0


class SimpleInfo:
    def __init__(self, language):
        self.language = language


def _make_store(
    config: Settings | None = None, *, loader_calls: List[int] | None = None
) -> ModelStore:
    """Production-shaped store: a fake model + a reload loader that records calls.

    ``loader_calls`` is mutated on every reload so tests can assert
    single-flight / reload counts.
    """
    cfg = config or Settings(idle_unload_s=DEFAULT_IDLE)
    calls = loader_calls if loader_calls is not None else []

    def loader() -> Any:
        calls.append(len(calls))
        return FakeModel()

    return ModelStore(cfg, loader=loader)


@contextmanager
def _make_client(store: ModelStore) -> Iterator[TestClient]:
    """TestClient with the module-global store swapped for the given fake.

    The routes and lifespan resolve ``app_module.store`` at call time, so
    swapping the module attribute suffices. The store uses a fast fake loader,
    so no model is downloaded. Swapping is undone after the client context
    exits. Note the lifespan calls ``store.get()`` eagerly, so the fake's
    ``reloads`` starts at 1 after the client is entered.
    """
    original = app_module.store
    app_module.store = store
    try:
        with TestClient(app_module.app) as client:
            yield client
    finally:
        app_module.store = original


# -- config -------------------------------------------------------------


def test_idle_config_defaults():
    cfg = Settings()
    assert cfg.idle_unload_s == 300  # 5 min
    assert cfg.idle_poll_s == 30


def test_idle_config_zero_disables():
    assert Settings(idle_unload_s=0).idle_unload_s == 0
    assert not ModelStore(Settings(idle_unload_s=0)).eviction_enabled


# -- engine lifecycle ---------------------------------------------------


def test_store_evicts_when_idle():
    calls: List[int] = []
    store = _make_store(loader_calls=calls)
    assert store.get() is not None
    assert calls == [0]  # initial load recorded via the loader
    store._last_activity -= 10_000
    assert store.maybe_unload(now=100_000) is True
    assert store._model is None
    assert store.loaded is False
    assert calls == [0]  # unload alone never calls the loader
    assert store.unloads == 1
    assert store.reloads == 1


def test_no_unload_when_recently_active():
    store = _make_store()
    store.get()
    store.touch()
    assert store.maybe_unload(now=store._last_activity + 1) is False
    assert store._model is not None


def test_no_unload_when_lock_held():
    """A transcription holds _lock; eviction must skip that pass."""
    store = _make_store()
    store.get()
    store._last_activity -= 10_000
    with store._lock:  # simulate an in-flight transcription
        assert store.maybe_unload(now=100_000) is False
    assert store._model is not None


def test_reload_restores_model_after_eviction():
    calls: List[int] = []
    store = _make_store(loader_calls=calls)
    old = store.get()
    store._last_activity -= 10_000
    assert store.maybe_unload(now=100_000) is True

    fresh = store.get()  # block-until-reloaded path
    assert fresh is not None
    assert fresh is not old
    assert store.loaded is True
    assert calls == [0, 1]  # initial load + one reload
    assert store.reloads == 2


def test_single_flight_reload():
    """Many concurrent waiters after an eviction share ONE rebuild."""
    import time

    calls: List[int] = []
    store = _make_store(loader_calls=calls)

    def slow_loader():
        calls.append(len(calls))
        time.sleep(0.05)
        return FakeModel()

    store._loader = slow_loader
    store._model = None  # simulate a prior eviction

    failures: List[Exception] = []
    lock = threading.Lock()

    def worker():
        try:
            store.get()
        except Exception as exc:  # pragma: no cover - failure path
            with lock:
                failures.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert failures == []
    assert len(calls) == 1  # single-flight: exactly one load
    assert store.reloads == 1


# -- health observability ------------------------------------------------


def test_health_reports_idle_fields():
    store = _make_store()
    with _make_client(store) as client:
        body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["loaded"] is True
    assert body["idle_unload_s"] == DEFAULT_IDLE
    assert body["unloads"] == 0
    assert body["reloads"] == 1  # eager startup load through the loader
    assert body["last_request_age_s"] >= 0


def test_health_reflects_eviction():
    store = _make_store()
    with _make_client(store) as client:
        store._last_activity -= 10_000
        assert store.maybe_unload(now=100_000) is True
        body = client.get("/health").json()
    assert body["status"] == "ok"  # still serving; next request wakes it
    assert body["loaded"] is False
    assert body["unloads"] == 1


def test_health_does_not_reset_idle_timer():
    """Health probes must NOT touch the idle window (would defeat eviction)."""
    store = _make_store()
    with _make_client(store) as client:
        before = store._last_activity
        for _ in range(3):
            client.get("/health")
        assert store._last_activity == before


# -- route-level wake path ------------------------------------------------


def test_transcribe_after_eviction_serves_200():
    """POST /v1/audio/transcriptions wakes an evicted store (block until
    reloaded) and returns 200 — not 503."""
    store = _make_store()
    with _make_client(store) as client:
        audio = b"\x00" * 4000  # FakeModel ignores content
        r1 = client.post(
            "/v1/audio/transcriptions",
            files={"file": ("a.wav", audio, "audio/wav")},
            data={"model": "whisper-1"},
        )
        assert r1.status_code == 200
        assert "hello world" in r1.json()["text"]

        store._last_activity -= 10_000
        assert store.maybe_unload(now=100_000) is True
        assert store.loaded is False

        r2 = client.post(
            "/v1/audio/transcriptions",
            files={"file": ("a.wav", audio, "audio/wav")},
            data={"model": "whisper-1"},
        )
        assert r2.status_code == 200
        assert "hello world" in r2.json()["text"]
        assert store.loaded is True
        assert store.reloads == 2  # initial load + wake reload
