"""Unit tests for inference-tuning knobs (docs/PLAN-performance.md Phase 1).

Covers env parsing, _build_kwargs forwarding, the temperature schedule
override, the cpu_threads loader wiring, and the /health knob reporting.
"""
import io
import wave
from contextlib import contextmanager

import numpy as np
import pytest
from fastapi.testclient import TestClient

import app as app_module
from app import ModelStore, Settings, _build_kwargs, _parse_temperatures


def _make_wav(seconds: float = 0.5, rate: int = 16000) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes((np.zeros(int(rate * seconds), dtype=np.int16)).tobytes())
    return buf.getvalue()


class FakeModel:
    """Fake faster-whisper model: records kwargs, returns a trivially small
    result so the aggregation unit produces a usable response."""

    def __init__(self):
        self.transcribe_kwargs = []

    def transcribe(self, path, **kwargs):
        self.transcribe_kwargs.append(kwargs)
        segments = []
        info = type("Info", (), {"language": "en", "duration": 0.5})()
        return segments, info


@contextmanager
def _client(cfg: Settings, model: FakeModel):
    store = ModelStore(config=cfg, loader=lambda: model)
    orig_store, orig_cfg = app_module.store, app_module.settings
    app_module.store, app_module.settings = store, cfg
    try:
        with TestClient(app_module.app) as client:
            yield client, model
    finally:
        app_module.store, app_module.settings = orig_store, orig_cfg


# -- temperature schedule parsing -----------------------------------------


def test_parse_temperatures():
    assert _parse_temperatures("") is None
    assert _parse_temperatures("   ") is None
    assert _parse_temperatures("0,0.2,0.4") == [0.0, 0.2, 0.4]
    assert _parse_temperatures(" 0 , 0.2 ") == [0.0, 0.2]
    assert _parse_temperatures("0") == [0.0]
    assert _parse_temperatures("bad") is None  # fallback, no raise
    assert _parse_temperatures("0,oops,0.4") is None


# -- _build_kwargs forwarding ----------------------------------------------


def test_build_kwargs_forwards_beam_and_best_of(monkeypatch):
    monkeypatch.setattr(app_module, "settings", Settings(beam_size=2, best_of=2))
    kw = _build_kwargs(language=None, prompt=None, temperature=0.0)
    assert kw["beam_size"] == 2
    assert kw["best_of"] == 2


def test_build_kwargs_defaults_beam_and_best_of(monkeypatch):
    monkeypatch.setattr(app_module, "settings", Settings())
    kw = _build_kwargs(language=None, prompt=None, temperature=0.0)
    assert kw["beam_size"] == 5
    assert kw["best_of"] == 5


def test_build_kwargs_schedule_overrides_scalar(monkeypatch):
    monkeypatch.setattr(
        app_module, "settings", Settings(temperature_schedule="0,0.2,0.4")
    )
    kw = _build_kwargs(language=None, prompt=None, temperature=0.0)
    assert kw["temperature"] == [0.0, 0.2, 0.4]


def test_build_kwargs_bad_schedule_falls_back_to_scalar(monkeypatch):
    monkeypatch.setattr(app_module, "settings", Settings(temperature_schedule="oops"))
    kw = _build_kwargs(language=None, prompt=None, temperature=0.7)
    assert kw["temperature"] == 0.7


# -- loader wiring ----------------------------------------------------------


def test_load_model_passes_cpu_threads(monkeypatch):
    captured = {}

    class FakeWhisperModel:
        def __init__(self, *args, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr("faster_whisper.WhisperModel", FakeWhisperModel)
    monkeypatch.setattr(app_module, "settings", Settings(cpu_threads=2))
    app_module._load_model()
    assert captured["cpu_threads"] == 2


def test_load_model_passes_zero_cpu_threads(monkeypatch):
    captured = {}

    class FakeWhisperModel:
        def __init__(self, *args, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr("faster_whisper.WhisperModel", FakeWhisperModel)
    monkeypatch.setattr(app_module, "settings", Settings(cpu_threads=0))
    app_module._load_model()
    assert captured["cpu_threads"] == 0  # fw default


# -- /health reporting ------------------------------------------------------


def test_health_reports_tuning_knobs():
    cfg = Settings(beam_size=2, best_of=2, temperature_schedule="0,0.2,0.4", cpu_threads=2)
    with _client(cfg, FakeModel()) as (client, _):
        body = client.get("/health").json()
    assert body["beam_size"] == 2
    assert body["best_of"] == 2
    assert body["temperature_schedule"] == "0,0.2,0.4"
    assert body["cpu_threads"] == 2


def test_health_reports_tuning_defaults():
    with _client(Settings(), FakeModel()) as (client, _):
        body = client.get("/health").json()
    assert body["beam_size"] == 5
    assert body["best_of"] == 5
    assert body["temperature_schedule"] is None
    assert body["cpu_threads"] == 0


# -- route uses the knobs (fake model sees settings-driven transcribe kwargs)


def test_route_forwards_beam_and_schedule_to_model():
    cfg = Settings(beam_size=2, best_of=2, temperature_schedule="0,0.2,0.4")
    model = FakeModel()
    with _client(cfg, model) as (client, _):
        resp = client.post(
            "/v1/audio/transcriptions",
            files={"file": ("a.wav", _make_wav(), "audio/wav")},
            data={"model": "whisper-1"},
        )
        assert resp.status_code == 200
        resp.json()
    kw = model.transcribe_kwargs[-1]
    assert kw["beam_size"] == 2
    assert kw["best_of"] == 2
    assert kw["temperature"] == [0.0, 0.2, 0.4]
