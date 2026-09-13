"""Phase-3 (PLAN-performance.md) unit tests: VAD + offline knobs, rolling
stage timings, and /health latency reporting.

Reuses tests/test_tuning.py's FakeModel / _client harness.
"""
import os
from types import SimpleNamespace

import pytest

import app as app_module
from app import ModelStore, Settings, _build_kwargs

from tests.test_tuning import FakeModel, _client, _make_wav


# -- WHISPER_VAD wiring ----------------------------------------------------


def test_build_kwargs_vad_filter_when_enabled(monkeypatch):
    monkeypatch.setattr(app_module, "settings", Settings(vad_filter=True))
    kw = _build_kwargs(language=None, prompt=None, temperature=0.0)
    assert kw["vad_filter"] is True


def test_build_kwargs_no_vad_filter_by_default(monkeypatch):
    monkeypatch.setattr(app_module, "settings", Settings())
    kw = _build_kwargs(language=None, prompt=None, temperature=0.0)
    assert "vad_filter" not in kw


def test_env_vad_parse(monkeypatch):
    monkeypatch.setenv("WHISPER_VAD", "true")
    assert app_module._env_bool("WHISPER_VAD", False) is True
    monkeypatch.setenv("WHISPER_VAD", "off")
    assert app_module._env_bool("WHISPER_VAD", False) is False


def test_env_hf_offline_parse(monkeypatch):
    monkeypatch.setenv("WHISPER_HF_OFFLINE", "1")
    assert app_module._env_bool("WHISPER_HF_OFFLINE", False) is True


# -- WHISPER_HF_OFFLINE -> local_files_only wiring -------------------------


def test_load_model_passes_local_files_only(monkeypatch):
    captured = {}
    class _FakeWhisperModel:
        def __init__(self, *args, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(app_module, "settings", Settings(hf_offline=True))
    monkeypatch.setattr("faster_whisper.WhisperModel", _FakeWhisperModel)
    app_module._load_model()
    assert captured["local_files_only"] is True


def test_load_model_offline_default_off(monkeypatch):
    captured = {}
    class _FakeWhisperModel:
        def __init__(self, *args, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(app_module, "settings", Settings())
    monkeypatch.setattr("faster_whisper.WhisperModel", _FakeWhisperModel)
    app_module._load_model()
    assert captured["local_files_only"] is False


# -- rolling stage timings -------------------------------------------------


def test_record_and_stats_rolling_window():
    app_module._stage_samples.clear()
    try:
        for i in range(1, 400):
            app_module._record_stage("inference", i / 1000.0)
        st = app_module._stage_stats()
        assert st["inference"]["n"] == app_module._STAGE_WINDOW  # 200 cap
        # window keeps ms values 200..399 -> median ~300, p95 ~389
        assert 295.0 <= st["inference"]["p50_ms"] <= 305.0
        assert st["inference"]["p95_ms"] > st["inference"]["p50_ms"]
        assert st["inference"]["p95_ms"] <= 400.0
    finally:
        app_module._stage_samples.clear()


def test_stage_stats_no_samples():
    app_module._stage_samples.clear()
    try:
        assert app_module._stage_stats() == {}
    finally:
        app_module._stage_samples.clear()


def test_percentile_edges():
    assert app_module._percentile_sorted_ms([], 0.5) == 0.0
    assert app_module._percentile_sorted_ms([42.0], 0.95) == 42.0
    assert app_module._percentile_sorted_ms([1.0, 2.0], 0.0) == 1.0
    assert app_module._percentile_sorted_ms([1.0, 2.0], 1.0) == 2.0


def test_route_records_all_stages(monkeypatch):
    """A real request records upload/duration/language/inference/total."""
    app_module._stage_samples.clear()
    cfg = Settings()
    model = FakeModel()
    try:
        with _client(cfg, model) as (client, _):
            resp = client.post(
                "/v1/audio/transcriptions",
                files={"file": ("a.wav", _make_wav(), "audio/wav")},
                data={"model": "whisper-1"},
            )
            assert resp.status_code == 200
        st = app_module._stage_stats()
        for stage in ("upload", "duration", "language", "inference", "total"):
            assert st[stage]["n"] >= 1, stage
            assert st[stage]["p50_ms"] >= 0.0
    finally:
        app_module._stage_samples.clear()


def test_stream_path_records_setup_stages(monkeypatch):
    """Stream=true: upload/duration/total recorded; inference not (it has not
    completed at response-construction time)."""
    app_module._stage_samples.clear()
    cfg = Settings()
    model = FakeModel()
    try:
        with _client(cfg, model) as (client, _):
            with client.stream(
                "POST",
                "/v1/audio/transcriptions",
                files={"file": ("a.wav", _make_wav(), "audio/wav")},
                data={"model": "whisper-1", "stream": "true"},
            ) as resp:
                assert resp.status_code == 200
                body = "".join(resp.iter_text())
            assert "[DONE]" in body
        st = app_module._stage_stats()
        assert "upload" in st and "total" in st
        assert "inference" not in st  # SSE worker finishes after the response
    finally:
        app_module._stage_samples.clear()


# -- /health reporting -----------------------------------------------------


def test_health_reports_vad_hf_offline():
    with _client(Settings(vad_filter=True, hf_offline=True), FakeModel()) as (client, _):
        body = client.get("/health").json()
        assert body["vad_filter"] is True
        assert body["hf_offline"] is True
    with _client(Settings(), FakeModel()) as (client, _):
        body = client.get("/health").json()
        assert body["vad_filter"] is False
        assert body["hf_offline"] is False
        assert body["latency_ms"] == {}


def test_health_latency_key_present_after_request():
    app_module._stage_samples.clear()
    cfg = Settings()
    model = FakeModel()
    try:
        with _client(cfg, model) as (client, _):
            client.post(
                "/v1/audio/transcriptions",
                files={"file": ("a.wav", _make_wav(), "audio/wav")},
                data={"model": "whisper-1"},
            )
            body = client.get("/health").json()
            assert "total" in body["latency_ms"]
            assert body["latency_ms"]["total"]["n"] >= 1
    finally:
        app_module._stage_samples.clear()
