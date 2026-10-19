"""Configuration tests: canonical STT_* env vars and legacy WHISPER_* aliases."""

import logging

from stt_api import config


def _clear(monkeypatch, *names):
    for name in names:
        monkeypatch.delenv(name, raising=False)


def test_stt_env_is_canonical(monkeypatch):
    _clear(monkeypatch, "WHISPER_PORT", "STT_PORT")
    monkeypatch.setenv("STT_PORT", "9123")
    assert config.load_settings().port == 9123


def test_legacy_whisper_env_alias(monkeypatch, caplog):
    _clear(monkeypatch, "STT_PORT", "WHISPER_PORT")
    monkeypatch.setenv("WHISPER_PORT", "9124")
    config._warned_legacy.clear()
    with caplog.at_level(logging.WARNING):
        assert config.load_settings().port == 9124
    assert any("WHISPER_PORT" in record.message for record in caplog.records)


def test_stt_wins_over_legacy_alias(monkeypatch):
    _clear(monkeypatch, "STT_PORT", "WHISPER_PORT")
    monkeypatch.setenv("WHISPER_PORT", "9124")
    monkeypatch.setenv("STT_PORT", "9125")
    assert config.load_settings().port == 9125


def test_legacy_alias_covers_bool_int_float_and_string(monkeypatch):
    _clear(
        monkeypatch,
        "STT_VAD", "WHISPER_VAD",
        "STT_TEMPERATURES", "WHISPER_TEMPERATURES",
        "STT_REQUEST_TIMEOUT_S", "WHISPER_REQUEST_TIMEOUT_S",
        "STT_MODEL_NAME", "WHISPER_MODEL_NAME",
    )
    monkeypatch.setenv("WHISPER_VAD", "true")
    monkeypatch.setenv("WHISPER_TEMPERATURES", "0,0.2,0.4")
    monkeypatch.setenv("WHISPER_REQUEST_TIMEOUT_S", "12.5")
    monkeypatch.setenv("WHISPER_MODEL_NAME", "small")
    cfg = config.load_settings()
    assert cfg.vad_filter is True
    assert cfg.temperature_schedule == "0,0.2,0.4"
    assert cfg.request_timeout_s == 12.5
    assert cfg.model_name == "small"


def test_legacy_warning_is_emitted_once(monkeypatch, caplog):
    _clear(monkeypatch, "STT_PORT", "WHISPER_PORT")
    monkeypatch.setenv("WHISPER_PORT", "9126")
    config._warned_legacy.clear()
    with caplog.at_level(logging.WARNING):
        config.load_settings()
        config.load_settings()
    messages = [r.message for r in caplog.records if "WHISPER_PORT" in r.message]
    assert len(messages) == 1
