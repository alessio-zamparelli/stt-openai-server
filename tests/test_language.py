"""
Unit tests for server-side language handling:

  - WHISPER_LANGUAGE         default language (used when the request omits it)
  - WHISPER_LANGUAGES        allowlist; a single code is forced, several codes
                             enable *constrained* detection (pick the best
                             probability among allowed languages only)
  - WHISPER_INITIAL_PROMPT   prompt applied when the client sends none

The resolution helper (_resolve_language) is tested directly with a fake
WhisperModel; the routes are exercised through TestClient with the store
swapped for a fast fake loader (no model downloads).

Run:  uv run pytest tests/test_language.py -v
"""

from __future__ import annotations

import io
import math
import wave
from contextlib import contextmanager
from typing import Any, Iterator, List, Optional, Tuple

import numpy as np
import pytest
from fastapi.testclient import TestClient

import app as app_module
from app import ModelStore, Settings, _resolve_language, _split_allowed_languages


# --------------------------------------------------------------------------
# Fakes + fixtures
# --------------------------------------------------------------------------

SUPPORTED = {"en", "it", "de", "es", "fr", "pt"}


class FakeModel:
    """Fake WhisperModel: records transcribe kwargs, scripts detect_language."""

    def __init__(self, detect_scores: Optional[List[Tuple[str, float]]] = None):
        """detect_scores: ranked list of (code, prob) returned by detection."""
        self.transcribe_kwargs: List[dict] = []
        self.detect_called = 0
        self.detect_scores = detect_scores or [("en", 0.6), ("de", 0.3), ("it", 0.1)]
        self.supported_languages = SUPPORTED

    def transcribe(self, audio, **kwargs):
        self.transcribe_kwargs.append(dict(kwargs))
        return iter([]), SimpleInfo(kwargs.get("language", "en"))

    def detect_language(self, audio) -> Tuple[str, float, List[Tuple[str, float]]]:
        self.detect_called += 1
        code, prob = self.detect_scores[0]
        return code, prob, self.detect_scores


class SimpleInfo:
    def __init__(self, language):
        self.language = language


def _make_store(model: FakeModel) -> ModelStore:
    return ModelStore(Settings(idle_unload_s=0), loader=lambda: model)


@contextmanager
def _make_client(store: ModelStore, cfg: Settings):
    """TestClient with module-global store AND settings swapped for the fakes."""
    orig_store, orig_cfg = app_module.store, app_module.settings
    app_module.store, app_module.settings = store, cfg
    try:
        with TestClient(app_module.app) as client:
            yield client
    finally:
        app_module.store, app_module.settings = orig_store, orig_cfg


def _sine_wav(seconds: float = 0.3, rate: int = 16000) -> bytes:
    """A tiny real mono 16k WAV so decode_audio() works in unit tests."""
    n = int(seconds * rate)
    pcm = (np.sin(2 * math.pi * 440 * np.arange(n) / rate) * 16000).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm.tobytes())
    return buf.getvalue()


@pytest.fixture()
def wav_path(tmp_path) -> str:
    p = tmp_path / "tiny.wav"
    p.write_bytes(_sine_wav())
    return str(p)


# --------------------------------------------------------------------------
# Resolution logic (unit-level, fake model)
# --------------------------------------------------------------------------

def test_requested_language_wins_over_default(monkeypatch):
    monkeypatch.setattr(
        app_module, "settings", Settings(default_language="it", allowed_languages="it,en")
    )
    assert _resolve_language(FakeModel(), "x.wav", "en") == "en"
    assert _resolve_language(FakeModel(), "x.wav", "DE") == "de"  # case-insensitive


def test_default_language_used_when_request_omits_it(monkeypatch):
    monkeypatch.setattr(app_module, "settings", Settings(default_language="it"))
    assert _resolve_language(FakeModel(), "x.wav", None) == "it"


def test_single_allowed_language_is_forced(monkeypatch):
    monkeypatch.setattr(app_module, "settings", Settings(allowed_languages=" it "))
    assert _resolve_language(FakeModel(), "x.wav", None) == "it"
    # even though detector would say en, the single-code allowlist wins
    model = FakeModel(detect_scores=[("en", 0.9)])
    assert _resolve_language(model, "x.wav", None) == "it"
    assert model.detect_called == 0  # no detection needed for one code


def test_multi_allowlist_constrains_detection(wav_path):
    # Detector ranks de highest, but 'de' is not allowed -> it wins.
    model = FakeModel(
        detect_scores=[("de", 0.9), ("it", 0.5), ("en", 0.4), ("fr", 0.05)]
    )
    app_module.settings = Settings(allowed_languages="it,en")
    try:
        assert _resolve_language(model, wav_path, None) == "it"
    finally:
        app_module.settings = Settings()
    assert model.detect_called == 1


def test_multi_allowlist_no_match_returns_none(wav_path):
    model = FakeModel(detect_scores=[("de", 0.9), ("fr", 0.1)])
    app_module.settings = Settings(allowed_languages="it,en")
    try:
        assert _resolve_language(model, wav_path, None) is None
    finally:
        app_module.settings = Settings()


def test_no_language_config_returns_none(monkeypatch):
    monkeypatch.setattr(app_module, "settings", Settings())
    assert _resolve_language(FakeModel(), "x.wav", None) is None


def test_requested_unsupported_language_raises_400(monkeypatch):
    monkeypatch.setattr(app_module, "settings", Settings())
    with pytest.raises(Exception) as exc:
        _resolve_language(FakeModel(detect_scores=[("en", 1.0)]), "x.wav", "klingon")
    assert getattr(exc.value, "status_code", 500) == 400


def test_bad_default_language_raises_400(monkeypatch):
    monkeypatch.setattr(app_module, "settings", Settings(default_language="xx"))
    with pytest.raises(Exception) as exc:
        _resolve_language(FakeModel(), "x.wav", None)
    assert getattr(exc.value, "status_code", 500) == 400


def test_empty_language_treated_as_absent(monkeypatch):
    monkeypatch.setattr(app_module, "settings", Settings(default_language="it"))
    assert _resolve_language(FakeModel(), "x.wav", "") == "it"


def test_whitespace_language_treated_as_absent(monkeypatch):
    monkeypatch.setattr(app_module, "settings", Settings(default_language="it"))
    assert _resolve_language(FakeModel(), "x.wav", "   ") == "it"


def test_mixed_whitespace_language_still_validates(monkeypatch):
    # Nested whitespace is fine; the code is trimmed, then validated.
    monkeypatch.setattr(app_module, "settings", Settings(default_language="it"))
    assert _resolve_language(FakeModel(), "x.wav", "  en ") == "en"


# --------------------------------------------------------------------------
# Allowlist parsing
# --------------------------------------------------------------------------

def test_split_allowed_languages():
    assert _split_allowed_languages(None) == []
    assert _split_allowed_languages("") == []
    assert _split_allowed_languages("it, en") == ["it", "en"]
    assert _split_allowed_languages(" it , EN ,, ") == ["it", "en"]
    # no comma -> a single (likely-invalid) token; validation catches it later
    assert _split_allowed_languages("italian english") == ["italian english"]


# --------------------------------------------------------------------------
# Routes (integration via TestClient + swapped store)
# --------------------------------------------------------------------------

def _audio_files() -> dict:
    return {"file": ("t.wav", _sine_wav(), "audio/wav")}


def test_route_applies_selected_language_to_transcribe():
    model = FakeModel(detect_scores=[("de", 0.9), ("it", 0.5), ("en", 0.4)])
    cfg = Settings(idle_unload_s=0, allowed_languages="it,en")
    with _make_client(_make_store(model), cfg) as c:
        r = c.post(
            "/v1/audio/transcriptions",
            files=_audio_files(),
            data={"model": "whisper-1"},
        )
    assert r.status_code == 200
    assert model.transcribe_kwargs[-1].get("language") == "it"


def test_route_client_language_overrides_default():
    model = FakeModel()
    cfg = Settings(idle_unload_s=0, default_language="it")
    with _make_client(_make_store(model), cfg) as c:
        r = c.post(
            "/v1/audio/transcriptions",
            files=_audio_files(),
            data={"model": "whisper-1", "language": "en"},
        )
    assert r.status_code == 200
    assert model.transcribe_kwargs[-1].get("language") == "en"


def test_route_default_prompt_applied_when_none_sent():
    model = FakeModel()
    cfg = Settings(idle_unload_s=0, initial_prompt="The transcript is:")
    with _make_client(_make_store(model), cfg) as c:
        r = c.post(
            "/v1/audio/transcriptions",
            files=_audio_files(),
            data={"model": "whisper-1"},
        )
    assert r.status_code == 200
    assert model.transcribe_kwargs[-1].get("initial_prompt") == "The transcript is:"


def test_route_client_prompt_wins_over_default_prompt():
    model = FakeModel()
    cfg = Settings(idle_unload_s=0, initial_prompt="server default")
    with _make_client(_make_store(model), cfg) as c:
        r = c.post(
            "/v1/audio/transcriptions",
            files=_audio_files(),
            data={"model": "whisper-1", "prompt": "client prompt"},
        )
    assert r.status_code == 200
    assert model.transcribe_kwargs[-1].get("initial_prompt") == "client prompt"


def test_route_applies_default_language_to_translation():
    model = FakeModel()
    cfg = Settings(idle_unload_s=0, default_language="it")
    with _make_client(_make_store(model), cfg) as c:
        r = c.post(
            "/v1/audio/translations",
            files=_audio_files(),
            data={"model": "whisper-1", "response_format": "text"},
        )
    assert r.status_code == 200
    # translation path resolves language too (source detection hint)
    assert model.transcribe_kwargs[-1].get("language") == "it"


def test_route_no_language_sent_and_no_config_passes_nothing():
    model = FakeModel()
    cfg = Settings(idle_unload_s=0)
    with _make_client(_make_store(model), cfg) as c:
        r = c.post(
            "/v1/audio/transcriptions",
            files=_audio_files(),
            data={"model": "whisper-1"},
        )
    assert r.status_code == 200
    assert "language" not in model.transcribe_kwargs[-1]


def test_health_reports_language_config():
    model = FakeModel()
    cfg = Settings(idle_unload_s=0, default_language="it", allowed_languages="it,en")
    with _make_client(_make_store(model), cfg) as c:
        body = c.get("/health").json()
    assert body["default_language"] == "it"
    assert body["allowed_languages"] == "it,en"
    assert body["loaded"] is True


def test_route_unsupported_client_language_400():
    model = FakeModel()
    cfg = Settings(idle_unload_s=0)
    with _make_client(_make_store(model), cfg) as c:
        r = c.post(
            "/v1/audio/transcriptions",
            files=_audio_files(),
            data={"model": "whisper-1", "language": "klingon"},
        )
    assert r.status_code == 400


def test_route_whitespace_language_uses_default():
    model = FakeModel()
    cfg = Settings(idle_unload_s=0, default_language="it")
    with _make_client(_make_store(model), cfg) as c:
        r = c.post(
            "/v1/audio/transcriptions",
            files=_audio_files(),
            data={"model": "whisper-1", "language": "   "},
        )
    assert r.status_code == 200
    assert model.transcribe_kwargs[-1].get("language") == "it"
