"""Language validation and resolution."""

from __future__ import annotations

import logging
from typing import Any, List, Optional

from fastapi import HTTPException

from .. import runtime
from .audio import _decode_audio_prefix

logger = logging.getLogger(__name__)

_LANGUAGE_DETECT_SECONDS = 30.0  # one Whisper encoder window


def _split_allowed_languages(raw: Optional[str]) -> List[str]:
    """Parse a STT_LANGUAGES value like 'it, en' -> ['it', 'en']."""
    if not raw:
        return []
    return [code.strip().lower() for code in raw.split(",") if code.strip()]


def _supported_language_codes(model: Any) -> Optional[set]:
    """Whisper's supported ISO codes for a model (None when unavailable)."""
    codes = getattr(model, "supported_languages", None)
    return set(codes) if codes else None


def _check_language(code: Optional[str], supported: Optional[set]) -> Optional[str]:
    """Normalize/validate an ISO code, raising a clear 400 on bad values."""
    if not code:
        return None
    code = code.strip().lower()
    if not code:  # whitespace-only counts as "no preference", like an empty field
        return None
    if supported and code not in supported:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported language {code!r}. Supported: {sorted(supported)}. "
                "Pass an ISO-639-1 code such as 'en' or 'it'."
            ),
        )
    return code


def _resolve_language(model: Any, audio: Any, requested: Optional[str]) -> Optional[str]:
    """Resolve the `language` hint forwarded to faster-whisper.

    Priority:
      1. The client's `language` field (explicit override; an empty or
         whitespace-only value is treated as absent and falls through).
      2. Server default  STT_LANGUAGE.
      3. STT_LANGUAGES allowlist:
         - a single code is used verbatim;
         - several codes enable *constrained* detection: run faster-whisper's
           language detector, then pick the highest-probability *allowed* code
           (plain auto-detect ignores the allowlist entirely).
      4. None -> faster-whisper full auto-detect.
    """
    supported = _supported_language_codes(model)

    if requested is not None:
        checked = _check_language(requested, supported)
        if checked:  # empty/whitespace-only -> absent, fall through to defaults
            return checked
    if runtime.settings.default_language:
        return _check_language(runtime.settings.default_language, supported)

    allowed = _split_allowed_languages(runtime.settings.allowed_languages)
    if len(allowed) == 1:
        return _check_language(allowed[0], supported)
    if len(allowed) > 1:
        # Validate the whole allowlist up front for a clear 400.
        for code in allowed:
            _check_language(code, supported)
        # Constrained detection needs only the first 30 s window; faster-whisper
        # returns the full ranked list, so we can ignore every code outside the
        # allowlist. Serialize the encoder pass on the same lock as transcribe().
        samples = _decode_audio_prefix(audio, _LANGUAGE_DETECT_SECONDS)
        if samples is None or not samples.size:
            return None  # undecodable prefix -> let transcribe() auto-detect
        with runtime._inference_lock:
            _code, _prob, all_scores = model.detect_language(samples)
        best: Optional[tuple] = None
        for code, prob in all_scores:
            candidate = code.strip().lower()
            if candidate in allowed and (best is None or prob > best[1]):
                best = (candidate, prob)
        return best[0] if best else None
    return None
