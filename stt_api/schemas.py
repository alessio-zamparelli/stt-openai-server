"""Pydantic request/response shapes (OpenAI-compatible)."""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class TranscriptionCreateRequest(BaseModel):
    """Mirrors OpenAI's /v1/audio/transcriptions request body."""

    # When sending via multipart/form-data, these come from form fields.
    # They are defined here for documentation and for JSON fallback.
    model: Optional[str] = Field(
        default=None, description="ID of the model to use (ignored — uses server default)."
    )
    language: Optional[str] = Field(
        default=None, description="Language of the input audio (e.g. 'en')."
    )
    prompt: Optional[str] = Field(
        default=None,
        description="An optional prompt to guide the model's behaviour.",
    )
    response_format: Optional[str] = Field(
        default="json",
        description="Format of the output: json, text, srt, verbose_json, vtt.",
    )
    temperature: Optional[float] = Field(
        default=0.0, description="Sampling temperature (0 = deterministic).", ge=0.0, le=1.0
    )
    timestamps: Optional[bool] = Field(
        default=True, description="Include timestamped segments in the response."
    )


class TranslationCreateRequest(BaseModel):
    """Mirrors OpenAI's /v1/audio/translations request body."""

    model: Optional[str] = Field(default=None, description="Ignored — uses server default.")
    prompt: Optional[str] = Field(
        default=None,
        description="An optional prompt to guide the model's behaviour.",
    )
    response_format: Optional[str] = Field(
        default="json",
        description="Format of the output: json, text, srt, verbose_json, vtt.",
    )
    temperature: Optional[float] = Field(
        default=0.0, description="Sampling temperature (0 = deterministic).", ge=0.0, le=1.0
    )
    timestamps: Optional[bool] = Field(
        default=True, description="Include timestamped segments in the response."
    )


class TranscriptionResponse(BaseModel):
    """Mirrors OpenAI's transcription response."""

    text: str
    task: str = "transcribe"
    language: Optional[str] = None
    segments: Optional[List[dict]] = None


class TranslationResponse(BaseModel):
    """Mirrors OpenAI's translation response."""

    text: str
    task: str = "translate"
    language: Optional[str] = None
    segments: Optional[List[dict]] = None
