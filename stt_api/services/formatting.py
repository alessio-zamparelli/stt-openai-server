"""Response formatting: OpenAI JSON shapes, plain text, SRT, and VTT."""

from __future__ import annotations

from typing import Optional

from fastapi import HTTPException
from fastapi.responses import PlainTextResponse

FORMAT_MAP = {
    "json": "json",
    "text": "txt",
    "srt": "srt",
    "verbose_json": "json",
    "vtt": "vtt",
}


def _format_response(
    segments,
    text: str,
    language: str,
    response_format: str,
    task: str,
) -> dict:
    """Format faster-whisper output into an OpenAI-compatible dict."""

    if response_format in ("text", "txt", "vtt", "srt"):
        # For text-like formats, faster-whisper returns a plain string from
        # the file written to disk.  We handle those in the endpoint.
        return {"text": text}

    # json / verbose_json
    return {
        "task": task,
        "language": language,
        "segments": segments if segments else [],
        "text": text,
    }


def _ts_srt(seconds: float) -> str:
    """Seconds -> SRT timestamp (HH:MM:SS,mmm)."""
    try:
        ms = max(0, int(round(seconds * 1000)))
    except (TypeError, ValueError, OverflowError):
        ms = 0
    h, rem = divmod(ms, 3600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _ts_vtt(seconds: float) -> str:
    """Seconds -> WebVTT timestamp (HH:MM:SS.mmm)."""
    return _ts_srt(seconds).replace(",", ".")


def _to_srt(segments) -> str:
    blocks = []
    for i, seg in enumerate(segments, start=1):
        blocks.append(
            f"{i}\n{_ts_srt(seg['start'])} --> {_ts_srt(seg['end'])}\n{seg['text'].strip()}\n"
        )
    return "\n".join(blocks)


def _to_vtt(segments) -> str:
    blocks = ["WEBVTT\n"]
    for seg in segments:
        blocks.append(
            f"\n{_ts_vtt(seg['start'])} --> {_ts_vtt(seg['end'])}\n{seg['text'].strip()}\n"
        )
    return "".join(blocks)


def _validate_response_format(response_format: Optional[str]) -> str:
    """Resolve response_format or raise 400, matching OpenAI behaviour."""
    fmt = FORMAT_MAP.get(response_format or "json")
    if fmt is None:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Invalid response_format: {response_format!r}. "
                f"Expected one of: {', '.join(sorted(FORMAT_MAP))}."
            ),
        )
    return fmt


def _text_response(fmt: str, segment_list, full_text: str) -> PlainTextResponse:
    """Return plain-text / subtitle responses, OpenAI style."""
    if fmt == "txt":
        return PlainTextResponse(full_text)
    if fmt == "srt":
        return PlainTextResponse(_to_srt(segment_list), media_type="application/x-subrip")
    return PlainTextResponse(_to_vtt(segment_list), media_type="text/vtt")
