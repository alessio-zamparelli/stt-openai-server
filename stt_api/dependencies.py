"""Shared FastAPI dependencies."""

from __future__ import annotations

from . import runtime
from .config import Settings
from .services.model_store import ModelStore


def get_settings() -> Settings:
    """Return the live process settings (late-bound for test overrides)."""
    return runtime.settings


def get_store() -> ModelStore:
    """Return the live model store (late-bound for test overrides)."""
    return runtime.store
