"""Compatibility launcher for the STT API server.

Application code lives in :mod:`stt_api`. This module preserves the historical
``python app.py`` / ``python app.py --healthcheck`` entrypoints (Docker, e2e
tests, and local docs) and exposes the FastAPI instance as ``app``.
"""

from stt_api.main import app, run

if __name__ == "__main__":
    run()
