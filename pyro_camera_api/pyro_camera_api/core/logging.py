# pyro_camera_api/core/logging.py
from __future__ import annotations

import logging
import os
import sys


def setup_logging() -> None:
    """
    Configure application wide logging.

    Priority:
      - LOG_LEVEL environment variable
      - default INFO
    Safe for FastAPI/Uvicorn reload (clears existing handlers).
    """
    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)

    # Avoid duplicate handlers (FastAPI reload)
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # Reduce noise from external libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("ffmpeg").setLevel(logging.WARNING)
