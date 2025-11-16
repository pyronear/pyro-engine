# pyro_camera_api/core/logging.py

from __future__ import annotations

import logging
import os


def setup_logging() -> None:
    """
    Configure application wide logging.

    Uses LOG_LEVEL environment variable if set, defaults to INFO.
    """
    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)

    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Optional: lower verbosity for noisy libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("ffmpeg").setLevel(logging.WARNING)
