# Copyright (C) 2022-2026, Pyronear.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.


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

    # force=True clears existing handlers (FastAPI reload, uvicorn own config)
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )

    # Reduce noise from external libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("ffmpeg").setLevel(logging.WARNING)

    # The engine polls capture endpoints every few seconds; one access line per request
    # drowns the log, so keep them for debug runs only.
    logging.getLogger("uvicorn.access").setLevel(logging.DEBUG if level <= logging.DEBUG else logging.WARNING)
