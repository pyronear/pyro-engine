# Copyright (C) 2022-2026, Pyronear.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.


from __future__ import annotations

import logging
import os
import sys

from pyro_camera_api.utils.redact import RedactSecretsFilter


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
    handler = logging.StreamHandler(sys.stdout)
    # Credentials are scrubbed at the handler so lines relayed from subprocesses
    # (ffmpeg stderr) and third-party loggers are covered too, not only our own calls.
    handler.addFilter(RedactSecretsFilter())
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[handler],
        force=True,
    )

    # Reduce noise from external libraries. Never below the configured level: a child logger
    # level is not re-checked against the root one, so WARNING here would leak warnings
    # through an ERROR root.
    noisy_level = max(level, logging.WARNING)
    logging.getLogger("urllib3").setLevel(noisy_level)
    logging.getLogger("PIL").setLevel(noisy_level)
    logging.getLogger("ffmpeg").setLevel(noisy_level)

    # Uvicorn installs its own handlers with propagate=False and an explicit INFO level, so its
    # startup and access lines would keep a timestamp-less format of their own and stay visible
    # even at LOG_LEVEL=ERROR. Hand them to the root handler and reset their level to NOTSET so
    # they inherit the configured one.
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        uvicorn_logger = logging.getLogger(name)
        uvicorn_logger.handlers.clear()
        uvicorn_logger.propagate = True
        uvicorn_logger.setLevel(logging.NOTSET)

    # The engine polls capture endpoints every few seconds; one access line per request
    # drowns the log, so keep them for debug runs only.
    logging.getLogger("uvicorn.access").setLevel(logging.DEBUG if level <= logging.DEBUG else noisy_level)
