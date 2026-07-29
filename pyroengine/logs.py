# Copyright (C) 2022-2026, Pyronear.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.


import logging
import os
import sys

__all__ = ["setup_logging"]


def setup_logging() -> None:
    """
    Configure application wide logging.

    Priority:
      - LOG_LEVEL environment variable
      - default INFO

    Only entrypoints should call this: library modules must not configure the root logger.
    """
    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)

    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )

    # Reduce noise from external libraries. Never below the configured level: a child logger
    # level is not re-checked against the root one, so WARNING here would leak warnings
    # through an ERROR root.
    noisy_level = max(level, logging.WARNING)
    logging.getLogger("urllib3").setLevel(noisy_level)
    logging.getLogger("PIL").setLevel(noisy_level)
