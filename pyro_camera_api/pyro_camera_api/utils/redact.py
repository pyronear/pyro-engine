# Copyright (C) 2022-2026, Pyronear.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.


from __future__ import annotations

import logging

__all__ = ["redact_url"]

logger = logging.getLogger(__name__)


def redact_url(url: str) -> str:
    """Mask the userinfo part of a URL so credentials never reach the logs."""
    try:
        if "://" in url and "@" in url:
            scheme, rest = url.split("://", 1)
            after_at = rest.split("@", 1)[1]
            return f"{scheme}://***:***@{after_at}"
    except Exception as exc:
        logger.debug("Could not redact credentials from URL: %s", exc)
    return url
