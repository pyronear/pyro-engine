# Copyright (C) 2022-2026, Pyronear.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.


from __future__ import annotations

import logging
from urllib.parse import urlsplit, urlunsplit

__all__ = ["redact_url"]

logger = logging.getLogger(__name__)


def redact_url(url: str) -> str:
    """Mask the userinfo part of a URL so credentials never reach the logs.

    The URL is parsed instead of being split on "@": a password may itself contain "@" (only
    the last one delimits the host, so splitting on the first leaks the rest of the password),
    and a path may contain "@" while the URL carries no credentials at all.

    Values that are not a scheme://host URL are returned untouched, so this is safe to map
    over a whole command line.
    """
    try:
        parts = urlsplit(url)
        if "@" not in parts.netloc:
            return url
        host = parts.netloc.rpartition("@")[2]
        return urlunsplit((parts.scheme, f"***:***@{host}", parts.path, parts.query, parts.fragment))
    except ValueError:
        # Unparsable, so never echo it back: it may still hold credentials.
        logger.debug("Could not parse URL for redaction")
        if "://" in url and "@" in url:
            return f"{url.split('://', 1)[0]}://***:***@{url.rsplit('@', 1)[1]}"
        return url
