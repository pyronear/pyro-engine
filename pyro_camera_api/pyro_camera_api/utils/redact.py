# Copyright (C) 2022-2026, Pyronear.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.


from __future__ import annotations

import logging
import re
from urllib.parse import urlsplit, urlunsplit

__all__ = ["RedactSecretsFilter", "redact_text", "redact_url"]

# scheme:// then a run of non-space/non-slash characters ending in "@": the greedy "*"
# reaches the LAST "@" of the run, so a password containing "@" is fully masked, and a
# "@" later in the path (after a "/") never matches.
_USERINFO = re.compile(r"(\w[\w+.-]*://)[^\s/]*@")

# Credential-like key=value pairs in query strings, MediaMTX streamid forms
# ("#!::u=user,p=pass,m=publish") and ffmpeg command lines. "u"/"p" are kept for the
# streamid form; the lookbehind keeps them from matching inside longer words.
_SENSITIVE_KV = re.compile(
    r"(?<![\w-])(password|passwd|pwd|passphrase|pass|secret|token|access_token|api_?key|"
    r"auth|username|usr|user|[up])=([^&,\s'\"]+)",
    re.IGNORECASE,
)


def redact_text(text: str) -> str:
    """Mask URL userinfo and credential-like key=value pairs anywhere in a string."""
    return _SENSITIVE_KV.sub(r"\1=***", _USERINFO.sub(r"\1***:***@", text))


def redact_url(url: str) -> str:
    """Mask the userinfo part and credential-like query parameters of a URL.

    The userinfo is parsed instead of split on "@": a password may itself contain "@" (only
    the last one delimits the host), and a path may contain "@" while the URL carries no
    credentials at all.

    Values that are not a scheme://host URL still get key=value masking, so this is safe to
    map over a whole command line.
    """
    try:
        parts = urlsplit(url)
        if "@" in parts.netloc:
            host = parts.netloc.rpartition("@")[2]
            url = urlunsplit((parts.scheme, f"***:***@{host}", parts.path, parts.query, parts.fragment))
    except ValueError:
        # Unparsable, so never echo it back: it may still hold credentials.
        return "***"
    return _SENSITIVE_KV.sub(r"\1=***", url)


class RedactSecretsFilter(logging.Filter):
    """Scrub credentials from every record at the handler, whatever the source.

    Per-call redaction cannot cover lines relayed from subprocesses (ffmpeg echoes the
    full input URL on its own stderr) or third-party loggers, so the last line of defense
    sits on the handler itself.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        redacted = redact_text(msg)
        if redacted != msg:
            record.msg = redacted
            record.args = None
        return True
