# pyro_camera_api/camera/backends/url.py
# Copyright (C) 2022-2025, Pyronear.
# Licensed under the Apache License 2.0

from __future__ import annotations

import logging
import re
from io import BytesIO
from typing import List, Optional, Tuple
from urllib.parse import urlparse, urlunparse

import requests
from PIL import Image
from pyro_camera_api.camera.base import BaseCamera
from requests.auth import HTTPBasicAuth, HTTPDigestAuth

logger = logging.getLogger(__name__)


class URLCamera(BaseCamera):
    """Camera that exposes an HTTP or HTTPS snapshot URL and returns one frame as a Pillow Image."""

    def __init__(
        self,
        camera_id: str,
        url: str,
        timeout: int = 5,
        cam_type: str = "static",
    ):
        super().__init__(camera_id=camera_id, cam_type=cam_type)
        self.url = url
        self.timeout = timeout

    @staticmethod
    def _strip_credentials(parsed) -> Tuple[str, Optional[Tuple[str, str]]]:
        """
        Remove user:pass@host from URL.
        Return (clean_url, (user, pass)) or (clean_url, None).
        """
        if "@" in parsed.netloc:
            creds, hostport = parsed.netloc.rsplit("@", 1)
            user, password = creds.split(":", 1) if ":" in creds else (creds, "")
            cleaned = parsed._replace(netloc=hostport)
            return urlunparse(cleaned), (user, password)
        return urlunparse(parsed), None

    @staticmethod
    def _extract_query_credentials(url: str) -> Optional[Tuple[str, str]]:
        """Detect query patterns like ?usr=...&pwd=... and return (user, pass) if found."""
        m_usr = re.search(r"[?&]usr=([^&]+)", url)
        m_pwd = re.search(r"[?&]pwd=([^&]+)", url)
        if m_usr and m_pwd:
            return m_usr.group(1), m_pwd.group(1)
        return None

    @staticmethod
    def _redact(url: str) -> str:
        """Return a log friendly version of the URL with credentials masked."""
        parsed = urlparse(url)
        netloc = parsed.netloc.split("@")[-1]
        cleaned = parsed._replace(netloc=netloc)
        redacted = urlunparse(cleaned)
        redacted = re.sub(r"(usr|user|username)=([^&]+)", r"\1=***", redacted, flags=re.IGNORECASE)
        redacted = re.sub(r"(pwd|pass|password)=([^&]+)", r"\1=***", redacted, flags=re.IGNORECASE)
        return redacted

    def _fetch_image(self, target_url: str, auth=None) -> Optional[Image.Image]:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0 Safari/537.36"
            ),
            "Accept": "image/*",
        }

        try:
            r = requests.get(target_url, headers=headers, timeout=self.timeout, auth=auth)
        except requests.RequestException as e:
            logger.error("Request to %s failed: %s", self._redact(target_url), e)
            return None

        if r.status_code != 200 or not r.content:
            logger.error("Failed to read image content from %s", self._redact(target_url))
            return None

        try:
            return Image.open(BytesIO(r.content)).convert("RGB")
        except Exception as e:
            logger.error("Error decoding image from %s: %s", self._redact(target_url), e)
            return None

    def capture(self, pos_id: Optional[int] = None) -> Optional[Image.Image]:
        """
        Try to fetch a single snapshot using different auth methods and return it as a Pillow Image.

        pos_id is accepted for API compatibility but ignored for URL cameras.
        """
        _ = pos_id  # unused, keeps same capture signature as other backends

        parsed = urlparse(self.url)
        clean_url, auth_tuple = self._strip_credentials(parsed)
        query_auth = self._extract_query_credentials(self.url)

        auth_candidates: List[Optional[object]] = [None]

        if query_auth:
            u, p = query_auth
            auth_candidates.append(HTTPBasicAuth(u, p))
            auth_candidates.append(HTTPDigestAuth(u, p))

        if auth_tuple:
            u, p = auth_tuple
            auth_candidates.append(HTTPBasicAuth(u, p))
            auth_candidates.append(HTTPDigestAuth(u, p))

        candidate_urls = [self.url, clean_url]
        seen = set()

        for auth in auth_candidates:
            for candidate_url in candidate_urls:
                if candidate_url in seen:
                    continue
                seen.add(candidate_url)

                img = self._fetch_image(candidate_url, auth=auth)
                if img is not None:
                    logger.info(
                        "URL capture OK from %s (%s), size=%s",
                        self._redact(candidate_url),
                        auth.__class__.__name__ if auth else "no auth",
                        img.size,
                    )
                    return img

        logger.error("All URL attempts failed for %s", self._redact(self.url))
        return None
