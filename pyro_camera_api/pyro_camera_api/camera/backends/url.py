# Copyright (C) 2022-2025, Pyronear.
# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

from __future__ import annotations

import logging
import re
from io import BytesIO
from typing import List, Optional, Tuple
from urllib.parse import urlparse, urlunparse

import requests
from PIL import Image
from requests.auth import HTTPBasicAuth, HTTPDigestAuth

from pyro_camera_api.camera.base import BaseCamera

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
        Remove user:pass@host from URL and return clean URL plus optional credentials.

        Returns:
            tuple[str, Optional[tuple[str, str]]]: (clean_url, (user, password)) or (clean_url, None).
        """
        if "@" in parsed.netloc:
            creds, hostport = parsed.netloc.rsplit("@", 1)
            user, password = creds.split(":", 1) if ":" in creds else (creds, "")
            cleaned = parsed._replace(netloc=hostport)
            return urlunparse(cleaned), (user, password)
        return urlunparse(parsed), None

    @staticmethod
    def _extract_query_credentials(url: str) -> Optional[Tuple[str, str]]:
        """
        Detect query patterns like ?usr=...&pwd=... and return (user, password) if found.
        """
        m_usr = re.search(r"[?&]usr=([^&]+)", url)
        m_pwd = re.search(r"[?&]pwd=([^&]+)", url)
        if m_usr and m_pwd:
            return m_usr.group(1), m_pwd.group(1)
        return None

    @staticmethod
    def _redact(url: str) -> str:
        """
        Return a log friendly version of the URL with credentials masked.
        """
        parsed = urlparse(url)
        netloc = parsed.netloc.split("@")[-1]
        cleaned = parsed._replace(netloc=netloc)
        redacted = urlunparse(cleaned)
        redacted = re.sub(r"(usr|user|username)=([^&]+)", r"\1=***", redacted, flags=re.IGNORECASE)
        redacted = re.sub(r"(pwd|pass|password)=([^&]+)", r"\1=***", redacted, flags=re.IGNORECASE)
        return redacted

    def _fetch_image(self, target_url: str, auth=None) -> Optional[Image.Image]:
        """
        Perform the HTTP GET and try to decode the response as an image.

        This method logs detailed information when the request fails so that
        camera side errors can be diagnosed from the logs.
        """
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0 Safari/537.36"
            ),
            "Accept": "image/*",
        }

        redacted = self._redact(target_url)

        try:
            response = requests.get(
                target_url,
                headers=headers,
                timeout=self.timeout,
                auth=auth,
            )
        except requests.RequestException as exc:
            logger.error("Request to %s failed, %s", redacted, exc)
            return None

        if response.status_code != 200:
            # Log status plus a small head of headers and body to help debugging
            header_sample = dict(list(response.headers.items())[:5])
            body_head = response.content[:200]
            logger.error(
                "HTTP error for %s, status=%s, headers=%r, body_head=%r",
                redacted,
                response.status_code,
                header_sample,
                body_head,
            )
            return None

        if not response.content:
            logger.error("Empty response content from %s, status=%s", redacted, response.status_code)
            return None

        try:
            img = Image.open(BytesIO(response.content)).convert("RGB")
        except Exception as exc:
            logger.error("Error decoding image from %s, %s", redacted, exc)
            return None

        return img

    def capture(self, pos_id: Optional[int] = None) -> Optional[Image.Image]:
        """
        Fetch a single snapshot from the configured URL and return it as a Pillow Image.

        For URL cameras pos_id is ignored but kept for API compatibility.
        The method tries multiple URL plus auth combinations:
        original URL, URL without inline credentials, then with optional
        basic or digest authentication inferred from inline credentials or
        usr and pwd parameters in the query string.
        """
        _ = pos_id

        parsed = urlparse(self.url)
        clean_url, auth_tuple = self._strip_credentials(parsed)
        query_auth = self._extract_query_credentials(self.url)

        auth_candidates: List[Optional[object]] = [None]

        if query_auth:
            user, password = query_auth
            auth_candidates.append(HTTPBasicAuth(user, password))
            auth_candidates.append(HTTPDigestAuth(user, password))

        if auth_tuple:
            user, password = auth_tuple
            auth_candidates.append(HTTPBasicAuth(user, password))
            auth_candidates.append(HTTPDigestAuth(user, password))

        candidate_urls = [self.url, clean_url]
        seen: set[str] = set()

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
