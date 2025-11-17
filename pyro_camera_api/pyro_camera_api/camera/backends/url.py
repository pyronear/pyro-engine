# Copyright (C) 2022-2025, Pyronear.
# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

from __future__ import annotations

import logging
import re
from io import BytesIO
from typing import Optional, Tuple
from urllib.parse import urlparse, urlunparse

import requests
from PIL import Image
from requests.auth import HTTPDigestAuth

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
    def _redact(url: str) -> str:
        """
        Mask credentials in URL for safe logging.
        """
        parsed = urlparse(url)
        # Drop user info from netloc
        netloc = parsed.netloc.split("@")[-1]
        cleaned = parsed._replace(netloc=netloc)
        redacted = urlunparse(cleaned)
        # Mask query credentials
        redacted = re.sub(r"(usr|user|username)=([^&]+)", r"\1=***", redacted, flags=re.IGNORECASE)
        redacted = re.sub(r"(pwd|pass|password)=([^&]+)", r"\1=***", redacted, flags=re.IGNORECASE)
        return redacted

    @staticmethod
    def _strip_credentials(parsed) -> Tuple[str, Optional[Tuple[str, str]]]:
        """
        Remove user:pass@ from the URL authority.

        Returns:
            clean_url, (user, password) or clean_url, None
        """
        if "@" in parsed.netloc:
            creds, hostport = parsed.netloc.rsplit("@", 1)
            user, pwd = creds.split(":", 1) if ":" in creds else (creds, "")
            cleaned = parsed._replace(netloc=hostport)
            return urlunparse(cleaned), (user, pwd)
        return urlunparse(parsed), None

    def _fetch_image(self, target_url: str, auth=None) -> Optional[Image.Image]:
        """
        Perform the HTTP GET and try to decode the response as an image.
        """
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0 Safari/537.36"
            ),
            "Accept": "image/*",
        }
        redacted = self._redact(target_url)

        try:
            resp = requests.get(
                target_url,
                headers=headers,
                timeout=self.timeout,
                auth=auth,
            )
        except requests.RequestException as exc:
            logger.error("Request to %s failed, %s", redacted, exc)
            return None

        if resp.status_code != 200:
            header_sample = dict(list(resp.headers.items())[:5])
            body_head = resp.content[:200]
            logger.error(
                "HTTP error for %s, status=%s, headers=%r, body_head=%r",
                redacted,
                resp.status_code,
                header_sample,
                body_head,
            )
            return None

        if not resp.content:
            logger.error("Empty response content from %s, status=%s", redacted, resp.status_code)
            return None

        try:
            img = Image.open(BytesIO(resp.content)).convert("RGB")
        except Exception as exc:
            logger.error("Error decoding image from %s, %s", redacted, exc)
            return None

        return img

    def _capture_foscam_style(self) -> Optional[Image.Image]:
        """
        For URLs that already embed usr and pwd as query params.

        Example:
        http://host:1340/cgi-bin/CGIProxy.fcgi?cmd=snapPicture2&usr=XXX&pwd=YYY
        """
        redacted = self._redact(self.url)
        logger.info("Trying Foscam style URL snapshot for %s", redacted)
        img = self._fetch_image(self.url, auth=None)
        if img is not None:
            logger.info("URL capture OK from %s (no auth), size=%s", redacted, img.size)
        return img

    def _capture_digest_style(self) -> Optional[Image.Image]:
        """
        For URLs that require HTTP Digest authentication.

        Example:
        http://user:pass@host:port/cgi-bin/snapshot.cgi
        """
        parsed = urlparse(self.url)
        clean_url, creds = self._strip_credentials(parsed)

        if not creds:
            logger.error(
                "Digest style URL snapshot requires inline credentials in %s",
                self._redact(self.url),
            )
            return None

        user, pwd = creds
        redacted = self._redact(clean_url)
        logger.info("Trying Digest auth snapshot for %s as user '%s'", redacted, user)

        auth = HTTPDigestAuth(user, pwd)
        img = self._fetch_image(clean_url, auth=auth)
        if img is not None:
            logger.info("URL capture OK from %s (HTTPDigestAuth), size=%s", redacted, img.size)
        return img

    def capture(self, pos_id: Optional[int] = None) -> Optional[Image.Image]:
        """
        Fetch a single snapshot from the configured URL and return it as a Pillow Image.

        For URL cameras pos_id is ignored but kept for API compatibility.

        Current behavior with your configuration:
        - If the URL contains 'CGIProxy.fcgi' it uses the URL as is,
          which works for adf_1340 that uses usr and pwd in the query.
        - Otherwise it expects 'user:password@host' in the URL and uses HTTP Digest auth
          against the cleaned URL without embedded credentials, which is the case
          for adf_1231, adf_1200, adf_1320, adf_5559, adf_5995.
        """
        _ = pos_id

        if "CGIProxy.fcgi" in self.url:
            return self._capture_foscam_style()

        return self._capture_digest_style()
