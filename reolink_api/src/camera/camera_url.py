# Copyright (C) 2022-2025, Pyronear.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

import logging
import re
from io import BytesIO
from typing import List, Optional, Tuple
from urllib.parse import urlparse, urlunparse

import requests
from PIL import Image
from requests.auth import HTTPBasicAuth, HTTPDigestAuth

logger = logging.getLogger("URLCamera")
logger.setLevel(logging.INFO)


class URLCamera:
    """
    Camera that exposes a direct HTTP(S) snapshot endpoint that returns an image.

    Supports:
    - http://user:pass@host/...
    - http://host/... ?usr=...&pwd=...

    capture() returns a Pillow Image in RGB, or None on failure.
    """

    def __init__(self, url: str, timeout: int = 5, cam_type: str = "static"):
        self.url = url
        self.timeout = timeout
        self.cam_type = cam_type
        self.last_images: dict[int, Image.Image] = {}
        logger.debug(
            "Initialized URLCamera url=%s timeout=%s cam_type=%s",
            self._redact(url),
            timeout,
            cam_type,
        )

    @staticmethod
    def _strip_credentials(parsed) -> Tuple[str, Optional[Tuple[str, str]]]:
        """
        Remove user:pass@host from URL.
        Returns (clean_url, (user, pass)) or (clean_url, None)
        """
        netloc = parsed.netloc
        if "@" in netloc:
            creds, hostport = netloc.rsplit("@", 1)
            if ":" in creds:
                user, password = creds.split(":", 1)
            else:
                user, password = creds, ""
            cleaned = parsed._replace(netloc=hostport)
            return urlunparse(cleaned), (user, password)
        return urlunparse(parsed), None

    @staticmethod
    def _extract_query_credentials(url: str) -> Optional[Tuple[str, str]]:
        """
        Detect patterns like ...?usr=admin&pwd=1234
        """
        m_usr = re.search(r"[?&]usr=([^&]+)", url)
        m_pwd = re.search(r"[?&]pwd=([^&]+)", url)
        if m_usr and m_pwd:
            return (m_usr.group(1), m_pwd.group(1))
        return None

    @staticmethod
    def _redact(url: str) -> str:
        """
        Produce a log friendly version of the URL with credentials masked.
        """
        parsed = urlparse(url)
        # remove user:pass@
        netloc = parsed.netloc.split("@")[-1]
        cleaned = parsed._replace(netloc=netloc)
        redacted = urlunparse(cleaned)
        # mask usr, user, username, pwd, pass, password in query string
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
            if r.status_code == 200 and r.content:
                try:
                    img = Image.open(BytesIO(r.content)).convert("RGB")
                    return img
                except Exception as e:
                    logger.debug(
                        "Failed to decode image from %s: %s",
                        self._redact(target_url),
                        e,
                    )
        except requests.RequestException as e:
            logger.error("Request to %s failed: %s", self._redact(target_url), e)
        return None

    def capture(self) -> Optional[Image.Image]:
        """
        Try multiple auth strategies and URL variants.
        """
        logger.info("Starting URL capture for %s", self._redact(self.url))

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

        logger.warning("All URL attempts failed for %s", self._redact(self.url))
        return None
