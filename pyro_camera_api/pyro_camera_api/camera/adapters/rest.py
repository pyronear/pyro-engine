# Copyright (C) 2022-2026, Pyronear.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.


from __future__ import annotations

import base64
import logging
import re
from io import BytesIO
from typing import Any, Dict, Optional

import requests
from PIL import Image

from pyro_camera_api.camera.base import BaseCamera

logger = logging.getLogger(__name__)

# Header names whose values must never be logged in clear.
_SENSITIVE_HEADERS = {"authorization", "x-api-key", "api-key", "apikey", "token"}
# Query parameters whose values must never be logged in clear.
_SENSITIVE_QUERY = re.compile(r"(token|access_token|api_?key|pwd|password)=([^&]+)", re.IGNORECASE)


class RestSnapshotCamera(BaseCamera):
    """Generic HTTP/REST snapshot camera (capture only).

    Fetches a single frame from an HTTP endpoint that may need custom
    authentication headers and may wrap the image inside a JSON envelope,
    which :class:`URLCamera` cannot handle. One adapter serves many providers
    through configuration:

    - ``response="image"``: the body is the raw image bytes.
    - ``response="json"``: the body is JSON; ``json_path`` (dot-separated)
      points to the payload field, decoded either as base64 (``encoding="base64"``)
      or as a nested image URL to re-fetch (``encoding="url"``).

    Example (vigilant.cat): ``response="json"``, ``json_path="data"``,
    ``encoding="base64"`` with an ``Authorization: Bearer <token>`` header.
    """

    def __init__(
        self,
        camera_id: str,
        url: str,
        headers: Optional[Dict[str, str]] = None,
        response: str = "image",
        json_path: Optional[str] = None,
        encoding: str = "base64",
        timeout: int = 15,
        retries: int = 2,
        cam_type: str = "static",
    ) -> None:
        super().__init__(camera_id=camera_id, cam_type=cam_type)
        self.url = url
        self.headers = headers or {}
        self.response = response
        self.json_path = json_path
        self.encoding = encoding
        self.timeout = timeout
        self.retries = retries
        self._session = requests.Session()

    @staticmethod
    def _redact_headers(headers: Dict[str, str]) -> Dict[str, str]:
        """Mask sensitive header values for safe logging."""
        return {k: ("***" if k.lower() in _SENSITIVE_HEADERS else v) for k, v in headers.items()}

    @staticmethod
    def _redact_url(url: str) -> str:
        """Mask credential-like query parameters for safe logging."""
        return _SENSITIVE_QUERY.sub(r"\1=***", url)

    @staticmethod
    def _dig(payload: Any, path: str) -> Any:
        """Navigate a dot-separated path into a nested JSON structure."""
        node = payload
        for key in path.split("."):
            node = node[key]
        return node

    def _decode_bytes(self, resp: requests.Response) -> bytes:
        """Turn the HTTP response into raw image bytes according to the config."""
        if self.response == "image":
            return resp.content

        if self.response == "json":
            if not self.json_path:
                raise ValueError("response='json' requires 'json_path'")
            field = self._dig(resp.json(), self.json_path)

            if self.encoding == "base64":
                if isinstance(field, str) and field.startswith("data:"):
                    # Strip a data URI prefix such as "data:image/jpeg;base64,".
                    field = field.split(",", 1)[1]
                return base64.b64decode(field)

            if self.encoding == "url":
                sub = self._session.get(field, headers=self.headers, timeout=self.timeout)
                sub.raise_for_status()
                return sub.content

            raise ValueError(f"unknown encoding '{self.encoding}'")

        raise ValueError(f"unknown response mode '{self.response}'")

    def capture(self, patrol_id: Optional[int] = None) -> Optional[Image.Image]:
        """Fetch a single snapshot and return it as a Pillow Image, or None on failure."""
        _ = patrol_id  # kept for API compatibility with PTZ adapters
        attempts = self.retries + 1
        redacted_url = self._redact_url(self.url)
        redacted_headers = self._redact_headers(self.headers)

        for attempt in range(1, attempts + 1):
            try:
                resp = self._session.get(self.url, headers=self.headers, timeout=self.timeout)
                resp.raise_for_status()
                img = Image.open(BytesIO(self._decode_bytes(resp))).convert("RGB")
                logger.info("REST capture OK from %s, size=%s", redacted_url, img.size)
                return img
            except Exception as exc:
                logger.warning(
                    "REST capture failed for %s (attempt %d/%d), headers=%r, %s",
                    redacted_url,
                    attempt,
                    attempts,
                    redacted_headers,
                    exc,
                )

        logger.error("REST capture giving up for %s after %d attempts", redacted_url, attempts)
        return None
