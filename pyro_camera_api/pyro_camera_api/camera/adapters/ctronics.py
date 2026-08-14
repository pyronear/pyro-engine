# Copyright (C) 2022-2026, Pyronear.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.


from __future__ import annotations

import logging
from io import BytesIO
from typing import Optional
from urllib.parse import parse_qsl, urlencode, urljoin, urlparse, urlunparse

import requests
from PIL import Image

from pyro_camera_api.camera.base import BaseCamera

logger = logging.getLogger(__name__)


class CTronicsCamera(BaseCamera):
    """CTronics camera using an authenticated HTTP snapshot endpoint."""

    def __init__(
        self,
        camera_id: str,
        ip_address: str,
        username: str,
        password: str,
        port: int = 80,
        protocol: str = "http",
        snapshot_path: str = "/tmpfs/snap.jpg",
        snapshot_command: Optional[str] = None,
        timeout: float = 5.0,
        model: Optional[str] = None,
        cam_type: str = "static",
    ) -> None:
        super().__init__(camera_id=camera_id, cam_type=cam_type)
        self.ip_address = ip_address
        self.username = username
        self.password = password
        self.port = port
        self.protocol = protocol
        self.snapshot_path = snapshot_path
        self.snapshot_command = snapshot_command
        self.timeout = timeout
        self.model = model

    @property
    def snapshot_url(self) -> str:
        """Build the authenticated snapshot URL without putting credentials in its authority."""
        base = f"{self.protocol}://{self.ip_address}:{self.port}/"
        path = self.snapshot_path.lstrip("/")
        query_params = {"usr": self.username, "pwd": self.password}
        if self.snapshot_command:
            query_params = {"cmd": self.snapshot_command, **query_params}
        query = urlencode(query_params)
        return urljoin(base, f"{path}?{query}")

    @staticmethod
    def _redact_url(url: str) -> str:
        parsed = urlparse(url)
        query = urlencode(
            [
                (key, "***" if key.lower() in {"usr", "user", "pwd", "password"} else value)
                for key, value in parse_qsl(parsed.query)
            ]
        )
        return urlunparse(parsed._replace(query=query))

    def capture(self, patrol_id: Optional[int] = None) -> Optional[Image.Image]:
        """Fetch one JPEG frame, returning ``None`` when capture fails."""
        _ = patrol_id
        url = self.snapshot_url
        redacted_url = self._redact_url(url)
        try:
            response = requests.get(url, timeout=self.timeout)
            response.raise_for_status()
            if not response.content:
                raise ValueError("empty response")
            image = Image.open(BytesIO(response.content)).convert("RGB")
        except Exception as exc:
            logger.error("CTronics capture failed for %s: %s", redacted_url, exc)
            return None

        logger.info("CTronics capture OK for %s, size=%s", redacted_url, image.size)
        return image