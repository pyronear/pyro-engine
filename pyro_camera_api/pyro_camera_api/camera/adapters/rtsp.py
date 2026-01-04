# Copyright (C) 2022-2026, Pyronear.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

from __future__ import annotations

import logging
import subprocess
import tempfile
from typing import Optional

import cv2
import numpy as np
from PIL import Image

from pyro_camera_api.camera.base import BaseCamera

logger = logging.getLogger(__name__)


def _safe_url(u: str) -> str:
    try:
        if "://" in u and "@" in u:
            scheme, rest = u.split("://", 1)
            after_at = rest.split("@", 1)[1]
            return f"{scheme}://***:***@{after_at}"
    except Exception:
        pass
    return u


class RTSPCamera(BaseCamera):
    """RTSP camera that grabs one frame via ffmpeg with a hard timeout, returns a Pillow Image."""

    def __init__(
        self,
        camera_id: str,
        rtsp_url: str,
        ip_address: str = "",
        cam_type: str = "rtsp",
        default_timeout_s: float = 5.0,
    ):
        super().__init__(camera_id=camera_id, cam_type=cam_type)
        self.rtsp_url = rtsp_url
        self.ip_address = ip_address
        self.default_timeout_s = default_timeout_s

    def _grab_frame_ffmpeg(self, timeout_s: float) -> Optional[np.ndarray]:
        """
        Call ffmpeg to grab a single JPEG frame with a hard process timeout.
        Read it back with OpenCV and return a BGR ndarray.
        """
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=True) as tmp:
            cmd = [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-rtsp_transport",
                "tcp",
                "-i",
                self.rtsp_url,
                "-frames:v",
                "1",
                "-y",
                tmp.name,
            ]
            try:
                subprocess.run(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=timeout_s,
                    check=False,
                )
            except subprocess.TimeoutExpired:
                logger.error("ffmpeg timed out after %.1fs for %s", timeout_s, _safe_url(self.rtsp_url))
                return None

            frame = cv2.imread(tmp.name)
            if frame is None:
                logger.error("ffmpeg returned no image for %s", _safe_url(self.rtsp_url))
                return None
            return frame

    def capture(
        self,
        pos_id: Optional[int] = None,
        timeout_s: Optional[float] = None,
    ) -> Optional[Image.Image]:
        """
        Grab one frame through ffmpeg, convert to RGB, return as Pillow Image.

        pos_id is accepted for API compatibility but ignored for RTSP cameras.
        """
        _ = pos_id  # unused but keeps the same signature as Reolink in the routes

        timeout = self.default_timeout_s if timeout_s is None else timeout_s
        frame = self._grab_frame_ffmpeg(timeout)
        if frame is None:
            return None
        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return Image.fromarray(rgb)
        except Exception as e:
            logger.error("Error converting frame for %s: %s", _safe_url(self.rtsp_url), e)
            return None
