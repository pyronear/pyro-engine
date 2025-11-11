# Copyright (C) 2022-2025, Pyronear.
# Licensed under the Apache License 2.0
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

import logging
import subprocess
from typing import Optional

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger("RTSPCamera")
logger.setLevel(logging.INFO)


class RTSPCamera:
    """Camera that exposes an RTSP stream and captures one frame as a Pillow Image."""

    def __init__(self, rtsp_url: str, ip_address: str = "", cam_type: str = "rtsp"):
        self.rtsp_url = rtsp_url
        self.ip_address = ip_address
        self.cam_type = cam_type
        self.last_images: dict[int, Image.Image] = {}

    def _grab_frame_with_ffmpeg(self, timeout_ms: int = 5000) -> Optional[np.ndarray]:
        """Grab a single frame from the RTSP stream using ffmpeg, returns a BGR NumPy array."""
        cmd = [
            "ffmpeg",
            "-rtsp_transport",
            "tcp",
            "-stimeout",
            str(timeout_ms * 1000),  # microseconds
            "-i",
            self.rtsp_url,
            "-frames:v",
            "1",
            "-f",
            "image2pipe",
            "-vcodec",
            "mjpeg",
            "pipe:1",
        ]

        try:
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,  # switch to STDOUT for debugging if needed
                check=False,
            )
        except Exception as e:
            logger.error("ffmpeg failed to start for %s: %s", self.rtsp_url, e)
            return None

        if not result.stdout:
            logger.error("ffmpeg returned no data for %s", self.rtsp_url)
            return None

        arr = np.frombuffer(result.stdout, np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            logger.error("cv2.imdecode failed for %s", self.rtsp_url)
        return frame

    def capture(self) -> Optional[Image.Image]:
        """Read a single frame via ffmpeg, return it as a Pillow Image."""
        frame = self._grab_frame_with_ffmpeg(timeout_ms=5000)
        if frame is None:
            return None

        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return Image.fromarray(rgb)
        except Exception as e:
            logger.error("Error converting frame for %s: %s", self.rtsp_url, e)
            return None
