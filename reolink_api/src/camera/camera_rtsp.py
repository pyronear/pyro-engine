# Copyright (C) 2022-2025, Pyronear.
# Licensed under the Apache License 2.0
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

import logging
from typing import Optional

import cv2
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

    def capture(self) -> Optional[Image.Image]:
        """Open the RTSP stream and read a single frame, returning it as a Pillow Image."""
        cap = cv2.VideoCapture(self.rtsp_url)
        if not cap.isOpened():
            logger.error("Unable to open RTSP stream: %s", self.rtsp_url)
            cap.release()
            return None

        ret, frame = cap.read()
        cap.release()

        if not ret or frame is None:
            logger.error("Failed to read frame from %s", self.rtsp_url)
            return None

        try:
            return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        except Exception as e:
            logger.error("Error converting frame for %s: %s", self.rtsp_url, e)
            return None
