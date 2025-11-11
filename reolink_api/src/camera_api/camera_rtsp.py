# camera/rtsp_camera.py or similar

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
        cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            logger.error("Unable to open RTSP stream: %s", self.rtsp_url)
            cap.release()
            return None

        ok, frame = cap.read()
        cap.release()

        if not ok or frame is None:
            logger.error("Failed to read frame from %s", self.rtsp_url)
            return None

        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return Image.fromarray(rgb)
        except Exception as e:
            logger.error("Error converting frame for %s: %s", self.rtsp_url, e)
            return None
