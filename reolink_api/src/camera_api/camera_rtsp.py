import logging
import subprocess
import tempfile
from typing import Optional

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger("RTSPCamera")
logger.setLevel(logging.INFO)


def _safe_url(u: str) -> str:
    try:
        if "://" in u and "@" in u:
            scheme, rest = u.split("://", 1)
            after_at = rest.split("@", 1)[1]
            return f"{scheme}://***:***@{after_at}"
    except Exception:
        pass
    return u


class RTSPCamera:
    """RTSP camera that grabs one frame via ffmpeg with a hard timeout, returns a Pillow Image."""

    def __init__(self, rtsp_url: str, ip_address: str = "", cam_type: str = "rtsp", default_timeout_s: float = 5.0):
        self.rtsp_url = rtsp_url
        self.ip_address = ip_address
        self.cam_type = cam_type
        self.default_timeout_s = default_timeout_s
        self.last_images: dict[int, Image.Image] = {}

    def _grab_frame_ffmpeg(self, timeout_s: float) -> Optional[np.ndarray]:
        """
        Call ffmpeg to grab a single JPEG frame with a hard process timeout.
        Read it back with OpenCV and return a BGR ndarray.
        """
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=True) as tmp:
            cmd = [
                "ffmpeg",
                "-hide_banner",
                "-loglevel", "error",
                "-rtsp_transport", "tcp",
                "-i", self.rtsp_url,
                "-frames:v", "1",
                "-y", tmp.name,
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

    def capture(self, timeout_s: Optional[float] = None) -> Optional[Image.Image]:
        """Grab one frame through ffmpeg, convert to RGB, return as Pillow Image."""
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
