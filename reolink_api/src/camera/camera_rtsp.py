# Copyright (C) 2022-2025, Pyronear.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

import logging
import time
from typing import Optional

import cv2
from PIL import Image

logger = logging.getLogger("RTSPCamera")
logger.setLevel(logging.INFO)


class RTSPCamera:
    """
    Camera that exposes an RTSP stream.
    capture() grabs one frame via OpenCV and returns it as a Pillow Image.
    This version is thread safe and does not rely on signal.alarm.
    """

    def __init__(self, rtsp_url: str, ip_address: str = "", cam_type: str = "rtsp"):
        self.rtsp_url = rtsp_url
        self.ip_address = ip_address
        self.cam_type = cam_type

    def capture(self, timeout: int = 10) -> Optional[Image.Image]:
        """
        Try to open the RTSP stream and read a frame within `timeout` seconds.
        Returns a Pillow Image (RGB) or None.
        Safe to call from worker threads.
        """
        start_time = time.perf_counter()

        cap = cv2.VideoCapture(self.rtsp_url)
        if not cap.isOpened():
            logger.error("Unable to open RTSP stream: %s", self.rtsp_url)
            cap.release()
            return None

        frame = None
        while True:
            # Check timeout
            if (time.perf_counter() - start_time) > timeout:
                logger.error("RTSP capture timed out for %s", self.rtsp_url)
                break

            ret, frm = cap.read()
            if ret and frm is not None:
                frame = frm
                break
            # tiny sleep could be added if you want to avoid busy loop
            # but usually cap.read() already blocks briefly

        cap.release()

        if frame is None:
            return None

        try:
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            return img
        except Exception as e:
            logger.error("RTSP capture error converting frame for %s: %s", self.rtsp_url, e)
            return None
