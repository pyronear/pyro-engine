import logging
import os
from typing import Optional

import cv2
import numpy as np


# Use TCP and a shorter timeout for RTSP inside OpenCV
# stimeout is in microseconds
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|stimeout;5000000"

logger = logging.getLogger(__name__)


CAMERAS_CONFIG = {
    "Serre de Gruas": {
        "rtsp_url": "rtsp://DDSIS\\pyronear:4kX<x64K+Pr4@srvcamera:554/live/E4CF7F9D-F85F-4ED6-AB56-E275181DD3EB",
    },
    "Blandine": {
        "rtsp_url": "rtsp://DDSIS\\pyronear:4kX<x64K+Pr4@srvcamera:554/live/1ECAC3E9-DB72-4CF3-8BD5-E55F4491356A",
    },
    "Aubignas": {
        "rtsp_url": "rtsp://DDSIS\\pyronear:4kX<x64K+Pr4@srvcamera:554/live/D2E6EC5F-5511-420B-A264-5B1447C6FF6F",
    },
    "Pieds de Boeufs": {
        "rtsp_url": "rtsp://DDSIS\\pyronear:4kX<x64K+Pr4@srvcamera:554/live/D4C8694C-964C-43BD-BD57-563E0E43C751",
    },
    "Saint Jean Chambre": {
        "rtsp_url": "rtsp://DDSIS\\pyronear:4kX<x64K+Pr4@srvcamera:554/live/6641704A-0873-40FE-82AE-22EC03AA4AA9",
    },
    "Bidon": {
        "rtsp_url": "rtsp://DDSIS\\pyronear:4kX<x64K+Pr4@srvcamera:554/live/14C4E0D6-E1D9-471D-802C-A903D91FE4C0",
    },
    "La Forestiere": {
        "rtsp_url": "rtsp://DDSIS\\pyronear:4kX<x64K+Pr4@srvcamera:554/live/3F8CD700-DFEE-401A-8445-CB9CB0DF3DFF",
    },
    "Sampzon": {
        "rtsp_url": "rtsp://DDSIS\\pyronear:4kX<x64K+Pr4@srvcamera:554/live/4E10857C-107B-465E-99B3-8E8F0DBCB3E7",
    },
}


class RTSPCamera:
    """Simple RTSP camera wrapper around cv2.VideoCapture."""

    def __init__(self, name: str, rtsp_url: str) -> None:
        self.name = name
        self.rtsp_url = rtsp_url

    def capture(self) -> Optional[np.ndarray]:
        """Open the RTSP stream, read a single frame as BGR array, then close."""
        cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            logger.error("[%s] Unable to open RTSP stream", self.name)
            cap.release()
            return None

        ok, frame = cap.read()
        cap.release()

        if not ok or frame is None:
            logger.error("[%s] Failed to read frame", self.name)
            return None

        return frame
