import logging
import subprocess
import tempfile
from typing import Optional

import cv2
import numpy as np


logger = logging.getLogger(__name__)


CAMERAS = {
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


def grab_frame_with_ffmpeg_to_file(rtsp_url: str, timeout_ms: int = 5000) -> Optional[np.ndarray]:
    """
    Grab a single frame from the RTSP stream using ffmpeg,
    write it to a temporary JPEG file, then load it as a BGR NumPy array.
    """
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=True) as tmp:
        cmd = [
            "ffmpeg",
            "-rtsp_transport",
            "tcp",
            "-stimeout",
            str(timeout_ms * 1000),  # microseconds
            "-i",
            rtsp_url,
            "-frames:v",
            "1",
            "-y",
            tmp.name,
        ]

        try:
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
        except Exception as e:
            logger.error("ffmpeg failed to start for %s: %s", rtsp_url, e)
            return None

        if result.returncode != 0:
            # log a bit of stderr to understand what ffmpeg complains about
            err = result.stderr.decode(errors="ignore")
            logger.error(
                "ffmpeg error for %s, return code %s, stderr: %s",
                rtsp_url,
                result.returncode,
                err[:300],
            )
            return None

        # load the written JPEG
        frame = cv2.imread(tmp.name)
        if frame is None:
            logger.error("cv2.imread failed for file created from %s", rtsp_url)
            return None

        return frame
