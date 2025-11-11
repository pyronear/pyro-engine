import logging
import subprocess
import time
from typing import Optional

import cv2
import numpy as np


# --------------------------------------------------------------------
# Camera configuration
# --------------------------------------------------------------------

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


# --------------------------------------------------------------------
# Settings
# --------------------------------------------------------------------

CAPTURE_INTERVAL = 30.0  # seconds between two passes on all cameras
PER_CAMERA_DELAY = 0.5   # short pause between cameras
FFMPEG_TIMEOUT_MS = 5000


# --------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------

def grab_frame_with_ffmpeg(rtsp_url: str, timeout_ms: int) -> Optional[np.ndarray]:
    """Grab a single frame from the RTSP stream using ffmpeg, return BGR frame or None."""
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
            stderr=subprocess.DEVNULL,
            check=False,
        )
    except Exception as e:
        logging.error("ffmpeg failed to start for %s: %s", rtsp_url, e)
        return None

    if not result.stdout:
        logging.error("ffmpeg returned no data for %s", rtsp_url)
        return None

    arr = np.frombuffer(result.stdout, np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        logging.error("cv2.imdecode failed for %s", rtsp_url)
    return frame


# --------------------------------------------------------------------
# Main loop
# --------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    logging.info("Starting RTSP health check loop")
    logging.info("Cameras: %s", ", ".join(CAMERAS.keys()))

    while True:
        cycle_start = time.time()
        logging.info("New capture cycle started")

        for name, cfg in CAMERAS.items():
            url = cfg["rtsp_url"]
            t0 = time.time()
            logging.info("[%s] capture attempt", name)

            frame = grab_frame_with_ffmpeg(url, timeout_ms=FFMPEG_TIMEOUT_MS)
            dt = time.time() - t0

            if frame is not None:
                h, w = frame.shape[:2]
                logging.info(
                    "[%s] capture ok in %.2fs, size %dx%d",
                    name,
                    dt,
                    w,
                    h,
                )
            else:
                logging.error("[%s] capture failed in %.2fs", name, dt)

            time.sleep(PER_CAMERA_DELAY)

        elapsed = time.time() - cycle_start
        remaining = CAPTURE_INTERVAL - elapsed
        logging.info(
            "Cycle finished in %.2fs, sleeping for %.2fs",
            elapsed,
            max(0.0, remaining),
        )

        if remaining > 0:
            time.sleep(remaining)


if __name__ == "__main__":
    main()
