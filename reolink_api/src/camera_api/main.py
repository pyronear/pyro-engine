# capture_loop.py
import logging
import os
import threading
import time
from typing import Dict, Optional

import cv2
import numpy as np

# Make OpenCV prefer TCP and use a shorter socket timeout, in case the server honors it
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|stimeout;5000000"

# -----------------------
# Camera configuration
# -----------------------
CAMERAS: Dict[str, str] = {
    "Serre de Gruas": r"rtsp://DDSIS\pyronear:4kX<x64K+Pr4@srvcamera:554/live/E4CF7F9D-F85F-4ED6-AB56-E275181DD3EB",
    "Blandine": r"rtsp://DDSIS\pyronear:4kX<x64K+Pr4@srvcamera:554/live/1ECAC3E9-DB72-4CF3-8BD5-E55F4491356A",
    "Aubignas": r"rtsp://DDSIS\pyronear:4kX<x64K+Pr4@srvcamera:554/live/D2E6EC5F-5511-420B-A264-5B1447C6FF6F",
    "Pieds de Boeufs": r"rtsp://DDSIS\pyronear:4kX<x64K+Pr4@srvcamera:554/live/D4C8694C-964C-43BD-BD57-563E0E43C751",
    "Saint Jean Chambre": r"rtsp://DDSIS\pyronear:4kX<x64K+Pr4@srvcamera:554/live/6641704A-0873-40FE-82AE-22EC03AA4AA9",
    "Bidon": r"rtsp://DDSIS\pyronear:4kX<x64K+Pr4@srvcamera:554/live/14C4E0D6-E1D9-471D-802C-A903D91FE4C0",
    "La Forestiere": r"rtsp://DDSIS\pyronear:4kX<x64K+Pr4@srvcamera:554/live/3F8CD700-DFEE-401A-8445-CB9CB0DF3DFF",
    "Sampzon": r"rtsp://DDSIS\pyronear:4kX<x64K+Pr4@srvcamera:554/live/4E10857C-107B-465E-99B3-8E8F0DBCB3E7",
}

CAPTURE_INTERVAL = 30.0   # seconds between two passes on all cameras
PER_CAMERA_DELAY = 0.2    # small pause between cameras
CAPTURE_TIMEOUT_S = 5.0   # hard timeout per camera

log = logging.getLogger("rtsp_loop")


def _read_once(rtsp_url: str) -> Optional[np.ndarray]:
    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        cap.release()
        return None
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        return None
    return frame


def capture_with_timeout(rtsp_url: str, timeout_s: float) -> Optional[np.ndarray]:
    """Run a single VideoCapture read with a hard timeout."""
    holder = {"frame": None}

    def worker():
        try:
            holder["frame"] = _read_once(rtsp_url)
        except Exception:
            holder["frame"] = None

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    t.join(timeout_s)
    if t.is_alive():
        return None
    return holder["frame"]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    log.info("Starting RTSP sequential capture loop")
    log.info("Cameras: %s", ", ".join(CAMERAS.keys()))

    while True:
        cycle_start = time.time()
        log.info("New cycle")

        for name, url in CAMERAS.items():
            t0 = time.time()
            log.info("[%s] capture attempt", name)

            frame = capture_with_timeout(url, CAPTURE_TIMEOUT_S)
            dt = time.time() - t0

            if frame is not None:
                h, w = frame.shape[:2]
                log.info("[%s] capture ok in %.2fs, size %dx%d", name, dt, w, h)
            else:
                log.error("[%s] timed out or failed in %.2fs", name, dt)

            time.sleep(PER_CAMERA_DELAY)

        elapsed = time.time() - cycle_start
        remaining = CAPTURE_INTERVAL - elapsed
        log.info("Cycle finished in %.2fs, sleep %.2fs", elapsed, max(0.0, remaining))
        if remaining > 0:
            time.sleep(remaining)


if __name__ == "__main__":
    main()
