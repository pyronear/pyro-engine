# capture_loop.py
import logging
import os
import subprocess
import tempfile
import time
from typing import Dict, Optional

import cv2
import numpy as np

# ask OpenCV to prefer TCP, the server may or may not honor stimeout
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
    "La Forestiere": r"rtsp://DDSIS\pyronear:4kX<x64K+Pr4@srvcamera:554/live/3F8CD700-DFEE-401A-8445-CB9CB0DF3DFF",
    "Sampzon": r"rtsp://DDSIS\pyronear:4kX<x64K+Pr4@srvcamera:554/live/4E10857C-107B-465E-99B3-8E8F0DBCB3E7",
    "Bidon": r"rtsp://DDSIS\pyronear:4kX<x64K+Pr4@srvcamera:554/live/14C4E0D6-E1D9-471D-802C-A903D91FE4C0",
}

CAPTURE_INTERVAL = 30.0   # seconds between two passes on all cameras
PER_CAMERA_DELAY = 0.2    # pause between cameras
CAPTURE_TIMEOUT_S = 5.0   # hard timeout per camera

# optional backoff to avoid hammering a bad camera
MAX_FAILS_BEFORE_SKIP = 1
SKIP_DURATION = 120.0     # seconds

log = logging.getLogger("rtsp_loop")


def grab_frame_ffmpeg(rtsp_url: str, timeout_s: float) -> Optional[np.ndarray]:
    """
    Grab one frame using ffmpeg with a hard process timeout.
    Writes to a temp JPEG then reads it with OpenCV.
    Returns BGR ndarray or None.
    """
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=True) as tmp:
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel", "error",
            "-rtsp_transport", "tcp",
            "-i", rtsp_url,
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
            log.error("ffmpeg timed out after %.1fs for %s", timeout_s, rtsp_url)
            return None

        frame = cv2.imread(tmp.name)
        if frame is None:
            log.error("ffmpeg returned no image for %s", rtsp_url)
            return None
        return frame


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    log.info("Starting RTSP sequential capture loop")
    log.info("Cameras: %s", ", ".join(CAMERAS.keys()))

    failure_count = {name: 0 for name in CAMERAS}
    skip_until = {name: 0.0 for name in CAMERAS}

    while True:
        cycle_start = time.time()
        log.info("New cycle")

        for name, url in CAMERAS.items():
            now = time.time()
            if now < skip_until[name]:
                left = skip_until[name] - now
                log.warning("[%s] skipped for %.0fs due to previous failure", name, left)
                continue

            t0 = time.time()
            log.info("[%s] capture attempt", name)

            frame = grab_frame_ffmpeg(url, CAPTURE_TIMEOUT_S)
            dt = time.time() - t0

            if frame is not None:
                h, w = frame.shape[:2]
                log.info("[%s] capture ok in %.2fs, size %dx%d", name, dt, w, h)
                failure_count[name] = 0
            else:
                failure_count[name] += 1
                log.error("[%s] timed out or failed in %.2fs", name, dt)
                if failure_count[name] >= MAX_FAILS_BEFORE_SKIP:
                    skip_until[name] = time.time() + SKIP_DURATION
                    log.error("[%s] will be skipped for %.0fs", name, SKIP_DURATION)

            time.sleep(PER_CAMERA_DELAY)

        elapsed = time.time() - cycle_start
        remaining = CAPTURE_INTERVAL - elapsed
        log.info("Cycle finished in %.2fs, sleep %.2fs", elapsed, max(0.0, remaining))
        if remaining > 0:
            time.sleep(remaining)


if __name__ == "__main__":
    main()
