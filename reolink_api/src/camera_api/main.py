# capture_loop.py
import logging
import time
from typing import Dict

from camera_rtsp import RTSPCamera 
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

CAPTURE_INTERVAL = 30.0
PER_CAMERA_DELAY = 0.2
CAPTURE_TIMEOUT_S = 5.0
MAX_FAILS_BEFORE_SKIP = 1
SKIP_DURATION = 120.0

log = logging.getLogger("rtsp_loop")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    log.info("Starting RTSP sequential capture loop")
    log.info("Cameras: %s", ", ".join(CAMERAS.keys()))

    cameras = {name: RTSPCamera(url, ip_address=name, default_timeout_s=CAPTURE_TIMEOUT_S) for name, url in CAMERAS.items()}
    failure_count = {name: 0 for name in CAMERAS}
    skip_until = {name: 0.0 for name in CAMERAS}

    while True:
        cycle_start = time.time()
        log.info("New cycle")

        for name, cam in cameras.items():
            now = time.time()
            if now < skip_until[name]:
                left = skip_until[name] - now
                log.warning("[%s] skipped for %.0fs due to previous failure", name, left)
                continue

            t0 = time.time()
            log.info("[%s] capture attempt", name)
            img = cam.capture()  # uses ffmpeg with hard timeout internally
            dt = time.time() - t0

            if img is not None:
                w, h = img.size
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
