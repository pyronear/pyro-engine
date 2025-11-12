# capture_loop.py
import logging
import time

from camera.registry import CAMERA_REGISTRY  # uses your RAW_CONFIG and builds the objects

CAPTURE_INTERVAL = 30.0
PER_CAMERA_DELAY = 0.2
MAX_FAILS_BEFORE_SKIP = 1
SKIP_DURATION = 120.0

log = logging.getLogger("rtsp_loop")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    cameras = CAMERA_REGISTRY  # Dict[str, object with .capture()]
    names = list(cameras.keys())
    log.info("Starting sequential capture loop with registry")
    log.info("Cameras: %s", ", ".join(names))

    failure_count = {name: 0 for name in names}
    skip_until = {name: 0.0 for name in names}

    while True:
        cycle_start = time.time()
        log.info("New cycle")

        for name in names:
            cam = cameras[name]
            now = time.time()
            if now < skip_until[name]:
                left = skip_until[name] - now
                log.warning("[%s] skipped for %.0fs due to previous failure", name, left)
                continue

            t0 = time.time()
            log.info("[%s] capture attempt", name)

            img = None
            try:
                img = cam.capture()  # RTSPCamera uses ffmpeg with hard timeout internally
            except Exception as e:
                log.error("[%s] capture raised: %s", name, e)

            dt = time.time() - t0

            if img is not None:
                try:
                    w, h = img.size  # Pillow Image
                except Exception:
                    # fallback if a camera returns a NumPy array
                    try:
                        h, w = img.shape[:2]
                    except Exception:
                        w, h = -1, -1
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
