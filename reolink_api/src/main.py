# simple_static_loop.py

import logging
import time

from camera.registry import CAMERA_REGISTRY

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s: %(message)s")

CAPTURE_INTERVAL = 30.0   # seconds between two captures for the same camera
PER_CAMERA_DELAY = 0.1    # small pause between cameras

def main():
    logging.info("Starting simple sequential capture loop")

    # one settle_until per camera, to avoid saving right after a reconnect
    settle_until = {ip: 0.0 for ip in CAMERA_REGISTRY}

    while True:
        cycle_start = time.time()

        for ip, cam in CAMERA_REGISTRY.items():
            try:
                image = cam.capture()
                now = time.time()

                # if cam has re opened recently, wait a bit before saving
                opened_at = getattr(cam, "_opened_at", 0)
                if opened_at:
                    settle_until[ip] = opened_at + 1.0

                if image and now >= settle_until[ip]:
                    print("image size", ip, image.size)
                    cam.last_images[-1] = image
                    logging.info(f"[{ip}] Updated static image")
            except Exception as e:
                logging.error(f"[{ip}] Error capturing static image: {e}")

            # short delay between cameras
            time.sleep(PER_CAMERA_DELAY)

        # keep roughly CAPTURE_INTERVAL seconds between two passes
        elapsed = time.time() - cycle_start
        remaining = CAPTURE_INTERVAL - elapsed
        if remaining > 0:
            time.sleep(remaining)


if __name__ == "__main__":
    main()
