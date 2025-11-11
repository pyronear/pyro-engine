import logging
import time

from camera_rtsp import CAMERAS, grab_frame_with_ffmpeg_to_file


CAPTURE_INTERVAL = 30.0  # seconds between two passes on all cameras
PER_CAMERA_DELAY = 0.5   # short pause between cameras
FFMPEG_TIMEOUT_MS = 5000


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

            frame = grab_frame_with_ffmpeg_to_file(url, timeout_ms=FFMPEG_TIMEOUT_MS)
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
