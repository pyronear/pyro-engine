# capture_once_per_camera.py

import json
import logging
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout

from PIL import Image

from reolink_api.src.camera.camera_rtsp import RTSPCamera  # your class

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s: %(message)s")
logger = logging.getLogger("CaptureOnce")

CAPTURE_TIMEOUT_S = 6  # per camera timeout in seconds


def load_credentials(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def save_image(img: Image.Image, camera_key: str, output_dir: Path) -> Path:
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    filename = f"{camera_key}_{ts}.jpg"
    out_path = output_dir / filename
    img.save(out_path, format="JPEG", quality=90)
    return out_path


def capture_with_timeout(cam: RTSPCamera, timeout_s: int) -> Image.Image | None:
    """
    Run cam.capture() in a worker thread and return within timeout.
    If the capture blocks longer than timeout, return None.
    """
    def _do_capture():
        # if your RTSPCamera supports skip_frames, you can pass it here
        try:
            return cam.capture()
        except Exception:
            return None

    # one off executor per call to avoid sharing stuck threads
    with ThreadPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(_do_capture)
        try:
            return fut.result(timeout=timeout_s)
        except FuturesTimeout:
            logger.warning("Capture timed out for camera %s", getattr(cam, "ip_address", "unknown"))
            return None
        except Exception as e:
            logger.error("Capture error for camera %s: %s", getattr(cam, "ip_address", "unknown"), e)
            return None


def main():
    creds_path = Path("data/credentials.json")
    output_dir = Path("captures")
    output_dir.mkdir(parents=True, exist_ok=True)

    if not creds_path.exists():
        logger.error("credentials file not found at %s", creds_path)
        return

    cameras = load_credentials(str(creds_path))

    for camera_key, info in cameras.items():
        rtsp_url = info.get("rtsp_url")
        ip_address = info.get("id") or camera_key
        cam_type = info.get("type", "rtsp")

        if not rtsp_url:
            logger.error("Missing rtsp_url for %s, skipping", camera_key)
            continue

        logger.info("Capturing from camera %s (%s)", camera_key, rtsp_url)

        cam = RTSPCamera(rtsp_url=rtsp_url, ip_address=str(ip_address), cam_type=cam_type)
        img = capture_with_timeout(cam, CAPTURE_TIMEOUT_S)

        if img is None:
            logger.error("FAILED to capture from %s", camera_key)
            continue

        try:
            out_path = save_image(img, camera_key, output_dir)
            logger.info("Saved capture for %s to %s", camera_key, out_path)
        except Exception as e:
            logger.error("Could not save image for %s: %s", camera_key, e)


if __name__ == "__main__":
    main()
