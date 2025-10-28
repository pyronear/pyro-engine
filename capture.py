# capture_once_per_camera.py

import json
import logging
from pathlib import Path
from datetime import datetime

from PIL import Image
from reolink_api.src.camera.camera_rtsp import RTSPCamera  # assumes the class below lives in rtsp_camera.py

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s: %(message)s")
logger = logging.getLogger("CaptureOnce")


def load_credentials(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def save_image(img: Image.Image, camera_key: str, output_dir: Path) -> Path:
    # timestamped filename to avoid cache and confirm freshness
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    filename = f"{camera_key}_{ts}.jpg"
    out_path = output_dir / filename
    img.save(out_path, format="JPEG", quality=90)
    return out_path


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
        ip_address = info.get("id")  # just something identifiable for logs
        cam_type = info.get("type", "rtsp")

        logger.info("Capturing from camera %s (%s)", camera_key, rtsp_url)

        cam = RTSPCamera(rtsp_url=rtsp_url, ip_address=str(ip_address), cam_type=cam_type)
        img = cam.capture()

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

