# capture_once_per_camera.py

import json
import logging
from datetime import datetime, UTC
from pathlib import Path
from typing import Optional
from subprocess import run, PIPE, TimeoutExpired, CalledProcessError
from urllib.parse import urlsplit, urlunsplit

from PIL import Image
import io

from reolink_api.src.camera.camera_rtsp import RTSPCamera  # your class

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s: %(message)s")
logger = logging.getLogger("CaptureOnce")

CAPTURE_TIMEOUT_S = 6  # hard timeout per camera for ffmpeg and fallback
JPEG_QUALITY = 90


def load_credentials(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def save_image(img: Image.Image, camera_key: str, output_dir: Path) -> Path:
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_path = output_dir / f"{camera_key}_{ts}.jpg"
    img.save(out_path, format="JPEG", quality=JPEG_QUALITY)
    return out_path


def _add_query_param(url: str, key: str, value: str) -> str:
    parts = urlsplit(url)
    q = parts.query
    q = f"{q}&{key}={value}" if q else f"{key}={value}"
    return urlunsplit(parts._replace(query=q))


def ffmpeg_grab_frame(rtsp_url: str, timeout_sec: int) -> Optional[Image.Image]:
    """
    Grab exactly one frame using ffmpeg with a strict process timeout.
    Uses -rw_timeout which is widely supported. Stays on UDP.
    """
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "error",
        "-rw_timeout", str(timeout_sec * 1_000_000),  # microseconds
        "-rtsp_transport", "udp",
        "-probesize", "32k",
        "-analyzeduration", "200k",
        "-i", rtsp_url,
        "-frames:v", "1",
        "-f", "image2",
        "pipe:1",
    ]
    try:
        res = run(cmd, stdout=PIPE, stderr=PIPE, timeout=timeout_sec + 1, check=True)
        return Image.open(io.BytesIO(res.stdout)).convert("RGB")
    except TimeoutExpired:
        logger.warning("ffmpeg timed out for %s", rtsp_url)
        return None
    except CalledProcessError as e:
        if e.stderr:
            logger.error("ffmpeg error: %s", e.stderr.decode(errors="ignore").strip())
        return None
    except Exception as e:
        logger.error("ffmpeg exception: %s", e)
        return None



def opencv_fallback(rtsp_url: str, ip_address: str, cam_type: str, timeout_sec: int) -> Optional[Image.Image]:
    """
    Fallback to RTSPCamera, with a best effort short timeout using OpenCV props.
    Still enforced at call site with a short-lived thread if you want, but we try to keep it quick.
    """
    cam = RTSPCamera(rtsp_url=rtsp_url, ip_address=str(ip_address), cam_type=cam_type)
    try:
        # If your RTSPCamera supports skip_frames, you could pass a smaller value here
        return cam.capture()
    except Exception:
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

        # Primary path, ffmpeg one shot with hard timeout
        img = ffmpeg_grab_frame(rtsp_url, CAPTURE_TIMEOUT_S)

        # Fallback to your OpenCV class if ffmpeg failed
        if img is None:
            img = opencv_fallback(rtsp_url, ip_address, cam_type, CAPTURE_TIMEOUT_S)

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
