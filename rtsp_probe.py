import time
import signal
import logging
import json
from typing import Optional, Dict, Any
from pathlib import Path

import cv2
from PIL import Image

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)


class RTSPCamera:
    """A class for interacting with cameras via RTSP."""

    def __init__(self, rtsp_url: str, ip_address: str = "", cam_type: str = "rtsp"):
        self.rtsp_url = rtsp_url
        self.ip_address = ip_address
        self.cam_type = cam_type

    def _timeout_handler(self, signum, frame):
        raise TimeoutError("Capture operation timed out.")

    def capture(self, timeout: int = 10) -> Optional[Image.Image]:
        """Captures an image from the camera and returns it as a PIL Image."""
        signal.signal(signal.SIGALRM, self._timeout_handler)
        signal.alarm(timeout)

        cap = None
        try:
            cap = cv2.VideoCapture(self.rtsp_url)

            if not cap.isOpened():
                logging.error("Unable to open RTSP stream.")
                return None

            ret, frame = cap.read()
            if not ret or frame is None:
                logging.error("Unable to read frame from RTSP stream.")
                return None

            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            return image

        except TimeoutError:
            logging.error("Capture operation timed out.")
            return None

        except Exception as e:
            logging.error(f"An error occurred: {e}")
            return None

        finally:
            if cap is not None:
                cap.release()
            signal.alarm(0)


def load_cameras_from_json(filepath: str) -> Dict[str, Dict[str, Any]]:
    """Loads camera configurations from a JSON file."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Camera JSON file not found: {filepath}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    logging.info(f"Loaded {len(data)} cameras from {filepath}")
    return data


def test_all_cameras(cameras: Dict[str, Dict[str, Any]], timeout: int = 10):
    """Capture one frame per camera and report results."""
    results = {}

    for cam_name, cfg in cameras.items():
        rtsp_url = cfg.get("rtsp_url")
        cam_type = cfg.get("type", "rtsp")

        logging.info(f"Capturing from {cam_name} ...")
        cam = RTSPCamera(rtsp_url=rtsp_url, cam_type=cam_type)

        start_time = time.perf_counter()
        img = cam.capture(timeout=timeout)
        end_time = time.perf_counter()
        elapsed_s = end_time - start_time

        if img is None:
            logging.error(f"{cam_name}: capture FAILED after {elapsed_s:.2f}s")
            results[cam_name] = {"success": False, "elapsed_s": elapsed_s}
        else:
            logging.info(f"{cam_name}: OK in {elapsed_s:.2f}s, size={img.size}")
            results[cam_name] = {"success": True, "elapsed_s": elapsed_s, "size": img.size}

    print("\n=== Capture summary ===")
    for name, r in results.items():
        if r["success"]:
            print(f"{name}: ✅ OK, {r['elapsed_s']:.2f}s, {r['size'][0]}x{r['size'][1]}")
        else:
            print(f"{name}: ❌ FAIL, {r['elapsed_s']:.2f}s")


if __name__ == "__main__":
    cameras = load_cameras_from_json("../cameras.json")
    test_all_cameras(cameras, timeout=10)
