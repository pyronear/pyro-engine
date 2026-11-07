# Copyright (C) 2022-2026, Pyronear.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

"""Log PTZ pose misalignment over time, without changing engine behavior.

Reads the frame the camera API already cached for each PTZ pose (no camera command),
registers it against a fixed per-pose reference and against the previous frame, and
appends the offset to a CSV. Images are kept only for large offsets or failed
registrations, up to a cap.

Measurement only (pyronear/vision-rd#89). On the Pi:

    cp src/measure_pose_drift.py data/
    docker compose exec -d engine python data/measure_pose_drift.py

Outputs in data/pose_drift/: drift.csv, ref/<cam_id>.jpg (delete it and restart to reset a reference),
anomalies/<cam_id>_<ts>.jpg. The process stops when the engine container restarts.
"""

import argparse
import csv
import hashlib
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
from PIL import Image
from pyro_camera_api_client.client import PyroCameraAPIClient

from pyroengine.core import is_day_time
from pyroengine.logs import setup_logging

logger = logging.getLogger("pose_drift")

CROP_TOP = 0.45  # keep the terrain, drop most of the sky
FIELDS = ["ts", "cam_id", "dx", "dy", "cc", "prev_dx", "prev_dy", "prev_cc"]


def gradient(frame: Image.Image) -> np.ndarray:
    gray = np.asarray(frame.convert("L"), dtype=np.float32) / 255.0
    gray = gray[int(gray.shape[0] * CROP_TOP) :, :]
    mag = cv2.magnitude(cv2.Sobel(gray, cv2.CV_32F, 1, 0), cv2.Sobel(gray, cv2.CV_32F, 0, 1))
    peak = float(mag.max())
    return mag / peak if peak > 0 else mag


def register(ref: np.ndarray, img: np.ndarray) -> Optional[Tuple[float, float, float]]:
    """Return (dx, dy, ecc_correlation): scene shift in px of img vs ref, or None if ECC fails."""
    if ref.shape != img.shape:
        return None
    window = cv2.createHanningWindow(ref.shape[::-1], cv2.CV_32F)
    # phaseCorrelate windows its inputs in place when sizes are FFT-optimal: keep the cached arrays intact.
    (sx, sy), _ = cv2.phaseCorrelate(ref.copy(), img.copy(), window)
    warp = np.array([[1, 0, sx], [0, 1, sy]], dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 300, 1e-6)
    try:
        cc, warp = cv2.findTransformECC(ref, img, warp, cv2.MOTION_TRANSLATION, criteria, None, 5)
    except cv2.error:
        return None
    return float(warp[0, 2]), float(warp[1, 2]), float(cc)


class DriftLogger:
    def __init__(self, out_dir: Path, max_px: float, min_cc: float, max_saved: int) -> None:
        self.ref_dir = out_dir / "ref"
        self.anomaly_dir = out_dir / "anomalies"
        self.ref_dir.mkdir(parents=True, exist_ok=True)
        self.anomaly_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = out_dir / "drift.csv"
        if not self.csv_path.exists():
            with self.csv_path.open("w", newline="") as f:
                csv.writer(f).writerow(FIELDS)
        self.max_px, self.min_cc, self.max_saved = max_px, min_cc, max_saved
        self.saved = len(list(self.anomaly_dir.glob("*.jpg")))
        self.refs: dict = {}
        self.prev: dict = {}
        self.last_hash: dict = {}

    def process(self, cam_id: str, frame: Image.Image) -> None:
        digest = hashlib.sha256(frame.tobytes()).hexdigest()
        if self.last_hash.get(cam_id) == digest:
            return
        self.last_hash[cam_id] = digest
        if not is_day_time(None, frame, "ir"):
            return

        img = gradient(frame)
        ref_path = self.ref_dir / f"{cam_id}.jpg"
        if cam_id not in self.refs:
            if not ref_path.exists():
                frame.save(ref_path, quality=95)
                logger.info("[%s] Reference saved to %s", cam_id, ref_path)
            self.refs[cam_id] = gradient(Image.open(ref_path))

        res = register(self.refs[cam_id], img)
        prev = register(self.prev[cam_id], img) if cam_id in self.prev else None
        self.prev[cam_id] = img

        ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
        with self.csv_path.open("a", newline="") as f:
            csv.writer(f).writerow([ts, cam_id, *_fmt(res), *_fmt(prev)])

        large = res is None or np.hypot(res[0], res[1]) > self.max_px or res[2] < self.min_cc
        if large and self.saved < self.max_saved:
            frame.save(self.anomaly_dir / f"{cam_id}_{ts.replace(':', '-')}.jpg", quality=90)
            self.saved += 1


def _fmt(res: Optional[Tuple[float, float, float]]) -> list:
    return [f"{v:.3f}" for v in res] if res is not None else ["", "", ""]


def main(args: argparse.Namespace) -> None:
    setup_logging()
    creds = json.loads(Path(args.creds).read_text())
    poses = [(ip, p) for ip, cam in creds.items() if cam.get("type") == "ptz" for p in cam.get("poses", [])]
    client = PyroCameraAPIClient(args.pyro_camera_api_url)
    drift = DriftLogger(Path(args.out_dir), args.max_px, args.min_cc, args.max_saved)
    logger.info("Measuring drift on %d poses every %ds into %s", len(poses), args.period, args.out_dir)

    while True:
        for ip, pose in poses:
            cam_id = f"{ip}_{pose}"
            try:
                frame = client.get_latest_image(ip, pose)
                if frame is not None:
                    drift.process(cam_id, frame)
            except Exception:
                logger.exception("[%s] Drift measurement failed", cam_id)
        time.sleep(args.period)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Log PTZ pose misalignment over time")
    parser.add_argument("--pyro_camera_api_url", type=str, default="http://127.0.0.1:8081", help="Camera api url")
    parser.add_argument("--creds", type=str, default="data/credentials.json", help="Camera credentials")
    parser.add_argument("--out_dir", type=str, default="data/pose_drift", help="Output folder")
    parser.add_argument("--period", type=int, default=30, help="Seconds between two passes")
    parser.add_argument("--max_px", type=float, default=5.0, help="Keep the frame above this offset (px)")
    parser.add_argument("--min_cc", type=float, default=0.5, help="Keep the frame below this ECC correlation")
    parser.add_argument("--max_saved", type=int, default=500, help="Maximum number of kept frames")
    main(parser.parse_args())
