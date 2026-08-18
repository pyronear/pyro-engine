# Copyright (C) 2022-2026, Pyronear.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

"""
FOV-per-zoom calibration via QR code, for cameras with hardware azimuth (Linovision).

Runs remotely against the camera API of a deployed station (over VPN), so the
camera stays on its mast. Someone on site only needs to fix a printed QR code
(~20-30 cm, any content) in the camera's field of view, roughly centered,
at any distance such that it stays inside the frame at max zoom (~10-20 m).

Method, two steps:
1. Anchor (absolute FOV at min zoom): capture the QR position, pan by a small
   angle, capture again. The camera's hardware azimuth gives the exact angle
   actually traveled, and the QR pixel shift gives the focal length in pixels:
       f_px = Δpx / tan(Δaz)   →   h_fov = 2·atan(W / 2f_px)
   No QR size, no distance, no timing needed.
2. Chained ratios: at each zoom level, the QR apparent size scales with focal
   length, so tan(fov(z)/2) = tan(fov(anchor)/2) · qr_px(anchor) / qr_px(z).

The report compares the measured table with the optical model
fov(Z) = 2·atan(tan(fov0/2)/Z) to check whether zoom_raw is the true optical ratio.

Usage:
    uv run --with opencv-python-headless,numpy,requests \\
        tools/fov_qr_calibration.py --api http://<station>:8082 --ip 192.168.1.11
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import requests

DETECTOR = cv2.QRCodeDetector()


class Api:
    def __init__(self, base: str, ip: str):
        self.base = base.rstrip("/")
        self.ip = ip

    def _req(self, method: str, path: str, **params) -> requests.Response:
        resp = requests.request(method, self.base + path, params={"camera_ip": self.ip, **params}, timeout=60)
        resp.raise_for_status()
        return resp

    def capture(self) -> np.ndarray:
        resp = self._req("GET", "/cameras/capture", anonymize="false")
        img = cv2.imdecode(np.frombuffer(resp.content, np.uint8), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise RuntimeError("capture: could not decode JPEG")
        return img

    def azimuth(self) -> float:
        return float(self._req("GET", "/control/azimuth").json()["azimuth_deg"])

    def zoom(self, level: int) -> None:
        requests.post(f"{self.base}/control/zoom/{self.ip}/{level}", timeout=120).raise_for_status()

    def pan(self, direction: str, degrees: float) -> None:
        self._req("POST", "/control/move_by_degrees", direction=direction, degrees=degrees, speed=10)

    def stop_patrol(self) -> bool:
        """Signal patrol stop. Returns True if a patrol was running."""
        resp = requests.post(f"{self.base}/patrol/stop_patrol", params={"camera_ip": self.ip}, timeout=30)
        if resp.status_code == 404:  # no patrol running
            return False
        resp.raise_for_status()
        return True

    def start_patrol(self) -> None:
        requests.post(f"{self.base}/patrol/start_patrol", params={"camera_ip": self.ip}, timeout=30).raise_for_status()


def detect_qr(img: np.ndarray) -> Optional[np.ndarray]:
    """Return the 4 QR corner points (4x2 float array), or None."""
    ok, points = DETECTOR.detect(img)
    if not ok or points is None:
        return None
    return points.reshape(-1, 2)


def qr_size_px(corners: np.ndarray) -> float:
    """Mean of the 4 side lengths."""
    return float(np.mean([np.linalg.norm(corners[i] - corners[(i + 1) % 4]) for i in range(4)]))


def capture_qr(api: Api, retries: int = 3) -> Optional[tuple[np.ndarray, np.ndarray]]:
    """Capture and detect; returns (image, corners) or None."""
    for _ in range(retries):
        img = api.capture()
        corners = detect_qr(img)
        if corners is not None:
            return img, corners
        time.sleep(1)
    return None


def signed_delta(az_from: float, az_to: float) -> float:
    return (az_to - az_from + 180.0) % 360.0 - 180.0


# The API zoom endpoint takes Reolink-style levels (0-64); the linovision
# adapter maps them linearly to the optical ratio 1-25.
def level_for_ratio(ratio: float) -> int:
    return max(0, min(64, round((ratio - 1.0) * 64.0 / 24.0)))


def ratio_for_level(level: int) -> float:
    return 1.0 + level * 24.0 / 64.0


def focal_from_pan(x1: float, x2: float, d_az_deg: float) -> float:
    """Solve atan(x1/f) - atan(x2/f) = d_az for the focal length f.

    x1/x2 are QR center offsets from the image center (px, positive right),
    before/after a pan of d_az degrees (positive = camera panned right).
    Exact off-axis solution of tan(d_az) = (x1-x2)·f / (f² + x1·x2), which the
    naive Δpx/tan(Δaz) only approximates when the QR is not centered.
    """
    if d_az_deg < 0:  # mirror to the pan-right case
        x1, x2, d_az_deg = -x1, -x2, -d_az_deg
    t = math.tan(math.radians(d_az_deg))
    dx = x1 - x2
    disc = dx * dx - 4 * t * t * x1 * x2
    if disc <= 0 or dx <= 0:
        return abs(dx) / t  # degenerate geometry: small-angle fallback
    return (dx + math.sqrt(disc)) / (2 * t)


def measure_anchor(api: Api, pan_deg: float, repeats: int) -> dict:
    """Absolute h_fov at current zoom via pan + hardware azimuth + QR center shift."""
    got = capture_qr(api)
    if got is None:
        raise SystemExit("QR not detected at anchor zoom. Move it closer to image center or use a bigger one.")
    h, w = got[0].shape

    f_px_samples = []
    for rep in range(repeats):
        got = capture_qr(api)
        if got is None:
            raise SystemExit("QR not detected at anchor zoom. Move it closer to image center or use a bigger one.")
        _, corners = got
        cx = float(corners[:, 0].mean())
        az0 = api.azimuth()
        # Pan toward the QR side: the scene shifts the opposite way, so the QR
        # moves toward the image center instead of out of frame.
        direction = "Left" if cx < w / 2 else "Right"
        api.pan(direction, pan_deg)
        time.sleep(1)
        az1 = api.azimuth()
        d_az = signed_delta(az0, az1)
        got = capture_qr(api)
        if got is None:
            raise SystemExit("QR lost after anchor pan. Reduce --anchor-pan.")
        _, corners_b = got
        if abs(d_az) < 1.0:
            raise SystemExit(f"Anchor pan only moved {d_az:.2f}° (hardware). Increase --anchor-pan.")
        x1 = cx - w / 2
        x2 = float(corners_b[:, 0].mean()) - w / 2
        f_px = focal_from_pan(x1, x2, d_az)
        f_px_samples.append(f_px)
        print(f"  anchor rep {rep + 1}: Δaz={d_az:+.2f}° x {x1:+.1f}px → {x2:+.1f}px → f_px={f_px:.1f}")
        # Pan back for the next repetition.
        api.pan("Left" if direction == "Right" else "Right", pan_deg)
        time.sleep(1)
    f_px = float(np.median(f_px_samples))
    return {
        "f_px": f_px,
        "width_px": w,
        "height_px": h,
        "h_fov": math.degrees(2 * math.atan(w / (2 * f_px))),
        "v_fov": math.degrees(2 * math.atan(h / (2 * f_px))),
        "samples": f_px_samples,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--api", required=True, help="Camera API base URL, e.g. http://192.168.255.58:8082")
    parser.add_argument("--ip", required=True, help="Camera IP as known by the API, e.g. 192.168.1.11")
    parser.add_argument("--zoom-min", type=int, default=1, help="Lowest optical zoom ratio to measure")
    parser.add_argument("--zoom-max", type=int, default=25, help="Highest optical zoom ratio to measure")
    parser.add_argument("--anchor-pan", type=float, default=8.0, help="Anchor pan command in degrees")
    parser.add_argument("--anchor-repeats", type=int, default=3)
    parser.add_argument("--settle", type=float, default=4.0, help="Seconds to wait after a zoom change")
    parser.add_argument(
        "--stop-wait",
        type=float,
        default=20.0,
        help="Seconds to wait after stopping the patrol (the worker may still finish an in-flight preset move)",
    )
    parser.add_argument("--out", default="fov_linovision.json")
    parser.add_argument("--no-restart-patrol", action="store_true")
    args = parser.parse_args()

    api = Api(args.api, args.ip)
    print("Stopping patrol...")
    patrol_was_running = api.stop_patrol()
    if patrol_was_running:
        # The API only signals the stop; the worker may still finish an
        # in-flight preset move before exiting.
        print(f"Waiting {args.stop_wait:.0f}s for the patrol worker to finish...")
        time.sleep(args.stop_wait)

    try:
        anchor_ratio = ratio_for_level(level_for_ratio(args.zoom_min))
        print(f"Zooming to ratio {anchor_ratio:.2f}x (anchor)...")
        api.zoom(level_for_ratio(args.zoom_min))
        time.sleep(args.settle)

        print("Measuring anchor FOV via hardware azimuth...")
        anchor = measure_anchor(api, args.anchor_pan, args.anchor_repeats)
        print(f"  anchor: h_fov={anchor['h_fov']:.2f}° v_fov={anchor['v_fov']:.2f}° (f_px={anchor['f_px']:.1f})")

        got = capture_qr(api)
        if got is None:
            raise SystemExit("QR not detected at anchor zoom after anchor measurement.")
        anchor_qr_px = qr_size_px(got[1])
        tan_half_h = math.tan(math.radians(anchor["h_fov"]) / 2)
        tan_half_v = math.tan(math.radians(anchor["v_fov"]) / 2)

        table: dict[int, Optional[dict[str, float]]] = {}
        for z in range(args.zoom_min, args.zoom_max + 1):
            level = level_for_ratio(z)
            achieved = ratio_for_level(level)
            print(f"Zoom ratio {achieved:.2f}x (level {level})...")
            api.zoom(level)
            time.sleep(args.settle)
            got = capture_qr(api)
            if got is None:
                print("  QR not detected, skipping")
                table[z] = None
                continue
            img, corners = got
            size = qr_size_px(corners)
            ratio = anchor_qr_px / size
            h_fov = math.degrees(2 * math.atan(tan_half_h * ratio))
            v_fov = math.degrees(2 * math.atan(tan_half_v * ratio))
            table[z] = {
                "zoom_ratio": round(achieved, 3),
                "h_fov": round(h_fov, 3),
                "v_fov": round(v_fov, 3),
                "qr_px": round(size, 1),
            }
            print(f"  qr={size:.0f}px → h_fov={h_fov:.2f}° v_fov={v_fov:.2f}°")
            if size > 0.8 * img.shape[1]:
                print("  QR nearly fills the frame, stopping here")
                break

        # Compare with the optical model fov(Z) = 2·atan(tan(fov0/2)/Z) anchored at zoom_min.
        model_check = {}
        for z, row in table.items():
            if row is None:
                continue
            model_h = math.degrees(2 * math.atan(tan_half_h * anchor_ratio / row["zoom_ratio"]))
            model_check[z] = {"measured_h": row["h_fov"], "model_h": round(model_h, 3)}

        report = {"camera_ip": args.ip, "anchor": anchor, "table": table, "optical_model_check": model_check}
        with Path(args.out).open("w") as f:
            json.dump(report, f, indent=2)
        print(f"\nReport written to {args.out}")
        print("If measured_h ≈ model_h across zooms, zoom_raw is the true optical ratio")
        print("and only the anchor wide FOV needs to go into the code.")

    finally:
        print(f"Restoring zoom ratio {args.zoom_min}x...")
        api.zoom(level_for_ratio(args.zoom_min))
        if patrol_was_running and not args.no_restart_patrol:
            print("Restarting patrol...")
            api.start_patrol()


if __name__ == "__main__":
    main()
