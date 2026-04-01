# Copyright (C) 2022-2026, Pyronear.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

"""
PTZ Calibration App — Streamlit UI for calibrating Reolink PTZ cameras.

Problem:
  Motion control is currently open loop: move(direction, speed) → sleep(T) → stop().
  This approach does not provide a direct relationship between T and traveled degrees,
  and is sensitive to start/stop delays (mechanical inertia).

Solution:
  For each axis (pan/tilt) and speed level (1-5), we measure real displacement
  using optical flow (OpenCV Farneback) on before/after impulse image pairs.
  We fit an affine model: δ ≈ ω·T + b
    - ω = effective speed (°/s)
    - b = bias linked to start/stop inertia

  Results can be used to update PAN_SPEEDS / TILT_SPEEDS in routes_control.py
  and implement a more accurate move_by_degrees.

Usage:
    pip install streamlit opencv-python-headless pillow requests pandas matplotlib
    streamlit run tools/ptz_calibration_app.py
"""

from __future__ import annotations

import json
import time
from io import BytesIO
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import streamlit as st
import urllib3
from PIL import Image, ImageDraw
from streamlit_image_coordinates import streamlit_image_coordinates

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ─── Constants ────────────────────────────────────────────────────────────────

# Measured FOV lookup table for Reolink RLC-823S2 (degrees), zoom levels 0–41.
# Calibrated via QR-code chained-ratio method. Plateau at zoom 41 (optical max).
H_FOV_TABLE: Dict[str, List[float]] = {
    "reolink-823S2": [
        54.2, 52.206, 50.405, 48.356, 46.167, 44.183, 42.63, 41.058, 39.117, 37.523,
        35.393, 33.804, 32.341, 30.742, 29.446, 27.829, 26.394, 24.992, 23.604, 22.136,
        20.948, 19.675, 18.652, 17.794, 16.352, 15.273, 14.278, 13.287, 12.577, 11.681,
        10.832, 9.992, 9.298, 8.644, 8.022, 7.411, 6.84, 6.323, 5.793, 5.303,
        4.787, 4.183,
    ],
    "reolink-823A16": [
        54.2, 52.029, 50.146, 47.986, 46.384, 44.431, 42.376, 40.915, 38.623, 37.135,
        35.303, 33.894, 32.273, 30.703, 29.167, 27.67, 26.181, 24.921, 23.489, 22.138,
        20.887, 19.701, 18.467, 17.618, 16.244, 15.203, 14.174, 13.242, 12.332, 11.606,
        10.771, 9.993, 9.283, 8.558, 7.914, 7.321, 6.777, 6.241, 5.744, 5.229,
        4.704, 4.118,
    ],
}
V_FOV_TABLE: Dict[str, List[float]] = {
    "reolink-823S2": [
        41.7, 40.166, 38.78, 37.204, 35.52, 33.993, 32.799, 31.589, 30.096, 28.869,
        27.23, 26.008, 24.882, 23.652, 22.655, 21.411, 20.307, 19.229, 18.16, 17.031,
        16.117, 15.138, 14.351, 13.69, 12.581, 11.751, 10.985, 10.223, 9.676, 8.987,
        8.334, 7.687, 7.154, 6.651, 6.172, 5.702, 5.263, 4.865, 4.457, 4.08,
        3.683, 3.219,
    ],
    "reolink-823A16": [
        41.7, 40.03, 38.581, 36.919, 35.686, 34.184, 32.603, 31.479, 29.716, 28.571,
        27.161, 26.077, 24.83, 23.622, 22.44, 21.289, 20.143, 19.174, 18.072, 17.032,
        16.07, 15.157, 14.208, 13.555, 12.498, 11.697, 10.905, 10.188, 9.488, 8.929,
        8.287, 7.688, 7.142, 6.584, 6.089, 5.633, 5.214, 4.802, 4.42, 4.023,
        3.619, 3.169,
    ],
}
_DEFAULT_ADAPTER = "reolink-823S2"
H_FOV_WIDE = 54.2
V_FOV_WIDE = 41.7

# Reference tables (existing measurements in routes_control.py)
REFERENCE_PAN_SPEEDS: Dict[str, Dict[int, float]] = {
    "reolink-823S2": {1: 1.5988, 2: 2.7877, 3: 4.5222, 4: 5.7913, 5: 6.3122},
    "reolink-823A16": {1: 1.3748, 2: 2.8895, 3: 4.5352, 4: 6.6175, 5: 7.3933},
}
REFERENCE_PAN_BIAS: Dict[str, Dict[int, float]] = {
    "reolink-823S2": {1: 0.6312, 2: 1.4915, 3: 1.748, 4: 2.9926, 5: 4.393},
    "reolink-823A16": {1: 2.0047, 2: 3.6327, 3: 5.5697, 4: 7.5964, 5: 10.176},
}
REFERENCE_TILT_SPEEDS: Dict[str, Dict[int, float]] = {
    "reolink-823S2": {1: 1.583, 2: 4.0438, 3: 6.9627},
    "reolink-823A16": {1: 2.0749, 2: 4.0741, 3: 5.5923},
}
REFERENCE_TILT_BIAS: Dict[str, Dict[int, float]] = {
    "reolink-823S2": {1: 1.462, 2: 2.1954, 3: 2.6174},
    "reolink-823A16": {1: 2.2971, 2: 4.5217, 3: 7.0047},
}

DEFAULT_IMPULSE_DURATIONS = [0.25, 0.4, 0.7, 1.2, 2.0]

# ─── Helpers ──────────────────────────────────────────────────────────────────


def draw_cross(img: Image.Image, x: int, y: int, color: str = "red", size: int = 22) -> Image.Image:
    """Draw a cross on the image at point (x, y)."""
    out = img.copy()
    d = ImageDraw.Draw(out)
    d.line([(x - size, y), (x + size, y)], fill=color, width=4)
    d.line([(x, y - size), (x, y + size)], fill=color, width=4)
    d.ellipse([(x - 6, y - 6), (x + 6, y + 6)], fill=color)
    return out



def fov_at_zoom(zoom_pos: int, adapter: Optional[str] = None) -> Tuple[float, float]:
    """Return (h_fov, v_fov) in degrees using measured lookup table with linear interpolation."""
    key = adapter if adapter in H_FOV_TABLE else _DEFAULT_ADAPTER
    h_table = H_FOV_TABLE[key]
    v_table = V_FOV_TABLE[key]
    z = max(0, min(zoom_pos, 41))
    z0 = int(z)
    z1 = min(z0 + 1, 41)
    t = z - z0
    h = h_table[z0] + t * (h_table[z1] - h_table[z0])
    v = v_table[z0] + t * (v_table[z1] - v_table[z0])
    return h, v


def estimate_displacement_deg(
    before: Image.Image,
    after: Image.Image,
    axis: str,
    h_fov: float,
    v_fov: float,
) -> float:
    """
    Estimate angular displacement (°) between two images via dense optical flow (Farneback).

    Sign convention:
      pan  : positive = camera moved to the right (Right)
      tilt : positive = camera moved upward (Up)
    """
    g1 = np.array(before.convert("L"))
    g2 = np.array(after.convert("L"))

    # Downscale if needed (faster computation)
    h, w = g1.shape
    if w > 1280:
        new_w, new_h = 1280, int(h * 1280.0 / w)
        g1 = cv2.resize(g1, (new_w, new_h))
        g2 = cv2.resize(g2, (new_w, new_h))
        h, w = new_h, new_w

    flow = cv2.calcOpticalFlowFarneback(
        g1, g2, None,
        pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
    )

    if axis == "pan":
        # Background moves left -> flow[...,0] negative -> camera goes right
        pixel_disp = float(-np.median(flow[..., 0]))
        return pixel_disp * h_fov / w
    else:
        # Background moves down -> flow[...,1] positive -> camera goes up
        pixel_disp = float(-np.median(flow[..., 1]))
        return pixel_disp * v_fov / h


def estimate_displacement_keypoints(
    before: Image.Image,
    after: Image.Image,
    axis: str,
    h_fov: float,
    v_fov: float,
) -> Tuple[float, int, float]:
    """
    Estimate angular displacement using ORB keypoint matching.

    Returns (displacement_deg, n_matches, median_px_delta).
    More robust than optical flow for large displacements.
    """
    g1 = np.array(before.convert("L"))
    g2 = np.array(after.convert("L"))
    h, w = g1.shape

    orb = cv2.ORB_create(nfeatures=2000)
    kp1, des1 = orb.detectAndCompute(g1, None)
    kp2, des2 = orb.detectAndCompute(g2, None)

    if des1 is None or des2 is None or len(kp1) < 10 or len(kp2) < 10:
        return 0.0, 0, 0.0

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    raw_matches = bf.knnMatch(des1, des2, k=2)

    # Lowe's ratio test
    good = []
    for m, n in raw_matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)

    if len(good) < 5:
        return 0.0, 0, 0.0

    # Compute pixel displacements for good matches
    if axis == "pan":
        deltas = [kp1[m.queryIdx].pt[0] - kp2[m.trainIdx].pt[0] for m in good]
        fov = h_fov
        size = w
    else:
        deltas = [kp2[m.trainIdx].pt[1] - kp1[m.queryIdx].pt[1] for m in good]
        fov = v_fov
        size = h

    # Use median to reject outliers (moving objects, mismatches)
    median_px = float(np.median(deltas))
    deg = abs(median_px) * fov / size
    return deg, len(good), median_px


def fit_model(durations: List[float], displacements: List[float]) -> Dict:
    """
    Affine fit: displacement = omega * duration + bias.
    Returns omega (°/s), bias (°), r2.
    """
    x = np.array(durations)
    y = np.array(displacements)
    if len(x) < 2:
        return {"omega": float(np.mean(y) / np.mean(x)) if (len(x) == 1 and np.mean(x) != 0) else 0.0, "bias": 0.0, "r2": 0.0}
    p = np.polyfit(x, y, 1)
    omega, bias = float(p[0]), float(p[1])
    y_pred = np.polyval(p, x)
    ss_res = float(np.sum((y - y_pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 1.0
    return {"omega": omega, "bias": bias, "r2": r2}


# ─── Camera clients ───────────────────────────────────────────────────────────


class DirectClient:
    """Directly controls a Reolink camera through its HTTP CGI API."""

    def __init__(self, ip: str, user: str, pwd: str, protocol: str = "https"):
        self.ip = ip
        self.user = user
        self.pwd = pwd
        self.protocol = protocol

    def _url(self, cmd: str) -> str:
        return (
            f"{self.protocol}://{self.ip}/cgi-bin/api.cgi"
            f"?cmd={cmd}&user={self.user}&password={self.pwd}&channel=0"
        )

    def capture(self) -> Optional[Image.Image]:
        try:
            r = requests.get(self._url("Snap"), verify=False, timeout=8)  # nosec: B501
            if r.status_code == 200:
                return Image.open(BytesIO(r.content)).convert("RGB")
        except Exception as exc:
            st.warning(f"Capture failed: {exc}")
        return None

    def _ptz(self, op: str, speed: int = 10, idx: int = 0) -> bool:
        data = [{"cmd": "PtzCtrl", "action": 0, "param": {"channel": 0, "op": op, "id": idx, "speed": speed}}]
        try:
            r = requests.post(self._url("PtzCtrl"), json=data, verify=False, timeout=5)  # nosec: B501
            return r.status_code == 200
        except Exception as exc:
            st.warning(f"PTZ command failed ({op}): {exc}")
            return False

    def move(self, direction: str, speed: int = 10, duration: Optional[float] = None) -> bool:
        ok = self._ptz(direction, speed)
        if ok and duration is not None:
            if duration > 0:
                time.sleep(duration)
            self._ptz("Stop")
        return ok

    def stop(self) -> bool:
        return self._ptz("Stop")

    def goto_preset(self, idx: int, speed: int = 50) -> bool:
        return self._ptz("ToPos", speed, idx)

    def zoom(self, level: int):
        data = [{"cmd": "StartZoomFocus", "action": 0,
                 "param": {"ZoomFocus": {"channel": 0, "pos": level, "op": "ZoomPos"}}}]
        try:
            requests.post(self._url("StartZoomFocus"), json=data, verify=False, timeout=5)  # nosec: B501
        except Exception as exc:
            st.warning(f"Zoom failed: {exc}")

    def get_zoom(self) -> Optional[int]:
        data = [{"cmd": "GetZoomFocus", "action": 0, "param": {"channel": 0}}]
        try:
            r = requests.post(self._url("GetZoomFocus"), json=data, verify=False, timeout=5)  # nosec: B501
            res = r.json()
            if res[0]["code"] == 0:
                return res[0]["value"]["ZoomFocus"].get("zoom", {}).get("pos")
        except Exception:
            pass
        return None

    def get_presets(self) -> List[Dict]:
        data = [{"cmd": "GetPtzPreset", "action": 1, "param": {"channel": 0}}]
        try:
            r = requests.post(self._url("GetPtzPreset"), json=data, verify=False, timeout=5)  # nosec: B501
            res = r.json()
            if res[0]["code"] == 0:
                return [p for p in res[0].get("value", {}).get("PtzPreset", []) if p.get("enable") == 1]
        except Exception:
            pass
        return []

    def set_preset(self, idx: int, name: str = "") -> bool:
        n = name or f"pos{idx}"
        data = [{"cmd": "SetPtzPreset", "action": 0,
                 "param": {"PtzPreset": {"channel": 0, "enable": 1, "id": idx, "name": n}}}]
        try:
            r = requests.post(self._url("SetPtzPreset"), json=data, verify=False, timeout=5)  # nosec: B501
            return r.status_code == 200
        except Exception as exc:
            st.warning(f"Set preset failed: {exc}")
            return False

    def stop_patrol(self):
        pass  # No patrol service in direct mode

    def get_patrol_status(self) -> Optional[Dict]:
        return None  # Direct mode: no patrol service


class APIClient:
    """Controls the camera through the pyro_camera_api FastAPI service."""

    def __init__(self, base_url: str, camera_ip: str):
        self.base = base_url.rstrip("/")
        self.ip = camera_ip

    def _get(self, path: str, **params) -> Optional[requests.Response]:
        try:
            r = requests.get(f"{self.base}/{path.lstrip('/')}", params=params, timeout=10)
            r.raise_for_status()
            return r
        except Exception as exc:
            st.warning(f"GET {path} failed: {exc}")
            return None

    def _post(self, path: str, **params) -> Optional[requests.Response]:
        try:
            r = requests.post(f"{self.base}/{path.lstrip('/')}", params=params, timeout=10)
            return r
        except Exception as exc:
            st.warning(f"POST {path} failed: {exc}")
            return None

    def capture(self) -> Optional[Image.Image]:
        r = self._get("cameras/capture", camera_ip=self.ip, anonymize="false")
        if r is not None:
            return Image.open(BytesIO(r.content)).convert("RGB")
        return None

    def move(self, direction: str, speed: int = 10, duration: Optional[float] = None) -> bool:
        params: Dict = {"camera_ip": self.ip, "direction": direction, "speed": speed}
        if duration is not None:
            params["duration"] = duration
        r = self._post("control/move", **params)
        return r is not None and r.status_code == 200

    def stop(self) -> bool:
        r = self._post(f"control/stop/{self.ip}")
        return r is not None and r.status_code == 200

    def goto_preset(self, idx: int, speed: int = 50) -> bool:
        r = self._post("control/move", camera_ip=self.ip, pose_id=idx, speed=speed)
        return r is not None and r.status_code == 200

    def zoom(self, level: int):
        self._post(f"control/zoom/{self.ip}/{level}")

    def get_zoom(self) -> Optional[int]:
        return None  # Not exposed by the API

    def get_presets(self) -> List[Dict]:
        r = self._get("control/preset/list", camera_ip=self.ip)
        if r is not None:
            return [p for p in r.json().get("presets", []) if p.get("enable") == 1]
        return []

    def set_preset(self, idx: int, name: str = "") -> bool:
        r = self._post("control/preset/set", camera_ip=self.ip, idx=idx)
        return r is not None and r.status_code == 200

    def stop_patrol(self):
        try:
            requests.post(f"{self.base}/patrol/stop_patrol", params={"camera_ip": self.ip}, timeout=5)
        except Exception:
            pass

    def get_patrol_status(self) -> Optional[Dict]:
        try:
            r = requests.get(f"{self.base}/patrol/patrol_status", params={"camera_ip": self.ip}, timeout=5)
            if r.status_code == 200:
                return r.json()
        except Exception:
            pass
        return None

    def click_to_move(self, click_x: int, click_y: int, image_width: int, image_height: int, zoom: int = 0) -> Optional[Dict]:
        r = self._post(
            "control/click_to_move",
            camera_ip=self.ip,
            click_x=click_x,
            click_y=click_y,
            image_width=image_width,
            image_height=image_height,
            zoom=zoom,
        )
        if r is not None and r.status_code == 200:
            return r.json()
        return None

    def get_adapter(self) -> Optional[str]:
        """Return the adapter name for this camera IP from /cameras/camera_infos."""
        try:
            r = requests.get(f"{self.base}/cameras/camera_infos", timeout=5)
            if r.status_code == 200:
                for cam in r.json().get("cameras", []):
                    if cam.get("ip") == self.ip or cam.get("camera_id") == self.ip:
                        return cam.get("adapter")
        except Exception:
            pass
        return None

    @staticmethod
    def list_cameras(base_url: str) -> List[str]:
        try:
            r = requests.get(f"{base_url.rstrip('/')}/cameras/cameras_list", timeout=5)
            if r.status_code == 200:
                return r.json().get("camera_ids", [])
        except Exception:
            pass
        return []


# ─── Streamlit App ────────────────────────────────────────────────────────────

st.set_page_config(page_title="PTZ Calibration", page_icon="📷", layout="wide")
_title_model = st.session_state.get("cam_model") or "PTZ"
st.title(f"📷 PTZ Calibration — {_title_model}")

# Session state initialization
for _k, _v in [
    ("client", None),
    ("cam_model", "reolink-823S2"),
    ("calib_zoom", 0),
    ("calib_results", {}),
    ("calib_raw_data", {}),
    ("live_img", None),
    ("presets_list", []),
    ("api_camera_list", []),
    # Semi-manual calibration state machine
    ("calib_phase", "setup"),       # "setup" | "annotating" | "complete"
    ("calib_pairs", []),            # [{speed, T, axis, direction, img_before, img_after, h_fov, v_fov, zoom}]
    ("calib_anno_idx", 0),          # index of pair being annotated
    ("calib_click_before", None),   # (x, y) or None
    ("calib_click_after", None),    # (x, y) or None
    ("calib_annotations", []),      # [{speed, T, axis, px_delta, disp_deg}]
    # Micro-pulse calibration state machine
    ("micro_phase", "idle"),        # "idle" | "annotating" | "complete"
    ("micro_pairs", []),            # [{img_before, img_after, h_fov, v_fov, axis, direction, rep}]
    ("micro_anno_idx", 0),
    ("micro_click_before", None),
    ("micro_click_after", None),
    ("micro_annotations", []),      # [{disp_deg, axis}]
    # Click-to-move state machine
    ("ctm_phase", "idle"),          # "idle" | "verify"
    ("ctm_img_before", None),       # image before move
    ("ctm_img_after", None),        # image after move
    ("ctm_target_click", None),     # (x, y) user clicked on before image
    ("ctm_verify_click", None),     # (x, y) user clicked on after image
    ("ctm_last_move", {}),          # {pan_deg, tilt_deg, pan_speed, tilt_speed, pan_dur, tilt_dur}
    # Zoom FOV calibration state machine
    ("zfov_phase", "idle"),         # "idle" | "calibrating" | "complete"
    ("zfov_levels", []),            # list of zoom levels to visit
    ("zfov_idx", 0),                # current index in zfov_levels
    ("zfov_measurements", []),       # [{zoom, px_size, fov_h, fov_v}]
    ("zfov_last_ref", None),        # {px, fov_h, fov_v} — reference for next ratio
    ("zfov_img", None),             # current captured image
    ("zfov_corners", None),         # detected QR corners (np.ndarray or None)
    ("zfov_auto_capture", False),   # trigger capture automatically on next render
]:
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ─── Sidebar: Connection ──────────────────────────────────────────────────────

with st.sidebar:
    st.header("🔌 Connection")
    mode = st.radio("Mode", ["Direct Reolink", "Through Camera API"], key="conn_mode")

    if mode == "Direct Reolink":
        ip = st.text_input("Camera IP", value="192.168.1.12")
        user = st.text_input("Username", value="admin")
        pwd = st.text_input("Password", type="password")
        proto = st.selectbox("Protocol", ["https", "http"])

        if st.button("🔗 Connect", key="btn_connect_direct"):
            with st.spinner("Testing connection..."):
                c = DirectClient(ip, user, pwd, proto)
                img = c.capture()
            if img is not None:
                st.session_state["client"] = c
                st.session_state["live_img"] = img
                st.success(f"Connected to {ip}")
            else:
                st.error("Connection failed - check IP/username/password")

    else:  # Through Camera API
        api_url = st.text_input("URL Camera API", value="http://192.168.255.62:8081")

        if st.button("🔍 List cameras"):
            with st.spinner("Fetching..."):
                cams = APIClient.list_cameras(api_url)
            if cams:
                st.session_state["api_camera_list"] = cams
                st.success(f"{len(cams)} camera(s) found")
            else:
                st.warning("No camera found or API unreachable")

        cam_list = st.session_state["api_camera_list"]
        if cam_list:
            selected_cam = st.selectbox("Camera", cam_list)
        else:
            selected_cam = st.text_input("Camera IP (manual)", value="192.168.1.12")

        if st.button("🔗 Connect", key="btn_connect_api"):
            with st.spinner("Testing connection..."):
                c = APIClient(api_url, selected_cam)
                img = c.capture()
                adapter = c.get_adapter()
            if img is not None:
                st.session_state["client"] = c
                st.session_state["live_img"] = img
                if adapter:
                    st.session_state["cam_model"] = adapter
                    st.session_state["sel_model"] = adapter
                    st.success(f"Connected through API ({selected_cam}) - adapter: {adapter}")
                else:
                    st.success(f"Connected through API ({selected_cam})")
            else:
                st.error("Connection failed")

    st.divider()
    st.header("⚙️ Model")
    known_adapters = ["reolink-823A16", "reolink-823S2"]
    current_model = st.session_state.get("cam_model", known_adapters[0])
    # Always ensure the detected adapter is selectable
    if current_model not in known_adapters:
        known_adapters = [current_model] + known_adapters
    st.session_state["cam_model"] = st.selectbox(
        "Adapter (auto-detected via API)", known_adapters, key="sel_model"
    )

    # Summary of ongoing calibrations
    if st.session_state["calib_results"]:
        st.divider()
        st.caption("**Completed calibrations:**")
        for key, r in st.session_state["calib_results"].items():
            st.caption(f"• {key}: ω={r['omega']:.3f} °/s, b={r['bias']:.3f}°")

# ─── Connection check ─────────────────────────────────────────────────────────

if st.session_state["client"] is None:
    st.info("👈 Connect to a camera through the sidebar to get started.")
    st.stop()

client = st.session_state["client"]
cam_model: str = st.session_state["cam_model"]

tab_view, tab_ctm, tab_zfov, tab_calib, tab_results, tab_presets = st.tabs([
    "👁️ Live View", "🎯 Click-to-Move", "🔭 Zoom FOV", "🔬 Calibration", "📈 Results & Export", "📌 Presets"
])

# ─── Tab 0 : Live View ────────────────────────────────────────────────────────

with tab_view:
    col_img, col_ctrl = st.columns([3, 1])

    with col_ctrl:
        st.subheader("Controls")

        if st.button("📸 Capture", key="btn_snap_live"):
            with st.spinner("Capturing..."):
                img = client.capture()
            if img is not None:
                st.session_state["live_img"] = img

        st.divider()
        st.write("**Zoom**")
        zoom_val = st.slider("Zoom position", 0, 64, st.session_state["calib_zoom"], key="zoom_live")
        if st.button("Apply zoom"):
            client.zoom(zoom_val)
            time.sleep(2)
            st.session_state["calib_zoom"] = zoom_val
            h_fov, v_fov = fov_at_zoom(zoom_val, cam_model)
            st.success(f"Zoom {zoom_val} → FOV H={h_fov:.1f}° V={v_fov:.1f}°")

        st.divider()
        st.write("**Manual movement**")
        speed_m = st.slider("Speed", 1, 10, 3, key="speed_manual")
        dur_m = st.slider("Duration (s)", 0.2, 3.0, 0.5, 0.1, key="dur_manual")

        cols_pad = st.columns([1, 1, 1])
        with cols_pad[1]:
            if st.button("⬆️"):
                client.move("Up", speed_m, duration=dur_m)
        cols_lr = st.columns([1, 1, 1])
        with cols_lr[0]:
            if st.button("⬅️"):
                client.move("Left", speed_m, duration=dur_m)
        with cols_lr[2]:
            if st.button("➡️"):
                client.move("Right", speed_m, duration=dur_m)
        cols_pad2 = st.columns([1, 1, 1])
        with cols_pad2[1]:
            if st.button("⬇️"):
                client.move("Down", speed_m, duration=dur_m)

        if st.button("🛑 STOP", type="primary"):
            client.stop()

    with col_img:
        if st.session_state["live_img"] is not None:
            st.image(st.session_state["live_img"], width="stretch")
            w, h = st.session_state["live_img"].size
            calib_zoom = st.session_state["calib_zoom"]
            h_fov, v_fov = fov_at_zoom(calib_zoom, cam_model)
            st.caption(
                f"Resolution: {w}×{h}px | Zoom: {calib_zoom} | "
                f"FOV H={h_fov:.1f}° ({h_fov/w*1000:.2f} mrad/px) | "
                f"FOV V={v_fov:.1f}°"
            )
        else:
            st.info("Click 'Capture' to view the image.")

# ─── Tab 1 : Click-to-Move ───────────────────────────────────────────────────

# Speed tables mirroring routes_control.py (used for local speed selection)
_CTM_PAN_SPEEDS: Dict[str, Dict[int, float]] = {
    "reolink-823S2": {1: 1.5988, 2: 2.7877, 3: 4.5222, 4: 5.7913, 5: 6.3122},
    "reolink-823A16": {1: 1.3748, 2: 2.8895, 3: 4.5352, 4: 6.6175, 5: 7.3933},
}
_CTM_PAN_BIAS: Dict[str, Dict[int, float]] = {
    "reolink-823S2": {1: 0.6312, 2: 1.4915, 3: 1.748, 4: 2.9926, 5: 4.393},
    "reolink-823A16": {1: 2.0047, 2: 3.6327, 3: 5.5697, 4: 7.5964, 5: 10.176},
}
_CTM_TILT_SPEEDS: Dict[str, Dict[int, float]] = {
    "reolink-823S2": {1: 1.583, 2: 4.0438, 3: 6.9627},
    "reolink-823A16": {1: 2.0749, 2: 4.0741, 3: 5.5923},
}
_CTM_TILT_BIAS: Dict[str, Dict[int, float]] = {
    "reolink-823S2": {1: 1.462, 2: 2.1954, 3: 2.6174},
    "reolink-823A16": {1: 2.2971, 2: 4.5217, 3: 7.0047},
}


def _pick_speed(target_deg: float, speeds: Dict[int, float], bias: Dict[int, float]) -> Optional[int]:
    """Pick the highest speed level where T = (target - b) / ω is in [0.3, 4.0]s."""
    best: Optional[int] = None
    for level in sorted(speeds.keys()):
        b = bias.get(level, 0.0)
        omega = speeds[level]
        if target_deg <= b:
            continue  # can't execute: coast alone overshoots
        duration = (target_deg - b) / omega
        if 0.3 <= duration <= 4.0:
            best = level  # keep highest valid level
    return best


def _ctm_move_axis(
    axis_deg: float,
    direction_pos: str,
    direction_neg: str,
    speeds: Dict[int, float],
    bias: Dict[int, float],
) -> Optional[Dict]:
    """Execute a single-axis move and return move metadata, or None if skipped."""
    if abs(axis_deg) < 0.5:
        return None
    direction = direction_pos if axis_deg > 0 else direction_neg
    speed = _pick_speed(abs(axis_deg), speeds, bias)
    if speed is None:
        st.warning(f"No valid speed level for {abs(axis_deg):.1f}° on axis {direction_pos}/{direction_neg}")
        return None
    b = bias.get(speed, 0.0)
    omega = speeds[speed]
    duration = (abs(axis_deg) - b) / omega
    client.move(direction, speed=speed, duration=duration)
    return {"deg": axis_deg, "direction": direction, "speed": speed, "duration": round(duration, 2), "bias": b, "omega": omega}


with tab_ctm:
    st.subheader("🎯 Click-to-Move — Accuracy test")
    st.markdown(
        "Click a point on the image -> the camera moves to center it -> "
        "click the same landmark on the new image to measure error."
    )

    ctm_phase = st.session_state["ctm_phase"]
    col_zoom, col_zoom_btn = st.columns([3, 1])
    ctm_zoom = col_zoom.number_input("Zoom level (0=wide, 64=tele)", 0, 64, int(st.session_state["calib_zoom"]), 1, key="ctm_zoom")
    if col_zoom_btn.button("Apply zoom", key="ctm_apply_zoom"):
        client.zoom(ctm_zoom)
        st.session_state["calib_zoom"] = ctm_zoom
        with st.spinner(f"Zoom {ctm_zoom} — waiting for camera…"):
            time.sleep(2.0)
            img_z = client.capture()
        if img_z:
            st.session_state["ctm_img_before"] = img_z
            st.session_state["ctm_target_click"] = None
        st.rerun()
    h_fov_ctm, v_fov_ctm = fov_at_zoom(ctm_zoom, cam_model)

    # ── Phase idle: capture + click target ──────────────────────────────────
    if ctm_phase == "idle":
        col_cap, col_rst = st.columns([1, 1])
        if col_cap.button("📸 Capture image", key="ctm_capture"):
            with st.spinner("Capturing..."):
                img = client.capture()
            if img:
                st.session_state["ctm_img_before"] = img
                st.session_state["ctm_target_click"] = None
                st.rerun()

        img_before: Optional[Image.Image] = st.session_state["ctm_img_before"]
        if img_before is None:
            st.info("Click 'Capture image' to start.")
        else:
            W_ctm, H_ctm = img_before.size
            _DISP_W_CTM = 900
            _scale_ctm = _DISP_W_CTM / W_ctm
            target_click: Optional[Tuple[int, int]] = st.session_state["ctm_target_click"]
            display_before = draw_cross(img_before, target_click[0], target_click[1]) if target_click else img_before
            display_before_small = display_before.resize((_DISP_W_CTM, int(H_ctm * _scale_ctm)), Image.LANCZOS)

            st.markdown("**Click the target point on the image:**")
            coord = streamlit_image_coordinates(display_before_small, key="ctm_click_before")
            if coord is not None:
                new_pt = (int(coord["x"] / _scale_ctm), int(coord["y"] / _scale_ctm))
                if new_pt != target_click:
                    st.session_state["ctm_target_click"] = new_pt
                    st.rerun()

            if target_click:
                dx_px = target_click[0] - W_ctm // 2
                dy_px = target_click[1] - H_ctm // 2
                pan_deg = dx_px * h_fov_ctm / W_ctm
                tilt_deg = dy_px * v_fov_ctm / H_ctm

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Click", f"({target_click[0]}, {target_click[1]})")
                c2.metric("Δ center", f"({dx_px:+d}px, {dy_px:+d}px)")
                c3.metric("Pan", f"{pan_deg:+.2f}°")
                c4.metric("Tilt", f"{tilt_deg:+.2f}°")

                pan_speeds_ctm = _CTM_PAN_SPEEDS.get(cam_model, {})
                pan_bias_ctm = _CTM_PAN_BIAS.get(cam_model, {})
                tilt_speeds_ctm = _CTM_TILT_SPEEDS.get(cam_model, {})
                tilt_bias_ctm = _CTM_TILT_BIAS.get(cam_model, {})

                pan_speed_sel = _pick_speed(abs(pan_deg), pan_speeds_ctm, pan_bias_ctm)
                tilt_speed_sel = _pick_speed(abs(tilt_deg), tilt_speeds_ctm, tilt_bias_ctm)

                cs1, cs2 = st.columns(2)
                cs1.info(f"Pan -> speed {pan_speed_sel}" if pan_speed_sel else "Pan -> too small (<0.5°), skipped")
                cs2.info(f"Tilt -> speed {tilt_speed_sel}" if tilt_speed_sel else "Tilt -> too small (<0.5°), skipped")

                if st.button("🚀 Move to this point", type="primary", key="ctm_go"):
                    move_log = {}
                    if isinstance(client, APIClient):
                        with st.spinner("click_to_move via API…"):
                            api_result = client.click_to_move(
                                target_click[0], target_click[1],
                                W_ctm, H_ctm,
                                zoom=ctm_zoom,
                            )
                        if api_result:
                            for mv in api_result.get("moves", []):
                                if not mv.get("skipped"):
                                    move_log[mv["axis"]] = mv
                    else:
                        with st.spinner("Pan movement..."):
                            r = _ctm_move_axis(pan_deg, "Right", "Left", pan_speeds_ctm, pan_bias_ctm)
                            if r:
                                move_log["pan"] = r
                        with st.spinner("Tilt movement..."):
                            r = _ctm_move_axis(tilt_deg, "Down", "Up", tilt_speeds_ctm, tilt_bias_ctm)
                            if r:
                                move_log["tilt"] = r
                    with st.spinner("Capturing after movement..."):
                        time.sleep(0.5)
                        img_after = client.capture()
                    if img_after:
                        st.session_state["ctm_img_after"] = img_after
                        st.session_state["ctm_last_move"] = move_log
                        st.session_state["ctm_verify_click"] = None
                        st.session_state["ctm_phase"] = "verify"
                        st.rerun()

    # ── Phase verify: show after image, user clicks same landmark ───────────
    elif ctm_phase == "verify":
        img_after_v: Optional[Image.Image] = st.session_state["ctm_img_after"]
        verify_click: Optional[Tuple[int, int]] = st.session_state["ctm_verify_click"]
        move_log_v = st.session_state["ctm_last_move"]
        img_before_v: Optional[Image.Image] = st.session_state["ctm_img_before"]
        target_click_v: Optional[Tuple[int, int]] = st.session_state["ctm_target_click"]

        if img_after_v is None:
            st.error("Missing after image.")
        else:
            W_v, H_v = img_after_v.size
            cx, cy = W_v // 2, H_v // 2

            st.markdown("### Movement summary")
            cols_mv = st.columns(len(move_log_v) * 3 or 1)
            i_col = 0
            for axis_name, mv in move_log_v.items():
                cols_mv[i_col].metric(f"{axis_name} direction", mv["direction"])
                cols_mv[i_col + 1].metric(f"{axis_name} speed", mv["speed"])
                cols_mv[i_col + 2].metric(f"{axis_name} duration", f"{mv['duration']}s")
                i_col += 3

            _CTM_DISP_W = 900

            st.markdown("**BEFORE image** (target in red)")
            if img_before_v is not None:
                W_b, H_b = img_before_v.size
                scale_b = _CTM_DISP_W / W_b
                bef_img = draw_cross(img_before_v, target_click_v[0], target_click_v[1]) if target_click_v else img_before_v
                bef_small = bef_img.resize((_CTM_DISP_W, int(H_b * scale_b)), Image.LANCZOS)
                st.image(bef_small, width="stretch")

            st.markdown("**AFTER image** — click the same landmark")
            scale_v = _CTM_DISP_W / W_v
            # Draw annotations at full res, then resize for display
            display_v = draw_cross(img_after_v, verify_click[0], verify_click[1], color="blue") if verify_click else img_after_v
            center_img = display_v.copy()
            d = ImageDraw.Draw(center_img)
            d.line([(cx - 30, cy), (cx + 30, cy)], fill="yellow", width=2)
            d.line([(cx, cy - 30), (cx, cy + 30)], fill="yellow", width=2)
            center_small = center_img.resize((_CTM_DISP_W, int(H_v * scale_v)), Image.LANCZOS)
            coord_v = streamlit_image_coordinates(center_small, key="ctm_click_after")
            if coord_v is not None:
                # Scale click back to full-res coordinates
                new_v = (int(coord_v["x"] / scale_v), int(coord_v["y"] / scale_v))
                if new_v != verify_click:
                    st.session_state["ctm_verify_click"] = new_v
                    st.rerun()

            if verify_click:
                err_px_x = verify_click[0] - cx
                err_px_y = verify_click[1] - cy
                err_deg_pan = err_px_x * h_fov_ctm / W_v
                err_deg_tilt = err_px_y * v_fov_ctm / H_v
                err_total = (err_deg_pan**2 + err_deg_tilt**2) ** 0.5

                st.divider()
                st.markdown("### Pointing error")
                e1, e2, e3, e4 = st.columns(4)
                e1.metric("Pan error", f"{err_deg_pan:+.2f}°")
                e2.metric("Tilt error", f"{err_deg_tilt:+.2f}°")
                e3.metric("Total error", f"{err_total:.2f}°")
                e4.metric("Pixel error", f"({err_px_x:+d}, {err_px_y:+d})")

                if err_total < 1.0:
                    st.success(f"Excellent accuracy: {err_total:.2f}° < 1°")
                elif err_total < 3.0:
                    st.warning(f"Acceptable accuracy: {err_total:.2f}°")
                else:
                    st.error(f"Large error: {err_total:.2f}° - recalibration recommended")

            st.divider()
            if st.button("🔄 Restart", key="ctm_reset"):
                st.session_state["ctm_phase"] = "idle"
                st.session_state["ctm_img_before"] = None
                st.session_state["ctm_img_after"] = None
                st.session_state["ctm_target_click"] = None
                st.session_state["ctm_verify_click"] = None
                st.session_state["ctm_last_move"] = {}
                st.rerun()

# ─── Tab 2 : Zoom FOV Calibration (QR code) ──────────────────────────────────

def _detect_qr(img: Image.Image) -> Optional[Tuple[float, np.ndarray]]:
    """Detect QR code and return (avg_side_px, corners). None if not found."""
    gray = np.array(img.convert("L"))
    detector = cv2.QRCodeDetector()
    ok, _, points, _ = detector.detectAndDecodeMulti(gray)
    if not ok or points is None or len(points) == 0:
        return None
    corners = points[0].reshape(4, 2)
    sides = [float(np.linalg.norm(corners[i] - corners[(i + 1) % 4])) for i in range(4)]
    return float(np.mean(sides)), corners


def _draw_qr(img: Image.Image, corners: np.ndarray) -> Image.Image:
    out = img.copy()
    d = ImageDraw.Draw(out)
    pts = [(int(corners[i][0]), int(corners[i][1])) for i in range(4)]
    for i in range(4):
        d.line([pts[i], pts[(i + 1) % 4]], fill="lime", width=4)
    cx = int(np.mean(corners[:, 0]))
    cy = int(np.mean(corners[:, 1]))
    d.ellipse([(cx - 8, cy - 8), (cx + 8, cy + 8)], fill="lime")
    return out


with tab_zfov:
    st.subheader("🔭 Zoom FOV Calibration — QR code method")
    st.markdown(
        "Place a QR code in front of the camera at a **fixed position**. "
        "The app iterates through zoom levels, measures the QR code size in pixels, "
        "and derives the actual H/V FOV at each level using zoom 0 as reference (54.2° / 41.7°)."
    )

    zfov_phase = st.session_state["zfov_phase"]

    # ── Setup ────────────────────────────────────────────────────────────────
    if zfov_phase == "idle":
        col_s1, col_s2, col_s3 = st.columns(3)
        zoom_min = col_s1.number_input("Zoom min", 0, 63, 0, 1, key="zfov_min")
        zoom_max = col_s2.number_input("Zoom max", 1, 64, 64, 1, key="zfov_max")
        zoom_step = col_s3.number_input("Step", 1, 16, 1, 1, key="zfov_step")

        levels = list(range(int(zoom_min), int(zoom_max) + 1, int(zoom_step)))
        st.info(f"{len(levels)} zoom levels to calibrate: {levels[:8]}{'…' if len(levels) > 8 else ''}")

        if st.button("🚀 Start FOV calibration", type="primary", key="zfov_start"):
            st.session_state["zfov_levels"] = levels
            st.session_state["zfov_idx"] = 0
            st.session_state["zfov_measurements"] = []
            st.session_state["zfov_last_ref"] = None
            st.session_state["zfov_img"] = None
            st.session_state["zfov_corners"] = None
            st.session_state["zfov_auto_capture"] = True
            st.session_state["zfov_phase"] = "calibrating"
            st.rerun()

    # ── Calibrating ──────────────────────────────────────────────────────────
    elif zfov_phase == "calibrating":
        levels = st.session_state["zfov_levels"]
        idx = st.session_state["zfov_idx"]
        measurements: List[Dict] = st.session_state["zfov_measurements"]
        last_ref: Optional[Dict] = st.session_state["zfov_last_ref"]  # {px, fov_h, fov_v}

        if idx >= len(levels):
            st.session_state["zfov_phase"] = "complete"
            st.rerun()

        current_zoom = levels[idx]
        st.progress(idx / len(levels), text=f"Zoom {current_zoom} — {idx}/{len(levels)}")
        st.markdown(f"### Zoom level **{current_zoom}** ({idx + 1}/{len(levels)})")

        if last_ref:
            st.caption(f"Reference: previous step px={last_ref['px']:.1f}, FOV H={last_ref['fov_h']:.2f}°")
        else:
            st.caption(f"First measurement — FOV will be set to {H_FOV_WIDE}° / {V_FOV_WIDE}°")

        settle_z = st.slider("Focus settle time (s)", 0.5, 10.0, 2.0, 0.5, key=f"zfov_settle_{idx}")

        def _zoom_and_detect(zoom: int, settle: float) -> None:
            client.zoom(zoom)
            time.sleep(settle)
            for attempt in range(10):
                img_z = client.capture()
                if img_z:
                    result_z = _detect_qr(img_z)
                    st.session_state["zfov_img"] = img_z
                    if result_z:
                        st.session_state["zfov_corners"] = result_z[1]
                        return
                    st.session_state["zfov_corners"] = None
                if attempt < 9:
                    time.sleep(1.0)

        # Auto-capture after validate (or on first step)
        if st.session_state["zfov_auto_capture"]:
            st.session_state["zfov_auto_capture"] = False
            with st.spinner(f"Zoom {current_zoom} — up to 10 detection attempts…"):
                _zoom_and_detect(current_zoom, settle_z)
            st.rerun()

        col_cap, col_skip = st.columns([3, 1])

        if col_cap.button("📸 Retry capture", key=f"zfov_cap_{idx}"):
            with st.spinner(f"Zoom {current_zoom} — up to 10 detection attempts…"):
                _zoom_and_detect(current_zoom, settle_z)
            st.rerun()

        if col_skip.button("⏭️ Skip", key=f"zfov_skip_{idx}"):
            st.session_state["zfov_idx"] = idx + 1
            st.session_state["zfov_img"] = None
            st.session_state["zfov_corners"] = None
            st.rerun()

        zfov_img: Optional[Image.Image] = st.session_state["zfov_img"]
        zfov_corners = st.session_state["zfov_corners"]

        if zfov_img is not None:
            W_z, H_z = zfov_img.size
            display_z = _draw_qr(zfov_img, zfov_corners) if zfov_corners is not None else zfov_img
            scale_z = 900 / W_z
            st.image(display_z.resize((900, int(H_z * scale_z)), Image.LANCZOS), width="stretch")

            if zfov_corners is not None:
                sides_z = [float(np.linalg.norm(zfov_corners[i] - zfov_corners[(i + 1) % 4])) for i in range(4)]
                px_size = float(np.mean(sides_z))

                # Compute FOV from previous reference (chained ratio)
                if last_ref is None:
                    fov_h_est, fov_v_est = H_FOV_WIDE, V_FOV_WIDE
                else:
                    fov_h_est = last_ref["fov_h"] * last_ref["px"] / px_size
                    fov_v_est = last_ref["fov_v"] * last_ref["px"] / px_size

                c1, c2, c3 = st.columns(3)
                c1.metric("QR size", f"{px_size:.1f} px")
                c2.metric("H FOV", f"{fov_h_est:.2f}°")
                c3.metric("V FOV", f"{fov_v_est:.2f}°")

                col_v, col_b = st.columns(2)

                if col_v.button("✅ Validate & next", type="primary", key=f"zfov_val_{idx}"):
                    measurements.append({"zoom": current_zoom, "px_size": round(px_size, 1),
                                         "fov_h": round(fov_h_est, 3), "fov_v": round(fov_v_est, 3)})
                    st.session_state["zfov_measurements"] = measurements
                    st.session_state["zfov_last_ref"] = {"px": px_size, "fov_h": fov_h_est, "fov_v": fov_v_est}
                    st.session_state["zfov_idx"] = idx + 1
                    st.session_state["zfov_img"] = None
                    st.session_state["zfov_corners"] = None
                    st.session_state["zfov_auto_capture"] = True
                    st.rerun()

                if col_b.button("📐 QR moved — set as new reference", key=f"zfov_bridge_{idx}"):
                    # Bridge: keep same FOV as last ref (or wide FOV if first), update px reference
                    bridge_fov_h = last_ref["fov_h"] if last_ref else H_FOV_WIDE
                    bridge_fov_v = last_ref["fov_v"] if last_ref else V_FOV_WIDE
                    st.session_state["zfov_last_ref"] = {"px": px_size, "fov_h": bridge_fov_h, "fov_v": bridge_fov_v}
                    st.session_state["zfov_img"] = None
                    st.session_state["zfov_corners"] = None
                    st.rerun()
            else:
                st.warning("QR not detected — retry capture or skip.")

        if st.button("🏁 Finish early & show results", key="zfov_finish"):
            st.session_state["zfov_phase"] = "complete"
            st.rerun()

    # ── Complete ──────────────────────────────────────────────────────────────
    elif zfov_phase == "complete":
        measurements_list: List[Dict] = st.session_state["zfov_measurements"]
        if len(measurements_list) < 2:
            st.error("Need at least 2 measurements. Go back and retry.")
        else:
            df_fov = pd.DataFrame(measurements_list).rename(columns={"px_size": "qr_px", "fov_h": "h_fov", "fov_v": "v_fov"})
            st.dataframe(df_fov, hide_index=True, width="stretch")

            fig_fov, ax_fov = plt.subplots(figsize=(10, 4))
            ax_fov.plot(df_fov["zoom"], df_fov["h_fov"], "o-", label="H FOV")
            ax_fov.plot(df_fov["zoom"], df_fov["v_fov"], "s-", label="V FOV")
            ax_fov.set_xlabel("Zoom level")
            ax_fov.set_ylabel("FOV (°)")
            ax_fov.set_title("FOV vs Zoom level")
            ax_fov.legend()
            ax_fov.grid(True)
            st.pyplot(fig_fov)

            export_fov = {
                "model": cam_model,
                "calibrated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "measurements": df_fov.to_dict(orient="records"),
            }
            st.download_button(
                "⬇️ Download FOV table JSON",
                data=json.dumps(export_fov, indent=2),
                file_name=f"fov_calib_{cam_model}_{time.strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
            )

        if st.button("🔄 Restart FOV calibration", key="zfov_restart"):
            st.session_state["zfov_phase"] = "idle"
            st.session_state["zfov_measurements"] = []
            st.session_state["zfov_last_ref"] = None
            st.rerun()

# ─── Tab 3: Calibration (semi-manual) ─────────────────────────────────────────

with tab_calib:

    phase = st.session_state["calib_phase"]

    # ── Phase SETUP ──────────────────────────────────────────────────────────
    if phase == "setup":
        patrol_status_setup = client.get_patrol_status()
        patrol_running_setup = bool(patrol_status_setup and patrol_status_setup.get("patrol_running", False))

        st.markdown("### Patrol status")
        if patrol_status_setup is None:
            st.info("Patrol status is not exposed in direct mode. Make sure patrol is disabled before calibration.")
        elif patrol_running_setup:
            st.error(
                f"⚠️ Patrol is ACTIVE ({patrol_status_setup.get('loop_type', '?')}). "
                "Stop it before starting captures."
            )
        else:
            st.success("Patrol is stopped ✓")

        with st.expander("ℹ️ How it works", expanded=False):
            st.markdown("""
            **Step 1 — Automatic captures:**
            The app sends each pulse (direction, speed, duration T) and captures one image before
            and one image after. No interaction required, just wait.

            **Step 2 — Manual annotation:**
            For each before/after pair, click the **same landmark** (tree, pylon,
            building corner) in both images. Pixel displacement is converted to degrees
            using FOV.

            **Model:** `δ ≈ ω·T + b`
            - **ω** (°/s) = effective speed (line slope)
            - **b** (°) = negative offset due to start/stop inertia
            - **td** = -b/ω = effective launch delay (seconds)
            """)

        col_s1, col_s2 = st.columns(2)

        with col_s1:
            st.subheader("1. Preparation")

            if st.button("🛑 Stop patrol", key="btn_stop_patrol_calib"):
                client.stop_patrol()
                client.stop()
                if patrol_status_setup is None:
                    st.success("Stop command sent.")
                else:
                    stop_ph = st.empty()
                    for i in range(8):
                        time.sleep(1)
                        st_now = client.get_patrol_status()
                        if st_now is None or not st_now.get("patrol_running", False):
                            stop_ph.success(f"Patrol stopped after {i+1}s ✓")
                            break
                        stop_ph.warning(f"Waiting for patrol to stop... {i+1}s")
                    else:
                        stop_ph.error("Patrol is still active after 8s.")
                st.rerun()

            zoom_calib = st.slider(
                "Zoom (0 = wide angle - set before starting)",
                0, 64, st.session_state["calib_zoom"], key="zoom_calib_slider"
            )
            if st.button("Apply and lock zoom"):
                client.zoom(zoom_calib)
                time.sleep(2)
                st.session_state["calib_zoom"] = zoom_calib
                h_fov_z, v_fov_z = fov_at_zoom(zoom_calib, cam_model)
                st.success(f"Zoom={zoom_calib} → FOV H={h_fov_z:.1f}° V={v_fov_z:.1f}°")

            st.divider()
            st.write("**Home preset (recommended)**")
            st.caption("Camera returns to it before each pulse.")

            presets_live = client.get_presets()
            st.session_state["presets_list"] = presets_live
            preset_opts: Dict[str, int] = {
                f"[{p['id']}] {p.get('name', '')}".strip(): p["id"]
                for p in presets_live
            }
            home_options = ["(none)"] + list(preset_opts.keys())
            # _home_default stores the label to auto-select after saving a new preset
            _home_default = st.session_state.get("_home_default")
            _home_index = home_options.index(_home_default) if _home_default in home_options else 0
            home_label = st.selectbox(
                "Preset home",
                home_options,
                index=_home_index,
                key="home_preset_sel",
            )
            home_preset_id: Optional[int] = preset_opts.get(home_label)

            c1, c2 = st.columns(2)
            if c1.button("💾 Save here"):
                used_ids = {p["id"] for p in presets_live}
                new_id = next((i for i in range(1, 65) if i not in used_ids), None)
                if new_id is not None:
                    ok = client.set_preset(new_id, "calib_home")
                    if ok:
                        refreshed = client.get_presets()
                        st.session_state["presets_list"] = refreshed
                        # Find the label that matches the new preset ID
                        new_label = next(
                            (f"[{p['id']}] {p.get('name', '')}".strip()
                             for p in refreshed if p["id"] == new_id),
                            None,
                        )
                        if new_label:
                            st.session_state["_home_default"] = new_label
                        st.toast(f"Home → preset {new_id} saved ✓")
                    else:
                        st.toast("Failed to save preset", icon="❌")
                    st.rerun()
                else:
                    st.error("No free preset slot (all 64 are used)")
            if home_preset_id is not None and c2.button("📍 Go to home"):
                client.goto_preset(home_preset_id, speed=50)
                time.sleep(2)
                img_h = client.capture()
                if img_h:
                    st.session_state["live_img"] = img_h

        with col_s2:
            st.subheader("2. Parameters")

            axis = st.radio("Axis", ["pan", "tilt"], horizontal=True, key="calib_axis_sel")
            direction_fwd = "Right" if axis == "pan" else "Down"
            max_speed = 5 if axis == "pan" else 3

            speeds_to_calib: List[int] = st.multiselect(
                f"Speed levels (1-{max_speed})",
                list(range(1, max_speed + 1)),
                default=list(range(1, min(4, max_speed + 1))),
                key="calib_speeds_sel",
            )
            impulse_durations: List[float] = st.multiselect(
                "Pulse durations (s)",
                [0.15, 0.25, 0.4, 0.7, 1.0, 1.2, 2.0],
                default=DEFAULT_IMPULSE_DURATIONS,
                key="calib_durations_sel",
            )
            settle_time = st.slider(
                "Settle time after Stop (s)", 0.5, 5.0, 3.0, 0.5, key="settle_time"
            )

            total_pairs = len(speeds_to_calib) * len(impulse_durations)
            st.info(
                f"**{total_pairs} pairs** to capture and annotate "
                f"(axis={axis}, speeds={speeds_to_calib})"
            )

        st.divider()

        if not speeds_to_calib or len(impulse_durations) < 2:
            st.warning("Select at least 1 speed and 2 durations.")
        else:
            if st.button(
                "🚀 Start captures",
                type="primary",
                key="btn_start_calib",
                disabled=patrol_running_setup,
            ):
                # Verify patrol
                patrol_st = client.get_patrol_status()
                if patrol_st and patrol_st.get("patrol_running"):
                    st.error("Patrol is active. Stop it first, then start captures.")
                    st.stop()

                durations_sorted = sorted(impulse_durations)
                pairs: List[Dict] = []

                # Use the fixed zoom set in setup (no adaptive zoom —
                # Reolink cameras limit PTZ speed at high zoom levels)
                calib_z = st.session_state["calib_zoom"]
                h_fov_c, v_fov_c = fov_at_zoom(calib_z, cam_model)

                pb = st.progress(0.0)
                ph = st.empty()
                total_c = len(speeds_to_calib) * len(durations_sorted)
                done_c = 0

                for speed_c in speeds_to_calib:
                    for T_c in durations_sorted:
                        done_c += 1
                        pb.progress(done_c / total_c)

                        ph.markdown(
                            f"Capture **speed={speed_c}** T={T_c}s "
                            f"zoom={calib_z} FOV={h_fov_c:.1f}° ({done_c}/{total_c})..."
                        )

                        # Go to home preset first
                        if home_preset_id is not None:
                            client.goto_preset(home_preset_id, speed=50)
                            time.sleep(2.5)

                        # Re-apply calibration zoom after preset (preset resets zoom)
                        client.zoom(calib_z)
                        time.sleep(3.5)

                        img_b = client.capture()
                        if img_b is None:
                            ph.warning(f"Before capture failed (speed={speed_c}, T={T_c}s) - skipped")
                            continue

                        client.move(direction_fwd, speed=speed_c, duration=T_c)
                        time.sleep(settle_time)

                        img_a = client.capture()
                        if img_a is None:
                            ph.warning(f"After capture failed (speed={speed_c}, T={T_c}s) - skipped")
                            continue

                        # Auto-compute displacement via keypoint matching
                        deg_kp, n_matches, median_px = estimate_displacement_keypoints(
                            img_b, img_a, axis, h_fov_c, v_fov_c
                        )
                        # Also compute via optical flow for comparison
                        deg_of = abs(estimate_displacement_deg(
                            img_b, img_a, axis, h_fov_c, v_fov_c
                        ))

                        ph.markdown(
                            f"speed={speed_c} T={T_c}s → "
                            f"**keypoints: {deg_kp:.3f}°** ({n_matches} matches, {median_px:.0f}px) | "
                            f"optical flow: {deg_of:.3f}° ({done_c}/{total_c})"
                        )

                        if n_matches < 5:
                            ph.warning(f"Too few keypoint matches ({n_matches}) for speed={speed_c} T={T_c}s - keeping for manual review")

                        pairs.append({
                            "speed": speed_c,
                            "T": T_c,
                            "axis": axis,
                            "direction": direction_fwd,
                            "img_before": img_b,
                            "img_after": img_a,
                            "h_fov": h_fov_c,
                            "v_fov": v_fov_c,
                            "zoom": calib_z,
                            "auto_deg_kp": round(deg_kp, 4),
                            "auto_deg_of": round(deg_of, 4),
                            "auto_n_matches": n_matches,
                            "auto_px_delta": round(median_px, 1),
                        })

                if not pairs:
                    st.error("No pair captured - check the connection.")
                else:
                    pb.progress(1.0)
                    # Auto-populate annotations from keypoint results
                    auto_annotations = [
                        {
                            "speed": p["speed"],
                            "T": p["T"],
                            "axis": p["axis"],
                            "px_delta": p["auto_px_delta"],
                            "disp_deg": p["auto_deg_kp"],
                            "zoom": p["zoom"],
                            "h_fov": p["h_fov"],
                            "v_fov": p["v_fov"],
                            "method": "keypoints",
                            "n_matches": p["auto_n_matches"],
                        }
                        for p in pairs
                        if p["auto_n_matches"] >= 5
                    ]
                    skipped = len(pairs) - len(auto_annotations)
                    ph.success(
                        f"✅ {len(pairs)} pairs captured — "
                        f"{len(auto_annotations)} auto-measured, {skipped} need manual review"
                    )
                    st.session_state["calib_pairs"] = pairs
                    st.session_state["calib_annotations"] = auto_annotations
                    if skipped > 0:
                        # Go to manual annotation for failed ones
                        st.session_state["calib_anno_idx"] = 0
                        st.session_state["calib_click_before"] = None
                        st.session_state["calib_click_after"] = None
                        st.session_state["calib_phase"] = "annotating"
                    else:
                        st.session_state["calib_phase"] = "complete"
                    st.rerun()

    # ── Phase ANNOTATING ─────────────────────────────────────────────────────
    elif phase == "annotating":
        pairs = st.session_state["calib_pairs"]
        # Filter to only pairs that need manual annotation (low keypoint matches)
        manual_pairs = [
            (i, p) for i, p in enumerate(pairs)
            if p.get("auto_n_matches", 0) < 5
        ]
        idx = st.session_state["calib_anno_idx"]
        total_p = len(manual_pairs)

        if idx >= total_p:
            st.session_state["calib_phase"] = "complete"
            st.rerun()

        _orig_idx, pair = manual_pairs[idx]
        st.info(f"Manual annotation needed for {total_p} pairs (auto-detection had too few matches)")
        img_b: Image.Image = pair["img_before"]
        img_a: Image.Image = pair["img_after"]
        ax_p = pair["axis"]
        h_fov_p = pair["h_fov"]
        v_fov_p = pair["v_fov"]
        W, H = img_b.size

        st.markdown(
            f"### Annotation {idx + 1} / {total_p} — "
            f"**{ax_p.upper()}** speed={pair['speed']} T={pair['T']}s "
            f"direction={pair['direction']}"
        )
        st.caption(
            f"zoom={pair.get('zoom', '?')} | "
            f"H_FOV={h_fov_p:.3f}° | V_FOV={v_fov_p:.3f}° | "
            f"image {W}×{H}px | "
            f"pixel scale: {h_fov_p/W*1000:.3f} mrad/px"
        )
        st.progress((idx) / total_p)

        pt_b: Optional[Tuple[int, int]] = st.session_state["calib_click_before"]
        pt_a: Optional[Tuple[int, int]] = st.session_state["calib_click_after"]

        _CALIB_DISP_W = 900
        _calib_scale = _CALIB_DISP_W / W

        st.markdown("**BEFORE image movement - click the landmark**")
        display_b = draw_cross(img_b, pt_b[0], pt_b[1]) if pt_b else img_b
        display_b_small = display_b.resize((_CALIB_DISP_W, int(H * _calib_scale)), Image.LANCZOS)
        coord_b = streamlit_image_coordinates(display_b_small, key=f"click_b_{idx}")
        if coord_b is not None:
            new_b = (int(coord_b["x"] / _calib_scale), int(coord_b["y"] / _calib_scale))
            if new_b != pt_b:
                st.session_state["calib_click_before"] = new_b
                st.rerun()
        if pt_b:
            st.success(f"Selected point: ({pt_b[0]}, {pt_b[1]})")
        else:
            st.info("Click a landmark on this image")

        st.markdown("**AFTER image movement - click the same landmark**")
        W_a, H_a = img_a.size
        _calib_scale_a = _CALIB_DISP_W / W_a
        display_a = draw_cross(img_a, pt_a[0], pt_a[1], color="blue") if pt_a else img_a
        display_a_small = display_a.resize((_CALIB_DISP_W, int(H_a * _calib_scale_a)), Image.LANCZOS)
        coord_a = streamlit_image_coordinates(display_a_small, key=f"click_a_{idx}")
        if coord_a is not None:
            new_a = (int(coord_a["x"] / _calib_scale_a), int(coord_a["y"] / _calib_scale_a))
            if new_a != pt_a:
                st.session_state["calib_click_after"] = new_a
                st.rerun()
        if pt_a:
            st.success(f"Selected point: ({pt_a[0]}, {pt_a[1]})")
        else:
            st.info("Click the same landmark on this image")

        # Displacement preview
        if pt_b and pt_a:
            st.divider()
            if ax_p == "pan":
                # Camera -> right: landmark moves left
                px_delta = pt_b[0] - pt_a[0]
            else:
                # Camera -> up: landmark moves down (Down -> inverse)
                px_delta = pt_a[1] - pt_b[1]
            deg = abs(px_delta * (h_fov_p if ax_p == "pan" else v_fov_p) / (W if ax_p == "pan" else H))

            ref_map = REFERENCE_PAN_SPEEDS if ax_p == "pan" else REFERENCE_TILT_SPEEDS
            ref_bias_map = REFERENCE_PAN_BIAS if ax_p == "pan" else REFERENCE_TILT_BIAS
            ref_omega = ref_map.get(cam_model, {}).get(pair["speed"])
            ref_bias = ref_bias_map.get(cam_model, {}).get(pair["speed"], 0.0)
            ref_deg = ((ref_omega or 0) * pair["T"] + ref_bias) if ref_omega else 0.0

            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Δ pixels", f"{px_delta} px")
            c2.metric("Measured displacement", f"{deg:.3f}°")
            c3.metric("Expected (ref)", f"{ref_deg:.3f}°" if ref_omega else "—")
            c4.metric("Ratio", f"{deg/ref_deg:.2f}" if ref_deg else "—")
            c5.metric("Zoom / FOV", f"z{pair.get('zoom', '?')} / {h_fov_p:.1f}°")

            if abs(deg) < 0.05:
                st.warning(
                    "⚠️ Very small displacement - verify the camera did move "
                    "or choose a more precise landmark."
                )

            col_val, col_skip, col_redo = st.columns([2, 1, 1])
            if col_val.button("✅ Validate and next", type="primary", key=f"btn_val_{idx}"):
                st.session_state["calib_annotations"].append({
                    "speed": pair["speed"],
                    "T": pair["T"],
                    "axis": ax_p,
                    "px_delta": px_delta,
                    "disp_deg": round(deg, 4),
                    "zoom": pair.get("zoom"),
                    "h_fov": round(h_fov_p, 3),
                    "v_fov": round(v_fov_p, 3),
                    "pt_before": list(pt_b),
                    "pt_after": list(pt_a),
                })
                st.session_state["calib_anno_idx"] = idx + 1
                st.session_state["calib_click_before"] = None
                st.session_state["calib_click_after"] = None
                if idx + 1 >= total_p:
                    st.session_state["calib_phase"] = "complete"
                st.rerun()

            if col_skip.button("⏭️ Skip", key=f"btn_skip_{idx}"):
                st.session_state["calib_anno_idx"] = idx + 1
                st.session_state["calib_click_before"] = None
                st.session_state["calib_click_after"] = None
                if idx + 1 >= total_p:
                    st.session_state["calib_phase"] = "complete"
                st.rerun()

            if col_redo.button("🔄 Reset clicks", key=f"btn_redo_{idx}"):
                st.session_state["calib_click_before"] = None
                st.session_state["calib_click_after"] = None
                st.rerun()
        else:
            st.info("Click a landmark in both images to continue.")

    # ── Phase COMPLETE ────────────────────────────────────────────────────────
    elif phase == "complete":
        annotations = st.session_state["calib_annotations"]
        if not annotations:
            st.warning("No annotation - restart calibration.")
        else:
            st.success(f"✅ {len(annotations)} annotated measurements - model fitted")

            # Group by (axis, speed) and fit
            from collections import defaultdict
            groups: Dict[str, List[Dict]] = defaultdict(list)
            for ann in annotations:
                groups[f"{ann['axis']}_speed{ann['speed']}"].append(ann)

            for key_n, anns in groups.items():
                if len(anns) < 2:
                    continue
                ts = [a["T"] for a in anns]
                ds = [a["disp_deg"] for a in anns]
                model_r = fit_model(ts, ds)
                # Retrieve FOV from captured pairs
                pair_match = next(
                    (p for p in st.session_state["calib_pairs"]
                     if f"{p['axis']}_speed{p['speed']}" == key_n), None
                )
                h_fov_r = pair_match["h_fov"] if pair_match else H_FOV_WIDE
                v_fov_r = pair_match["v_fov"] if pair_match else V_FOV_WIDE
                zoom_r = pair_match["zoom"] if pair_match else 0
                axis_r = anns[0]["axis"]

                st.session_state["calib_results"][key_n] = {
                    "axis": axis_r,
                    "speed": anns[0]["speed"],
                    "omega": round(model_r["omega"], 4),
                    "bias": round(model_r["bias"], 4),
                    "r2": round(model_r["r2"], 4),
                    "zoom": zoom_r,
                    "h_fov": h_fov_r,
                    "v_fov": v_fov_r,
                }
                st.session_state["calib_raw_data"][key_n] = [
                    {
                        "duration": a["T"],
                        "displacement_deg": a["disp_deg"],
                        "px_delta": a.get("px_delta"),
                        "zoom": a.get("zoom"),
                        "h_fov": a.get("h_fov"),
                        "v_fov": a.get("v_fov"),
                        "pt_before": a.get("pt_before"),
                        "pt_after": a.get("pt_after"),
                    }
                    for a in anns
                ]
                td = -model_r["bias"] / model_r["omega"] if model_r["omega"] != 0 else 0
                st.write(
                    f"**{key_n}** : ω = {model_r['omega']:.4f} °/s | "
                    f"b = {model_r['bias']:.4f}° | "
                    f"delay td = {td:.3f}s | R² = {model_r['r2']:.3f}"
                )

            st.info("See the **Results & Export** tab for plots and export.")

        if st.button("🔄 New calibration"):
            st.session_state["calib_phase"] = "setup"
            st.session_state["calib_pairs"] = []
            st.session_state["calib_annotations"] = []
            st.session_state["calib_anno_idx"] = 0
            st.session_state["calib_click_before"] = None
            st.session_state["calib_click_after"] = None
            st.rerun()

    # ── Micro-pulse calibration ─────────────────────────────────────────────
    st.divider()
    st.subheader("⚡ Micro-pulse calibration")
    st.markdown(
        "Measure the minimum displacement from a brief impulse at **speed 1**. "
        "At high zoom, small angles fall below the calibrated bias — "
        "this tells you the actual nudge distance of a micro-impulse."
    )

    micro_phase = st.session_state["micro_phase"]

    if micro_phase == "idle":
        mc1, mc2, mc3, mc4 = st.columns(4)
        micro_axis = mc1.radio("Axis", ["pan", "tilt"], horizontal=True, key="micro_axis")
        micro_reps = mc2.number_input("Repetitions", 3, 20, 5, 1, key="micro_reps")
        micro_dur = mc3.number_input("Impulse duration (s)", 0.0, 0.5, 0.0, 0.01, key="micro_dur", format="%.2f")
        micro_zoom = mc4.number_input("Zoom (max=better precision)", 0, 64, 41, 1, key="micro_zoom")

        micro_settle = st.slider("Settle time after Stop (s)", 0.5, 5.0, 3.0, 0.5, key="micro_settle")

        h_fov_preview, v_fov_preview = fov_at_zoom(micro_zoom, cam_model)
        st.caption(
            f"At zoom {micro_zoom}: FOV H={h_fov_preview:.2f}° V={v_fov_preview:.2f}° — "
            f"pixel precision ~{h_fov_preview / 3840 * 1000:.2f} mrad/px (4K)"
        )

        # Reuse home preset from main calibration
        micro_home_label = st.session_state.get("home_preset_sel", "(none)")
        presets_micro = st.session_state.get("presets_list", [])
        micro_preset_map = {f"[{p['id']}] {p.get('name', '')}".strip(): p["id"] for p in presets_micro}
        micro_home_id: Optional[int] = micro_preset_map.get(micro_home_label)

        if micro_home_id is None:
            st.warning("Select a home preset in the calibration setup above before starting.")

        if st.button("🚀 Start micro-pulse captures", type="primary", key="btn_micro_start",
                      disabled=micro_home_id is None):
            # Apply zoom and wait for focus settle
            client.zoom(micro_zoom)
            st.session_state["calib_zoom"] = micro_zoom
            time.sleep(3.0)

            h_fov_m, v_fov_m = fov_at_zoom(micro_zoom, cam_model)
            direction_m = "Right" if micro_axis == "pan" else "Down"

            pairs_m: List[Dict] = []
            pb_m = st.progress(0.0)
            ph_m = st.empty()

            for rep in range(int(micro_reps)):
                pb_m.progress((rep) / int(micro_reps))
                ph_m.markdown(f"Micro-pulse **{rep + 1}/{int(micro_reps)}** …")

                if micro_home_id is not None:
                    client.goto_preset(micro_home_id, speed=50)
                    time.sleep(2.5)

                img_bm = client.capture()
                if img_bm is None:
                    ph_m.warning(f"Before capture failed (rep {rep + 1}) — skipped")
                    continue

                client.move(direction_m, speed=1, duration=micro_dur)
                time.sleep(micro_settle)

                img_am = client.capture()
                if img_am is None:
                    ph_m.warning(f"After capture failed (rep {rep + 1}) — skipped")
                    continue

                pairs_m.append({
                    "img_before": img_bm,
                    "img_after": img_am,
                    "h_fov": h_fov_m,
                    "v_fov": v_fov_m,
                    "axis": micro_axis,
                    "direction": direction_m,
                    "rep": rep + 1,
                    "impulse_dur": micro_dur,
                    "zoom": micro_zoom,
                })

            if not pairs_m:
                st.error("No pair captured.")
            else:
                pb_m.progress(1.0)
                ph_m.success(f"✅ {len(pairs_m)} micro-pulse pairs — starting annotation…")
                st.session_state["micro_pairs"] = pairs_m
                st.session_state["micro_anno_idx"] = 0
                st.session_state["micro_click_before"] = None
                st.session_state["micro_click_after"] = None
                st.session_state["micro_annotations"] = []
                st.session_state["micro_phase"] = "annotating"
                st.rerun()

    elif micro_phase == "annotating":
        m_pairs = st.session_state["micro_pairs"]
        m_idx = st.session_state["micro_anno_idx"]
        m_total = len(m_pairs)

        if m_idx >= m_total:
            st.session_state["micro_phase"] = "complete"
            st.rerun()

        mp = m_pairs[m_idx]
        m_img_b: Image.Image = mp["img_before"]
        m_img_a: Image.Image = mp["img_after"]
        m_ax = mp["axis"]
        m_h_fov = mp["h_fov"]
        m_v_fov = mp["v_fov"]
        m_W, m_H = m_img_b.size

        st.markdown(
            f"### Micro-pulse {m_idx + 1} / {m_total} — "
            f"**{m_ax.upper()}** rep={mp['rep']} dur={mp['impulse_dur']}s"
        )
        st.progress(m_idx / m_total)

        m_pt_b: Optional[Tuple[int, int]] = st.session_state["micro_click_before"]
        m_pt_a: Optional[Tuple[int, int]] = st.session_state["micro_click_after"]

        _M_DISP_W = 900
        _m_scale = _M_DISP_W / m_W

        st.markdown("**BEFORE — click a landmark**")
        m_disp_b = draw_cross(m_img_b, m_pt_b[0], m_pt_b[1]) if m_pt_b else m_img_b
        m_disp_b_s = m_disp_b.resize((_M_DISP_W, int(m_H * _m_scale)), Image.LANCZOS)
        m_coord_b = streamlit_image_coordinates(m_disp_b_s, key=f"micro_b_{m_idx}")
        if m_coord_b is not None:
            m_new_b = (int(m_coord_b["x"] / _m_scale), int(m_coord_b["y"] / _m_scale))
            if m_new_b != m_pt_b:
                st.session_state["micro_click_before"] = m_new_b
                st.rerun()
        if m_pt_b:
            st.success(f"Selected: ({m_pt_b[0]}, {m_pt_b[1]})")

        st.markdown("**AFTER — click the same landmark**")
        m_W_a, m_H_a = m_img_a.size
        _m_scale_a = _M_DISP_W / m_W_a
        m_disp_a = draw_cross(m_img_a, m_pt_a[0], m_pt_a[1], color="blue") if m_pt_a else m_img_a
        m_disp_a_s = m_disp_a.resize((_M_DISP_W, int(m_H_a * _m_scale_a)), Image.LANCZOS)
        m_coord_a = streamlit_image_coordinates(m_disp_a_s, key=f"micro_a_{m_idx}")
        if m_coord_a is not None:
            m_new_a = (int(m_coord_a["x"] / _m_scale_a), int(m_coord_a["y"] / _m_scale_a))
            if m_new_a != m_pt_a:
                st.session_state["micro_click_after"] = m_new_a
                st.rerun()
        if m_pt_a:
            st.success(f"Selected: ({m_pt_a[0]}, {m_pt_a[1]})")

        if m_pt_b and m_pt_a:
            st.divider()
            if m_ax == "pan":
                # Camera right → landmark moves left → before_x > after_x
                m_px_delta = m_pt_b[0] - m_pt_a[0]
                m_deg = abs(m_px_delta * m_h_fov / m_W)
            else:
                # Camera down → landmark moves up → before_y > after_y
                m_px_delta = m_pt_b[1] - m_pt_a[1]
                m_deg = abs(m_px_delta * m_v_fov / m_H)

            mc1, mc2 = st.columns(2)
            mc1.metric("Δ pixels", f"{m_px_delta} px")
            mc2.metric("Displacement", f"{m_deg:.3f}°")

            mcv, mcs, mcr = st.columns([2, 1, 1])
            if mcv.button("✅ Validate", type="primary", key=f"micro_val_{m_idx}"):
                st.session_state["micro_annotations"].append({
                    "disp_deg": round(m_deg, 4),
                    "axis": m_ax,
                })
                st.session_state["micro_anno_idx"] = m_idx + 1
                st.session_state["micro_click_before"] = None
                st.session_state["micro_click_after"] = None
                if m_idx + 1 >= m_total:
                    st.session_state["micro_phase"] = "complete"
                st.rerun()
            if mcs.button("⏭️ Skip", key=f"micro_skip_{m_idx}"):
                st.session_state["micro_anno_idx"] = m_idx + 1
                st.session_state["micro_click_before"] = None
                st.session_state["micro_click_after"] = None
                if m_idx + 1 >= m_total:
                    st.session_state["micro_phase"] = "complete"
                st.rerun()
            if mcr.button("🔄 Reset", key=f"micro_reset_{m_idx}"):
                st.session_state["micro_click_before"] = None
                st.session_state["micro_click_after"] = None
                st.rerun()

    elif micro_phase == "complete":
        m_anns = st.session_state["micro_annotations"]
        if not m_anns:
            st.warning("No measurements — restart.")
        else:
            m_vals = [a["disp_deg"] for a in m_anns]
            m_mean = float(np.mean(m_vals))
            m_std = float(np.std(m_vals))
            m_axis_label = m_anns[0]["axis"]

            st.success(f"✅ {len(m_vals)} micro-pulse measurements")

            r1, r2, r3 = st.columns(3)
            r1.metric("Mean displacement", f"{m_mean:.3f}°")
            r2.metric("Std deviation", f"{m_std:.3f}°")
            r3.metric("Range", f"{min(m_vals):.3f}° — {max(m_vals):.3f}°")

            # Store in calib_results for export
            micro_key = f"{m_axis_label}_micro"
            micro_pairs_done = st.session_state["micro_pairs"]
            micro_zoom_used = micro_pairs_done[0].get("zoom", st.session_state["calib_zoom"]) if micro_pairs_done else st.session_state["calib_zoom"]
            micro_h_fov = micro_pairs_done[0]["h_fov"] if micro_pairs_done else 0
            micro_v_fov = micro_pairs_done[0]["v_fov"] if micro_pairs_done else 0
            st.session_state["calib_results"][micro_key] = {
                "axis": m_axis_label,
                "speed": 1,
                "omega": 0.0,
                "bias": round(m_mean, 4),
                "r2": 0.0,
                "zoom": micro_zoom_used,
                "h_fov": micro_h_fov,
                "v_fov": micro_v_fov,
                "micro_pulse": True,
                "micro_mean_deg": round(m_mean, 4),
                "micro_std_deg": round(m_std, 4),
                "micro_n": len(m_vals),
            }

            # Individual measurements
            with st.expander("Individual measurements"):
                for i, v in enumerate(m_vals):
                    st.write(f"Rep {i + 1}: {v:.4f}°")

            st.caption(
                f"**Update `routes_control.py`:** set `_MICRO_IMPULSE_DUR` and expect "
                f"~{m_mean:.2f}° ± {m_std:.2f}° per micro-pulse at speed 1 for {cam_model}."
            )

        if st.button("🔄 New micro-pulse calibration", key="btn_micro_restart"):
            st.session_state["micro_phase"] = "idle"
            st.session_state["micro_pairs"] = []
            st.session_state["micro_annotations"] = []
            st.session_state["micro_anno_idx"] = 0
            st.session_state["micro_click_before"] = None
            st.session_state["micro_click_after"] = None
            st.rerun()

# ─── Tab 3: Results ───────────────────────────────────────────────────────────

with tab_results:
    results = st.session_state["calib_results"]
    raw_data = st.session_state["calib_raw_data"]

    if not results:
        st.info("No result available. Run calibration first.")
    else:
        st.subheader("Summary table")

        rows = []
        for key, r in results.items():
            ref_map = REFERENCE_PAN_SPEEDS if r["axis"] == "pan" else REFERENCE_TILT_SPEEDS
            ref_bias_map = REFERENCE_PAN_BIAS if r["axis"] == "pan" else REFERENCE_TILT_BIAS
            ref_omega = ref_map.get(cam_model, {}).get(r["speed"])
            ref_bias = ref_bias_map.get(cam_model, {}).get(r["speed"])
            delta_pct = ((r["omega"] - ref_omega) / ref_omega * 100) if ref_omega else None
            delta_bias = (r["bias"] - ref_bias) if ref_bias is not None else None
            rows.append({
                "Key": key,
                "Axis": r["axis"],
                "Speed": r["speed"],
                "ω measured (°/s)": r["omega"],
                "ω reference (°/s)": ref_omega,
                "Δ vs ref (%)": round(delta_pct, 1) if delta_pct is not None else "—",
                "Bias b (°)": r["bias"],
                "b reference (°)": ref_bias,
                "Δb vs ref (°)": round(delta_bias, 3) if delta_bias is not None else "—",
                "R²": r["r2"],
                "Zoom": r["zoom"],
            })

        df = pd.DataFrame(rows)
        st.dataframe(df, width="stretch", hide_index=True)

        # ── Plot ────────────────────────────────────────────────────────────
        st.subheader("Fit plot")
        key_sel = st.selectbox("Select a measurement", list(results.keys()))

        if key_sel in raw_data:
            rd = raw_data[key_sel]
            r = results[key_sel]
            ts = [m["duration"] for m in rd]
            ds = [m["displacement_deg"] for m in rd]

            fig, ax = plt.subplots(figsize=(9, 4))
            ax.scatter(ts, ds, color="steelblue", s=80, zorder=5, label="Measurements (optical flow)")

            t_fit = np.linspace(0, max(ts) * 1.15, 200)
            d_fit = r["omega"] * t_fit + r["bias"]
            ax.plot(t_fit, d_fit, "r-", lw=2,
                    label=f"Model: δ = {r['omega']:.4f}·T + {r['bias']:.4f}°   (R²={r['r2']:.3f})")

            ref_map = REFERENCE_PAN_SPEEDS if r["axis"] == "pan" else REFERENCE_TILT_SPEEDS
            ref_bias_map = REFERENCE_PAN_BIAS if r["axis"] == "pan" else REFERENCE_TILT_BIAS
            ref_omega = ref_map.get(cam_model, {}).get(r["speed"])
            ref_bias = ref_bias_map.get(cam_model, {}).get(r["speed"], 0.0)
            if ref_omega:
                d_ref = ref_omega * t_fit + ref_bias
                ax.plot(t_fit, d_ref, "g--", lw=1.5, alpha=0.7,
                        label=f"Current reference: δ = {ref_omega}·T + {ref_bias:.4f}°")

            ax.axhline(0, color="gray", lw=0.5)
            ax.set_xlabel("Duration T (s)")
            ax.set_ylabel("Measured displacement (°)")
            ax.set_title(f"{key_sel}  —  {r['axis'].upper()}  speed={r['speed']}  zoom={r['zoom']}")
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            plt.close(fig)

            # Interpretation
            td_eff = -r["bias"] / r["omega"] if r["omega"] != 0 else 0
            st.info(
                f"**Interpretation:** effective delay td = {td_eff:.3f}s  |  "
                f"To target X°, command time T ≈ (X - {r['bias']:.3f}) / {r['omega']:.4f}"
            )

        st.divider()

        # ── Test move_by_degrees ────────────────────────────────────────────
        st.subheader("Test : move_by_degrees")
        col_t1, col_t2 = st.columns(2)
        with col_t1:
            test_axis = st.radio("Axis", ["pan", "tilt"], horizontal=True, key="test_axis")
            test_speed = st.number_input("Speed", 1, 5, 1, key="test_speed")
            test_deg = st.number_input("Target degrees", 0.5, 30.0, 5.0, 0.5, key="test_deg")
        with col_t2:
            test_key = f"{test_axis}_speed{test_speed}"
            if test_key in results:
                r_t = results[test_key]
                T_cmd = (test_deg - r_t["bias"]) / r_t["omega"] if r_t["omega"] != 0 else 0
                st.metric("Computed duration (s)", f"{T_cmd:.3f}")
                st.caption(f"Formula: T = ({test_deg} - {r_t['bias']:.3f}) / {r_t['omega']:.4f}")
                if T_cmd <= 0:
                    st.warning(f"Target {test_deg}° < bias {r_t['bias']:.2f}° — cannot execute at this speed.")
                elif st.button("▶️ Run", key="btn_test_move"):
                    direction_test = "Right" if test_axis == "pan" else "Down"
                    img_before_t = client.capture()
                    client.move(direction_test, speed=test_speed, duration=T_cmd)
                    time.sleep(1.5)
                    img_after_t = client.capture()
                    if img_before_t and img_after_t:
                        h_fov_t, v_fov_t = fov_at_zoom(r_t["zoom"], cam_model)
                        actual = estimate_displacement_deg(img_before_t, img_after_t, test_axis, h_fov_t, v_fov_t)
                        st.metric("Measured real displacement (°)", f"{actual:.3f}°",
                                  delta=f"{actual - test_deg:+.3f}° vs target")
                        col_b1, col_b2 = st.columns(2)
                        col_b1.image(img_before_t, caption="Before", width="stretch")
                        col_b2.image(img_after_t, caption="After", width="stretch")
            else:
                st.warning(f"No calibration yet for {test_key}")

        st.divider()

        # ── Export ──────────────────────────────────────────────────────────
        st.subheader("Export")
        col_e1, col_e2 = st.columns(2)

        with col_e1:
            export_obj = {
                "model": cam_model,
                "calibrated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "results": results,
                "raw_data": raw_data,
            }
            st.download_button(
                "⬇️ Download JSON",
                data=json.dumps(export_obj, indent=2),
                file_name=f"ptz_calib_{cam_model}_{time.strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
            )

        with col_e2:
            pan_speeds_new = {
                r["speed"]: r["omega"]
                for r in results.values() if r["axis"] == "pan"
            }
            tilt_speeds_new = {
                r["speed"]: r["omega"]
                for r in results.values() if r["axis"] == "tilt"
            }
            bias_notes = "\n".join(
                f'# {k}: b={r["bias"]:.4f}° (effective delay ≈ {-r["bias"]/r["omega"]:.3f}s)'
                for k, r in results.items() if r["omega"] != 0
            )
            snippet = (
                f"# ── Update routes_control.py ──────────────────────────────────\n"
                f'PAN_SPEEDS["{cam_model}"] = {json.dumps(pan_speeds_new)}\n'
                f'TILT_SPEEDS["{cam_model}"] = {json.dumps(tilt_speeds_new)}\n\n'
                f"# Measured bias values (term b - integrate into move_by_degrees):\n"
                f"{bias_notes}\n"
            )
            st.code(snippet, language="python")
            st.caption(
                "Copy/paste into "
                "`pyro_camera_api/pyro_camera_api/api/routes_control.py`"
            )

# ─── Tab 4 : Presets ─────────────────────────────────────────────────────────

with tab_presets:
    st.subheader("PTZ preset management")

    col_p1, col_p2 = st.columns(2)

    with col_p1:
        st.write("**Available presets**")
        if st.button("🔄 Refresh", key="btn_refresh_presets"):
            st.session_state["presets_list"] = client.get_presets()

        presets = st.session_state["presets_list"]
        if not presets:
            st.session_state["presets_list"] = client.get_presets()
            presets = st.session_state["presets_list"]

        if presets:
            for p in presets:
                c1, c2, c3 = st.columns([3, 1, 1])
                c1.write(f"**[{p['id']}]** {p.get('name', '—')}")
                if c2.button("📍 Go", key=f"goto_{p['id']}"):
                    client.goto_preset(p["id"], speed=50)
                    time.sleep(2)
                    img = client.capture()
                    if img is not None:
                        st.session_state["live_img"] = img
                    st.success(f"Moved to preset {p['id']}")
        else:
            st.info("No active preset found.")

    with col_p2:
        st.write("**Save current position**")
        new_id = st.number_input("ID (1–64)", 1, 64, 1, key="new_preset_id")
        new_name = st.text_input("Name", value=f"pos{new_id}", key="new_preset_name")
        if st.button("💾 Save", key="btn_save_preset"):
            ok = client.set_preset(int(new_id), new_name)
            if ok:
                st.success(f"Preset {new_id} ('{new_name}') saved")
                st.session_state["presets_list"] = client.get_presets()
                st.rerun()
            else:
                st.error("Save failed")
