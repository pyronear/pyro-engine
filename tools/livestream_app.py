# Copyright (C) 2022-2026, Pyronear.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

"""
Pyronear Livestream + Click-to-Move app.

Small Streamlit tool to:
  - stop the current patrol
  - start the live stream
  - preview the public livestream page (iframe)
  - click on a fresh snapshot to recenter the camera via click_to_move

Run:
    uv run streamlit run tools/livestream_app.py
"""

from __future__ import annotations

from typing import Optional

import streamlit as st
from PIL import Image
from pyro_camera_api_client import PyroCameraAPIClient
from streamlit_image_coordinates import streamlit_image_coordinates

st.set_page_config(page_title="Pyro Livestream", page_icon="📡", layout="wide")
st.title("📡 Pyronear Livestream — Click to Move")


# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Connection")
    api_url = st.text_input("Camera API URL", value="http://192.168.255.62:8081")
    stream_url = st.text_input("Livestream URL", value="https://livestream.pyronear.org/st-peray/")

    if st.button("🔍 List cameras"):
        try:
            data = PyroCameraAPIClient(api_url, timeout=5.0).list_cameras()
            cams = data.get("camera_ids", []) if isinstance(data, dict) else list(data)
            st.session_state["cam_list"] = cams
            st.success(f"{len(cams)} camera(s) found")
        except Exception as exc:
            st.error(f"list_cameras failed: {exc}")

    cam_list = st.session_state.get("cam_list", [])
    camera_ip = (
        st.selectbox("Camera", cam_list, key="cam_sel") if cam_list else st.text_input("Camera IP (manual)", value="")
    )

if not camera_ip:
    st.info("Pick or type a camera IP in the sidebar.")
    st.stop()

api = PyroCameraAPIClient(api_url, timeout=60.0)


# ─── Controls ────────────────────────────────────────────────────────────────
c1, c2, c3 = st.columns([2, 1, 1])

if c1.button("⏸ Stop patrol + ▶️ Start stream", type="primary"):
    try:
        api.stop_patrol(camera_ip)
    except Exception as exc:
        # 404 if no patrol running is fine
        st.caption(f"stop_patrol: {exc}")
    try:
        res = api.start_stream(camera_ip)
        st.toast(f"stream started ({res.get('mode', '?')})")
    except Exception as exc:
        st.error(f"start_stream: {exc}")

if c2.button("⏹ Stop stream"):
    try:
        api.stop_stream()
        st.toast("stream stopped")
    except Exception as exc:
        st.error(f"stop_stream: {exc}")

if c3.button("🔄 Refresh snapshot"):
    # bump nonce so streamlit_image_coordinates reinitialises
    st.session_state["snap_nonce"] = st.session_state.get("snap_nonce", 0) + 1


# ─── Zoom ────────────────────────────────────────────────────────────────────
z1, z2 = st.columns([4, 1])
zoom_level = z1.slider("Zoom (0 = wide, 41 = tele)", 0, 41, st.session_state.get("zoom_level", 0))
if z2.button("Apply zoom"):
    try:
        res = api.zoom(camera_ip, zoom_level)
        st.session_state["zoom_level"] = zoom_level
        st.toast(f"zoom → {zoom_level} (settled {res.get('settle', '?')}s)")
        st.session_state["snap_nonce"] = st.session_state.get("snap_nonce", 0) + 1
        st.rerun()
    except Exception as exc:
        msg = str(exc)
        if "409" in msg:
            st.warning("Camera busy — let the current move finish, then retry.")
        else:
            st.error(f"zoom: {exc}")


# ─── Live stream preview (read-only) ─────────────────────────────────────────
st.markdown("### 🎥 Live stream")
st.caption(
    "Low-latency HLS/SRT preview. Clicks on this iframe cannot reach the server (cross-origin) — "
    "use the snapshot below to aim."
)
st.iframe(stream_url, height=400)


# ─── Clickable snapshot → click_to_move ──────────────────────────────────────
st.markdown("### 🎯 Click to aim")
st.caption("Click a point on the snapshot to recenter the camera on it.")

try:
    img: Optional[Image.Image] = api.capture_image(camera_ip, anonymize=False)
except Exception as exc:
    st.error(f"capture failed: {exc}")
    img = None

if img is not None:
    W, H = img.size
    DISP_W = 1000
    scale = DISP_W / W
    disp_h = int(H * scale)
    small = img.resize((DISP_W, disp_h), Image.LANCZOS)

    nonce = st.session_state.get("snap_nonce", 0)
    coord = streamlit_image_coordinates(small, key=f"click_stream_{nonce}")

    last = st.session_state.get("last_click")
    if coord is not None and coord != last:
        st.session_state["last_click"] = coord
        nx = float(coord["x"]) / DISP_W
        ny = float(coord["y"]) / disp_h
        with st.spinner(f"click_to_move at ({nx:.3f}, {ny:.3f})…"):
            try:
                res = api.click_to_move(camera_ip, nx, ny)
                moves = res.get("moves", [])
                st.success(f"zoom={res.get('zoom')}  moves={moves}")
            except Exception as exc:
                msg = str(exc)
                if "409" in msg:
                    st.warning("Camera busy — let the current move finish, then retry.")
                else:
                    st.error(f"click_to_move: {exc}")
        # Force a fresh snapshot + reset the click widget on next run
        st.session_state["snap_nonce"] = nonce + 1
        st.rerun()
