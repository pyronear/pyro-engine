# PTZ Calibration Tools

## Overview

The PTZ calibration app measures the relationship between movement commands and actual angular displacement for Reolink PTZ cameras. It produces speed tables used by `routes_control.py` for precise click-to-move and patrol operations.

**Motor model:** `displacement = omega * T + bias`
- `omega` = angular velocity (deg/s)
- `T` = commanded duration (s)
- `bias` = coast distance from mechanical inertia (deg)
- At T=0 (micro-pulse), displacement = bias

## Prerequisites

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) if you don't already have it:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Install dependencies and run the app (uv reads `tools/pyproject.toml`):

```bash
cd tools
uv sync
```

The camera must be accessible either directly (HTTP) or via the Camera API service.

## Launch

```bash
cd tools
uv run streamlit run ptz_calibration_app.py
```

Open a second instance on a different port to calibrate another camera in parallel:
```bash
uv run streamlit run ptz_calibration_app.py --server.port 8502
```

## Step-by-step Calibration

### 1. Connect to the camera

- Select **Direct** (camera IP + credentials) or **API** (Camera API URL)
- Verify the live preview shows the correct camera
- Select the correct **camera model** (reolink-823A16 or reolink-823S2)

### 2. Preparation (Calibration tab)

- Click **Stop patrol** to ensure the camera is idle
- Move the camera to a good home position with visible distant landmarks (pylons, buildings, trees)
- Click **Save here** to save the current position as a home preset
- Select the saved preset in the **Preset home** dropdown

### 3. Calibrate speeds 2-5 at zoom 0

This gives the full speed table for wide-angle patrol moves.

- Set **Zoom mode** to "Single zoom"
- Set zoom slider to **0**, click **Apply and lock zoom**
- Select **Axis**: pan
- Select **Speed levels**: 2, 3, 4, 5
- Select **Pulse durations**: 0.25, 0.4, 0.7, 1.2, 2.0 (at least 4 values)
- Set **Settle time**: 3.0s
- Click **Start captures**

The app automatically captures before/after images, computes displacement via ORB keypoint matching, and fits the affine model. No manual annotation needed at zoom 0.

### 4. Calibrate speed 1 at zoom 41

Speed 1 is the only speed that works at all zoom levels. Calibrating at max zoom gives the best precision.

- Click **New calibration**
- Set zoom slider to **41**, click **Apply and lock zoom**
- Select **Speed levels**: 1
- Select **Pulse durations**: 0.15, 0.25, 0.4, 0.7, 1.0, 1.2 (avoid 2.0s at high zoom)
- Click **Start captures**

### 5. Calibrate micro-pulse

A micro-pulse is a T=0 move. Its displacement equals the bias at speed 1. Measuring it directly at zoom 41 gives the most precise value.

- Scroll down to **Micro-pulse calibration**
- Set **Axis**: pan (then repeat for tilt)
- Set **Repetitions**: 5-10
- Set **Impulse duration**: 0.0
- Set **Zoom**: 41
- Click **Start micro-pulse captures**

At zoom 41, keypoint matching may not work reliably — manual annotation (clicking the same landmark in before/after images) is used as fallback.

### 6. Review and export

Go to the **Results & Export** tab:
- Check R2 > 0.98 for all speeds
- Download the JSON with results

### 7. Update routes_control.py

Update the speed/bias tables in `pyro_camera_api/pyro_camera_api/api/routes_control.py`. Both adapter keys (`reolink-823S2` and `reolink-823A16`) and both axes (pan + tilt) must be kept in sync:

```python
PAN_SPEEDS = {
    "reolink-823S2":  {1: <omega>, 2: <omega>, 3: <omega>, 4: <omega>, 5: <omega>},
    "reolink-823A16": {1: <omega>, 2: <omega>, 3: <omega>, 4: <omega>, 5: <omega>},
}
PAN_BIAS = {
    "reolink-823S2":  {1: <bias>, 2: <bias>, 3: <bias>, 4: <bias>, 5: <bias>},
    "reolink-823A16": {1: <bias>, 2: <bias>, 3: <bias>, 4: <bias>, 5: <bias>},
}
TILT_SPEEDS = {
    "reolink-823S2":  {1: <omega>, 2: <omega>, 3: <omega>},
    "reolink-823A16": {1: <omega>, 2: <omega>, 3: <omega>},
}
TILT_BIAS = {
    "reolink-823S2":  {1: <bias>, 2: <bias>, 3: <bias>},
    "reolink-823A16": {1: <bias>, 2: <bias>, 3: <bias>},
}
```

All values come from zoom 0 calibration. The `_pick_speed()` function automatically restricts to speed 1 when zoom > 0.

### 8. Calibrate the FOV table (Zoom FOV tab)

The camera API interprets click-to-move clicks using a per-zoom FOV lookup. Recalibrate whenever a new camera model is added or optics change.

- Place a QR code at a fixed distance in front of the camera
- Open the **Zoom FOV Calibration** tab
- Set zoom range (default 0–64, step 1) and click **Start FOV calibration**
- For each zoom level the app captures a snapshot, detects the QR, and computes FOV by chained ratio from zoom 0 (anchored by `H_FOV_WIDE` / `V_FOV_WIDE`)
- Export the resulting `h_fov` / `v_fov` arrays (42 values each) into `pyro_camera_api/pyro_camera_api/api/fov_tables.json` under the right adapter key

### 9. Test with click-to-move

- Use `livestream_app.py` (see below) or any client that calls `/control/click_to_move`
- Click on different points in the image at zoom 0 and zoomed in
- Verify the camera centers on the clicked point
- At zoom 0, higher speeds should be used for large moves
- At zoom > 0, only speed 1 is used

Repeat steps 3-5 for tilt axis (3 speed levels instead of 5).

## Important Notes

### Zoom-speed limitation

Reolink cameras internally cap PTZ speed when zoomed in. All speeds collapse to ~1.5 deg/s at zoom > 0. Only speed 1 is useful when zoomed in. See `ptz_zoom_speed_calibration_report.md` for research data.

### Server-side timing

All move commands use the `duration` parameter which executes move+sleep+stop on the Pi's local network. This eliminates VPN latency from measurements.

## Livestream / click-to-move app

`livestream_app.py` is a small Streamlit companion tool for operators:

- stops the current patrol
- starts the public RTSP→HLS live stream and embeds it as an iframe
- on a fresh snapshot, click anywhere to recenter the camera via `/control/click_to_move`

```bash
cd tools
uv run streamlit run livestream_app.py
```

It talks to the Camera API only (no direct HTTP to the camera) and uses the same calibrated speed + FOV tables the PTZ routes rely on.

## Files

- `ptz_calibration_app.py` — Streamlit app for PTZ speed/bias and zoom-FOV calibration
- `livestream_app.py` — Streamlit livestream + click-to-move operator tool
- `ptz_zoom_speed_calibration_report.md` — Calibration results and zoom-speed research
- `pyproject.toml` / `uv.lock` — `uv sync` dependency spec for the tools
