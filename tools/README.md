# PTZ Calibration Tools

## Overview

The PTZ calibration app measures the relationship between movement commands and actual angular displacement for Reolink PTZ cameras. It produces speed tables used by `routes_control.py` for precise click-to-move and patrol operations.

**Motor model:** `displacement = omega * T + bias`
- `omega` = angular velocity (deg/s)
- `T` = commanded duration (s)
- `bias` = coast distance from mechanical inertia (deg)
- At T=0 (micro-pulse), displacement = bias

## Prerequisites

```bash
pip install streamlit opencv-python-headless pillow requests pandas matplotlib streamlit-image-coordinates
```

The camera must be accessible either directly (HTTP) or via the Camera API service.

## Launch

```bash
streamlit run tools/ptz_calibration_app.py
```

Open a second instance on a different port to calibrate another camera in parallel:
```bash
streamlit run tools/ptz_calibration_app.py --server.port 8502
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

Update the speed tables in `pyro_camera_api/pyro_camera_api/api/routes_control.py`:

```python
PAN_SPEEDS = {
    "reolink-823A16": {1: <omega>, 2: <omega>, 3: <omega>, 4: <omega>, 5: <omega>},
}
PAN_BIAS = {
    "reolink-823A16": {1: <bias>, 2: <bias>, 3: <bias>, 4: <bias>, 5: <bias>},
}
```

All values come from zoom 0 calibration. The `_pick_speed()` function automatically restricts to speed 1 when zoom > 0.

### 8. Test with click-to-move

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

## Files

- `ptz_calibration_app.py` — Streamlit calibration app
- `ptz_zoom_speed_calibration_report.md` — Calibration results and zoom-speed research
