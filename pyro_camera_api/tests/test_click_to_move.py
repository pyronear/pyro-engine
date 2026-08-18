# Copyright (C) 2022-2026, Pyronear.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.


from pyro_camera_api.api.routes_control import click_to_move
from pyro_camera_api.camera.adapters.mock import MockCamera
from pyro_camera_api.camera.registry import CAMERA_REGISTRY


def test_click_to_move_uncalibrated_adapter_moves(monkeypatch):
    # Regression: adapters without calibrated speed tables (e.g. linovision)
    # must fall back to the "speed≈°/s" proxy instead of skipping the move.
    ip = "203.0.113.9"
    cam = MockCamera(camera_id=ip, cam_type="ptz", cam_poses=[0], cam_azimuths=[0])
    calls = []
    monkeypatch.setattr(cam, "move_camera", lambda op, speed=20, _idx=0: calls.append((op, speed)))
    monkeypatch.setitem(CAMERA_REGISTRY, ip, cam)

    result = click_to_move(camera_ip=ip, click_x=0.55, click_y=0.5)

    assert result["status"] == "ok"
    assert result["moves"], "expected at least one move"
    assert all("skipped" not in m for m in result["moves"])
    assert ("Right", result["moves"][0]["speed"]) in calls
    assert ("Stop", 20) in calls


class _RelativeMockCamera(MockCamera):
    """Mimics the linovision adapter: hardware relative moves + PTZ status."""

    def get_ptz_status(self):
        return {"azimuth_deg": 100.0, "elevation_deg": 5.0, "zoom_raw": 20, "zoom_ratio": 2.0}

    def move_relative_deg(self, delta_azimuth_deg, delta_elevation_deg=0.0):
        self.last_relative = (delta_azimuth_deg, delta_elevation_deg)
        return {"azimuth_deg": 100.0 + delta_azimuth_deg, "elevation_deg": 5.0 + delta_elevation_deg}


def test_click_to_move_relative_hardware_path(monkeypatch):
    ip = "203.0.113.10"
    cam = _RelativeMockCamera(camera_id=ip, cam_type="ptz", cam_poses=[0], cam_azimuths=[0])
    monkeypatch.setitem(CAMERA_REGISTRY, ip, cam)

    # Click right of center and above center.
    result = click_to_move(camera_ip=ip, click_x=0.75, click_y=0.25)

    assert result["status"] == "ok"
    assert result["zoom_ratio"] == 2.0
    # FOV must shrink with zoom: at 2x it is well below the wide-end value.
    assert result["h_fov"] < 54.2 / 1.5
    d_az, d_el = cam.last_relative
    assert d_az > 0  # click right → pan right → azimuth increases
    assert d_el > 0  # click above center → camera looks up → elevation increases
    assert result["moves"][0]["mode"] == "relative"
