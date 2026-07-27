# Copyright (C) 2022-2026, Pyronear.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.


import pytest

from pyro_camera_api.api.routes_control import _shortest_delta_deg, _update_tracked_azimuth
from pyro_camera_api.camera.adapters.mock import MockCamera


def _ptz_mock() -> MockCamera:
    return MockCamera(
        camera_id="mock",
        cam_type="ptz",
        cam_poses=[0, 1, 2],
        cam_azimuths=[0, 90, 180],
    )


def test_azimuth_unknown_at_start():
    cam = _ptz_mock()
    assert cam.get_azimuth() is None
    assert cam.azimuth_source == "tracked"


def test_topos_sets_azimuth_from_pose_mapping():
    cam = _ptz_mock()
    cam.move_camera("ToPos", idx=1)
    assert cam.get_azimuth() == 90.0
    cam.move_camera("ToPos", idx=2)
    assert cam.get_azimuth() == 180.0


def test_topos_unmapped_pose_invalidates_azimuth():
    # An accepted preset outside the mapping still moves the camera, so the
    # previous reference is stale and must be dropped.
    cam = _ptz_mock()
    cam.move_camera("ToPos", idx=1)
    cam.move_camera("ToPos", idx=99)
    assert cam.get_azimuth() is None


def test_pan_operation_invalidates_azimuth():
    cam = _ptz_mock()
    cam.move_camera("ToPos", idx=1)
    cam.move_camera("Left", speed=10)
    assert cam.get_azimuth() is None


def test_tilt_and_zoom_keep_azimuth():
    cam = _ptz_mock()
    cam.move_camera("ToPos", idx=1)
    cam.move_camera("Up", speed=10)
    cam.move_camera("ZoomIn", speed=10)
    cam.move_camera("Stop")
    assert cam.get_azimuth() == 90.0


@pytest.mark.parametrize(
    ("current", "target", "expected"),
    [
        (350.0, 10.0, 20.0),
        (10.0, 350.0, -20.0),
        (90.0, 90.0, 0.0),
        (0.0, 180.0, -180.0),
        (0.0, 90.0, 90.0),
        (180.0, 90.0, -90.0),
    ],
)
def test_shortest_delta_deg(current, target, expected):
    assert _shortest_delta_deg(current, target) == pytest.approx(expected)


def test_update_tracked_azimuth_right_wraps():
    cam = _ptz_mock()
    _update_tracked_azimuth(cam, start_azimuth=350.0, direction="Right", degrees=30.0)
    assert cam.get_azimuth() == pytest.approx(20.0)


def test_update_tracked_azimuth_left():
    cam = _ptz_mock()
    _update_tracked_azimuth(cam, start_azimuth=10.0, direction="Left", degrees=30.0)
    assert cam.get_azimuth() == pytest.approx(340.0)


def test_update_tracked_azimuth_ignores_tilt_and_unknown_start():
    cam = _ptz_mock()
    _update_tracked_azimuth(cam, start_azimuth=90.0, direction="Up", degrees=30.0)
    assert cam.get_azimuth() is None
    _update_tracked_azimuth(cam, start_azimuth=None, direction="Right", degrees=30.0)
    assert cam.get_azimuth() is None


def test_update_tracked_azimuth_ignores_hardware_source():
    cam = _ptz_mock()
    cam.azimuth_source = "hardware"
    _update_tracked_azimuth(cam, start_azimuth=90.0, direction="Right", degrees=30.0)
    assert cam.current_azimuth is None
