# Copyright (C) 2022-2026, Pyronear.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.


import pytest
from fastapi import HTTPException

from pyro_camera_api.api.routes_control import _shortest_delta_deg, _update_tracked_azimuth, move_to_azimuth
from pyro_camera_api.camera.adapters.mock import MockCamera
from pyro_camera_api.camera.registry import CAMERA_REGISTRY


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


@pytest.fixture
def registered_cam():
    cam = _ptz_mock()
    CAMERA_REGISTRY["mock-azimuth-test"] = cam
    yield cam
    del CAMERA_REGISTRY["mock-azimuth-test"]


def test_move_to_azimuth_409_without_pose_mapping(registered_cam):
    registered_cam.cam_azimuths = []
    with pytest.raises(HTTPException) as exc:
        move_to_azimuth("mock-azimuth-test", 90.0)
    assert exc.value.status_code == 409
    assert "not resolved" in exc.value.detail


@pytest.mark.usefixtures("registered_cam")
def test_move_to_azimuth_anchors_on_closest_pose():
    # Target 92°: pose 1 (90°) is the closest anchor, then a 2° residual pan.
    resp = move_to_azimuth("mock-azimuth-test", 92.0, speed=5)
    assert resp["pose_id"] == 1
    assert resp["pose_azimuth_deg"] == 90.0
    assert resp["residual_move"]["direction"] == "Right"
    assert resp["azimuth_deg"] == pytest.approx(92.0)


@pytest.mark.usefixtures("registered_cam")
def test_move_to_azimuth_anchor_only_when_target_is_a_pose():
    # Target exactly on a pose: no residual move needed, azimuth re-anchored.
    resp = move_to_azimuth("mock-azimuth-test", 180.0)
    assert resp["pose_id"] == 2
    assert "residual_move" not in resp
    assert resp["azimuth_deg"] == 180.0


def test_move_to_azimuth_works_without_prior_reference(registered_cam):
    # No preset move since boot: the pose anchor provides the reference, so
    # the old 409 "azimuth unknown" must not happen.
    assert registered_cam.get_azimuth() is None
    resp = move_to_azimuth("mock-azimuth-test", 0.0)
    assert resp["azimuth_deg"] == 0.0


@pytest.mark.usefixtures("registered_cam")
def test_move_to_azimuth_wraps_shortest_path():
    # Target 358°: pose 0 (0°) is closest through the wrap, residual goes Left.
    resp = move_to_azimuth("mock-azimuth-test", 358.0, speed=5)
    assert resp["pose_id"] == 0
    assert resp["residual_move"]["direction"] == "Left"
    assert resp["azimuth_deg"] == pytest.approx(358.0)


def test_move_to_azimuth_skips_when_already_on_target(registered_cam):
    registered_cam.move_camera("ToPos", idx=1)
    resp = move_to_azimuth("mock-azimuth-test", 90.2)
    assert resp["skipped"] is True
    assert resp["azimuth_deg"] == 90.0
