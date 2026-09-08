# Copyright (C) 2022-2026, Pyronear.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.


from unittest.mock import MagicMock, patch

import pytest

from pyro_camera_api.camera.adapters.mock import MockCamera
from pyro_camera_api.camera.pose_azimuths import fetch_pose_azimuths, resolve_camera_azimuths


def _response(json_body, status=200):
    resp = MagicMock()
    resp.status_code = status
    resp.json.return_value = json_body
    if status >= 400:
        resp.raise_for_status.side_effect = Exception(f"HTTP {status}")
    else:
        resp.raise_for_status.return_value = None
    return resp


API_POSES = [
    {"id": 277, "camera_id": 11, "azimuth": 10.0, "patrol_id": 0, "active": True},
    {"id": 156, "camera_id": 11, "azimuth": 75.5, "patrol_id": 1, "active": True},
    {"id": 281, "camera_id": 11, "azimuth": 140.0, "patrol_id": 2, "active": True},
    {"id": 999, "camera_id": 11, "azimuth": 200.0, "patrol_id": None, "active": True},
]


def test_fetch_pose_azimuths_maps_patrol_id():
    with patch("pyro_camera_api.camera.pose_azimuths.requests.get", return_value=_response(API_POSES)) as get:
        mapping = fetch_pose_azimuths("https://api.example.org", "tok")
    assert mapping == {0: 10.0, 1: 75.5, 2: 140.0}  # pose without patrol_id is skipped
    url = get.call_args[0][0]
    assert url == "https://api.example.org/api/v1/poses/"
    assert get.call_args.kwargs["headers"] == {"Authorization": "Bearer tok"}


def test_fetch_pose_azimuths_wraps_azimuth():
    poses = [{"id": 1, "azimuth": 360.0, "patrol_id": 0}]
    with patch("pyro_camera_api.camera.pose_azimuths.requests.get", return_value=_response(poses)):
        assert fetch_pose_azimuths("https://api.example.org", "tok") == {0: 0.0}


def test_fetch_pose_azimuths_raises_on_http_error():
    with (
        patch("pyro_camera_api.camera.pose_azimuths.requests.get", return_value=_response([], status=401)),
        pytest.raises(Exception, match="HTTP 401"),
    ):
        fetch_pose_azimuths("https://api.example.org", "bad-token")


def test_resolve_camera_azimuths_aligns_with_poses():
    cam = MockCamera(camera_id="mock", cam_type="ptz", cam_poses=[0, 2, 1])
    assert resolve_camera_azimuths(cam, {0: 10.0, 1: 75.5, 2: 140.0})
    assert cam.cam_azimuths == [10.0, 140.0, 75.5]
    # The pose sync now produces a value
    cam.move_camera("ToPos", idx=2)
    assert cam.get_azimuth() == 140.0


def test_resolve_camera_azimuths_rejects_partial_mapping():
    cam = MockCamera(camera_id="mock", cam_type="ptz", cam_poses=[0, 1, 5])
    assert not resolve_camera_azimuths(cam, {0: 10.0, 1: 75.5})
    assert cam.cam_azimuths == []
