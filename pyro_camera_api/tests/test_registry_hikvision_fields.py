# Copyright (C) 2022-2026, Pyronear.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

from unittest.mock import patch

import pytest

from pyro_camera_api.camera.adapters.linovision import LinovisionCamera
from pyro_camera_api.camera.registry import build_camera_object


@pytest.mark.parametrize("adapter", ["hikvision", "linovision"])
@pytest.mark.parametrize(
    ("fields", "expected_elevation", "expected_timeout"),
    [
        ({}, 0, 3.0),
        ({"timeout": 7.5}, 0, 7.5),
        ({"default_elevation_deg": -4.5, "timeout": 9.0}, -4.5, 9.0),
        ({"default_elevation_deg": 0, "timeout": 8.0}, 0, 8.0),
        ({"default_elevation_deg": None, "timeout": 8.0}, None, 8.0),
    ],
)
def test_elevation_and_timeout_are_independent(adapter, fields, expected_elevation, expected_timeout):
    # Exercise the real constructor without its startup ISAPI request.
    with patch.object(LinovisionCamera, "disable_ptz_osd"):
        camera = build_camera_object("test-camera", {"adapter": adapter, **fields})
    assert isinstance(camera, LinovisionCamera)
    assert camera.default_elevation_deg == expected_elevation
    assert camera.timeout == expected_timeout


def test_preset_move_uses_configured_elevation():
    with patch.object(LinovisionCamera, "disable_ptz_osd"):
        camera = build_camera_object(
            "test-camera",
            {
                "adapter": "hikvision",
                "type": "ptz",
                "poses": [3],
                "azimuths": [120],
                "azimuth_offset_deg": 9,
                "default_elevation_deg": -4.5,
                "timeout": 9.0,
            },
        )
    assert isinstance(camera, LinovisionCamera)
    with patch.object(camera, "move_absolute_perfect") as move:
        camera.move_camera("ToPos", idx=3)
    move.assert_called_once_with(
        azimuth_deg=129.0,
        elevation_deg=-4.5,
        timeout_s=15.0,
        poll_s=0.15,
        prefer_current_elevation=False,
    )
