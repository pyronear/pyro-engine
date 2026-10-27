# Copyright (C) 2022-2026, Pyronear.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

from unittest.mock import patch

import pytest

from pyro_camera_api.camera.adapters.hikvision import HikvisionCamera
from pyro_camera_api.camera.adapters.linovision import LinovisionCamera
from pyro_camera_api.camera.registry import build_camera_object


@pytest.mark.parametrize(("adapter", "cls"), [("hikvision", HikvisionCamera), ("linovision", LinovisionCamera)])
@pytest.mark.parametrize(
    ("fields", "expected_elevation", "expected_timeout"),
    [
        ({"timeout": 7.5}, 0, 7.5),
        ({"default_elevation_deg": -4.5}, -4.5, 3.0),
    ],
)
def test_elevation_and_timeout_are_independent(adapter, cls, fields, expected_elevation, expected_timeout):
    # Exercise the real constructor without its startup ISAPI request.
    with patch.object(cls, "disable_ptz_osd"):
        camera = build_camera_object("test-camera", {"adapter": adapter, **fields})
    assert isinstance(camera, cls)
    assert camera.default_elevation_deg == expected_elevation
    assert camera.timeout == expected_timeout


def test_preset_move_uses_configured_elevation():
    with patch.object(HikvisionCamera, "disable_ptz_osd"):
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
    assert isinstance(camera, HikvisionCamera)
    assert camera.timeout == 9.0
    with patch.object(camera, "move_absolute_blocking") as move:
        camera.move_camera("ToPos", idx=3)
    move.assert_called_once_with(azimuth_deg=129.0, elevation_deg=-4.5)
