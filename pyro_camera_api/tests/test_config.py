# Copyright (C) 2022-2026, Pyronear.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

import pytest

from pyro_camera_api.core.config import build_rtsp_input_url


@pytest.mark.parametrize(
    ("cfg", "expected_path"),
    [
        ({"brand": "ctronics", "rtsp_path": "11"}, "/11"),
        ({"brand": "ctronics", "rtsp_path": "/11"}, "/11"),
        ({"brand": "linovision", "rtsp_path": "Streaming/Channels/102"}, "/Streaming/Channels/102"),
    ],
)
def test_build_rtsp_input_url_normalizes_path_and_uses_camera_credentials(monkeypatch, cfg, expected_path):
    monkeypatch.setattr("pyro_camera_api.core.config.CAM_USER", "global user")
    monkeypatch.setattr("pyro_camera_api.core.config.CAM_PWD", "global/password")
    cfg.update({"username": "camera user", "password": "camera/password"})

    assert build_rtsp_input_url("192.0.2.10", cfg) == (
        f"rtsp://camera%20user:camera%2Fpassword@192.0.2.10:554{expected_path}"
    )


def test_build_rtsp_input_url_falls_back_to_global_credentials(monkeypatch):
    monkeypatch.setattr("pyro_camera_api.core.config.CAM_USER", "global user")
    monkeypatch.setattr("pyro_camera_api.core.config.CAM_PWD", "global/password")

    assert build_rtsp_input_url("192.0.2.10", {"brand": "ctronics", "rtsp_path": "11"}) == (
        "rtsp://global%20user:global%2Fpassword@192.0.2.10:554/11"
    )