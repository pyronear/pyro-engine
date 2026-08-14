# Copyright (C) 2022-2026, Pyronear.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.


from io import BytesIO
from unittest.mock import MagicMock, patch

from PIL import Image

from pyro_camera_api.camera.adapters.ctronics import CTronicsCamera
from pyro_camera_api.camera.base import FocusMixin, PTZMixin


def test_ctronics_exposes_ptz_and_focus_capabilities():
    camera = CTronicsCamera("cam", "192.0.2.10", "user", "secret", cam_type="ptz", onvif_port=8080)

    assert isinstance(camera, PTZMixin)
    assert isinstance(camera, FocusMixin)
    assert camera.onvif_port == 8080


def test_capture_builds_tmpfs_snapshot_url_and_returns_rgb_image():
    payload = BytesIO()
    Image.new("RGB", (8, 8), (10, 20, 30)).save(payload, format="JPEG")
    response = MagicMock(content=payload.getvalue())
    response.raise_for_status.return_value = None
    camera = CTronicsCamera("cam", "192.0.2.10", "user", "secret")

    with patch("pyro_camera_api.camera.adapters.ctronics.requests.get", return_value=response) as get:
        image = camera.capture()

    assert image is not None
    assert image.mode == "RGB"
    assert get.call_args.kwargs["timeout"] == 5.0
    assert get.call_args.args[0] == (
        "http://192.0.2.10:80/tmpfs/snap.jpg?usr=user&pwd=secret"
    )


def test_snapshot_path_and_command_are_configurable_per_model():
    camera = CTronicsCamera(
        "cam",
        "192.0.2.10",
        "user",
        "secret",
        port=8080,
        snapshot_path="/api/snapshot",
        snapshot_command="image",
        model="future-ctronics-model",
    )

    assert camera.snapshot_url == "http://192.0.2.10:8080/api/snapshot?cmd=image&usr=user&pwd=secret"