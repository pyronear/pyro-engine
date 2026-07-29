# Copyright (C) 2022-2026, Pyronear.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.


from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

from pyro_camera_api.camera import focus_manager
from pyro_camera_api.camera.adapters.mock import MockCamera
from pyro_camera_api.camera.adapters.reolink import ReolinkCamera
from pyro_camera_api.camera.focus_manager import fine_adjustment, full_calibration
from pyro_camera_api.camera.registry import MOVE_LOCKS


@pytest.fixture(autouse=True)
def fast_settle(monkeypatch):
    monkeypatch.setattr(focus_manager, "FOCUS_SETTLE_TIME", 0.0)


def _flat_image(size=(64, 64)):
    return Image.new("RGB", size, (128, 128, 128))


def _sharp_image(size=(64, 64)):
    rng = np.random.default_rng(0)
    arr = rng.integers(0, 255, (size[1], size[0], 3), dtype=np.uint8)
    return Image.fromarray(arr, "RGB")


def _ptz_mock(camera_id, focus_position=None):
    cam = MockCamera(camera_id=camera_id, cam_type="ptz", cam_poses=[0, 1], focus_position=focus_position)
    cam._cached_image = _flat_image()
    return cam


class FocusDependentCamera(MockCamera):
    """Mock camera whose image is sharp only at one focus position."""

    def __init__(self, sharp_at: int, **kwargs):
        super().__init__(**kwargs)
        self.sharp_at = sharp_at

    def capture(self, **kwargs):
        _ = kwargs
        if self.focus_position == self.sharp_at:
            return _sharp_image()
        return _flat_image()


def test_full_calibration_sets_reference():
    cam = _ptz_mock("calib-ok")
    assert full_calibration(cam) == 720
    assert cam.focus_position == 720


def test_full_calibration_skips_static_camera():
    cam = MockCamera(camera_id="calib-static", cam_type="static")
    assert full_calibration(cam) is None


def test_full_calibration_skips_when_stream_active(monkeypatch):
    cam = _ptz_mock("calib-stream")
    monkeypatch.setattr(focus_manager, "stream_is_active", lambda _ip: True)
    assert full_calibration(cam) is None
    assert cam.focus_position is None


def test_full_calibration_skips_when_camera_busy():
    cam = _ptz_mock("calib-busy")
    lock = MOVE_LOCKS[cam.camera_id]
    assert lock.acquire(blocking=False)
    try:
        assert full_calibration(cam) is None
    finally:
        lock.release()
    assert full_calibration(cam) == 720


def test_fine_adjustment_requires_reference():
    cam = _ptz_mock("fine-noref")
    assert fine_adjustment(cam) is None


def test_fine_adjustment_keeps_reference_without_clear_gain():
    cam = _ptz_mock("fine-stable", focus_position=700)
    assert fine_adjustment(cam) == 700
    assert cam.focus_position == 700


def test_fine_adjustment_moves_reference_when_sharper():
    cam = FocusDependentCamera(sharp_at=702, camera_id="fine-move", cam_type="ptz", focus_position=700)
    assert fine_adjustment(cam) == 702
    assert cam.focus_position == 702


def test_fine_adjustment_aborts_when_stream_starts(monkeypatch):
    cam = _ptz_mock("fine-abort", focus_position=700)
    # First check (entry) passes, the next one (before the first offset) aborts
    monkeypatch.setattr(focus_manager, "stream_is_active", MagicMock(side_effect=[False, True]))
    assert fine_adjustment(cam) == 700
    assert cam.focus_position == 700


def test_reolink_focus_finder_aborts_before_first_capture():
    camera = ReolinkCamera("reolink-abort", "192.168.1.99", "user", "pwd", "ptz", focus_position=700)
    response = MagicMock(status_code=200)
    response.json.return_value = [{"code": 0, "value": {}}]
    with patch("pyro_camera_api.camera.adapters.reolink.requests.post", return_value=response):
        best = camera.focus_finder(should_abort=lambda: True)
    assert best == 700
    assert camera.focus_position == 700
