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
from pyro_camera_api.camera.base import BaseCamera, FocusAbortedError, FocusMixin
from pyro_camera_api.camera.focus_manager import (
    cancel_focus_and_wait,
    fine_adjustment,
    full_calibration,
    supports_focus_search,
)
from pyro_camera_api.camera.registry import FOCUS_CANCEL_EVENTS, MOVE_LOCKS


@pytest.fixture(autouse=True)
def fast_settle(monkeypatch):
    monkeypatch.setattr(focus_manager, "FOCUS_SETTLE_TIME", 0.0)
    monkeypatch.setattr(focus_manager, "POSE_SETTLE_TIME", 0.0)


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


class ReturnOnlyCamera(MockCamera):
    """Adapter whose focus_finder returns a value without storing it itself."""

    def __init__(self, result: int, **kwargs):
        super().__init__(**kwargs)
        self._result = result

    def focus_finder(self, save_images=False, retry_depth=0, should_abort=None):
        _ = save_images, retry_depth, should_abort
        return self._result


class NoSearchCamera(BaseCamera, FocusMixin):
    """Adapter inheriting the FocusMixin default focus_finder (no search)."""

    def capture(self, **kwargs):
        _ = kwargs
        return _flat_image()

    def set_manual_focus(self, position: int) -> None:
        self.focus_position = position

    def get_focus_level(self):
        return {"focus": self.focus_position, "zoom": 0}


def test_measure_sharpness_ignores_upper_half():
    from pyro_camera_api.utils.image_utils import measure_sharpness

    def split_image(sharp_on_top: bool, size=(64, 64)):
        flat = np.full((size[1] // 2, size[0], 3), 128, dtype=np.uint8)
        rng = np.random.default_rng(0)
        textured = rng.integers(0, 255, (size[1] // 2, size[0], 3), dtype=np.uint8)
        halves = (textured, flat) if sharp_on_top else (flat, textured)
        return Image.fromarray(np.vstack(halves), "RGB")

    # Texture only in the sky half must not register as sharpness
    assert measure_sharpness(split_image(sharp_on_top=True)) == 0.0
    assert measure_sharpness(split_image(sharp_on_top=False)) > 0.0


def test_full_calibration_sets_reference():
    cam = _ptz_mock("calib-ok")
    assert full_calibration(cam) == 720
    assert cam.focus_position == 720


def test_full_calibration_commits_reference_itself():
    cam = ReturnOnlyCamera(701, camera_id="calib-commit", cam_type="ptz")
    assert full_calibration(cam) == 701
    assert cam.focus_position == 701


def test_full_calibration_rejects_implausible_result():
    cam = ReturnOnlyCamera(-1, camera_id="calib-implausible", cam_type="ptz")
    assert full_calibration(cam) is None
    assert cam.focus_position is None


def test_supports_focus_search_excludes_default_adapter():
    cam = NoSearchCamera(camera_id="no-search", cam_type="ptz")
    assert not supports_focus_search(cam)
    assert full_calibration(cam) is None
    assert supports_focus_search(_ptz_mock("search-ok"))


def test_full_calibration_skips_on_external_abort():
    cam = _ptz_mock("calib-ext-abort")
    assert full_calibration(cam, should_abort=lambda: True) is None
    assert cam.focus_position is None


def test_cancel_focus_and_wait():
    cam_id = "cancel-wait"
    assert cancel_focus_and_wait(cam_id, timeout=0.2)
    assert not FOCUS_CANCEL_EVENTS[cam_id].is_set()

    lock = MOVE_LOCKS[cam_id]
    assert lock.acquire(blocking=False)
    try:
        assert not cancel_focus_and_wait(cam_id, timeout=0.2)
    finally:
        lock.release()
    assert not FOCUS_CANCEL_EVENTS[cam_id].is_set()


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


class ProbeFailureCamera(MockCamera):
    """Mock camera whose capture starts failing after the first probe."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._cached_image = _flat_image()
        self.captures = 0

    def capture(self, **kwargs):
        _ = kwargs
        self.captures += 1
        if self.captures > 1:
            raise RuntimeError("camera unreachable")
        return _flat_image()


def test_fine_adjustment_restores_reference_when_probe_raises():
    cam = ProbeFailureCamera(camera_id="fine-error", cam_type="ptz", focus_position=700)
    with pytest.raises(RuntimeError):
        fine_adjustment(cam)
    assert cam.focus_position == 700


def test_fine_adjustment_aborts_when_stream_starts(monkeypatch):
    cam = _ptz_mock("fine-abort", focus_position=700)
    # First check (entry) passes, the next one (before the first offset) aborts
    monkeypatch.setattr(focus_manager, "stream_is_active", MagicMock(side_effect=[False, True]))
    assert fine_adjustment(cam) == 700
    assert cam.focus_position == 700


def test_route_refuses_focus_finder_while_patrol_running():
    from fastapi import HTTPException

    from pyro_camera_api.api.routes_focus import run_focus_optimization
    from pyro_camera_api.camera.registry import CAMERA_REGISTRY, PATROL_FLAGS, PATROL_THREADS

    cam_ip = "route-patrol"
    CAMERA_REGISTRY[cam_ip] = _ptz_mock(cam_ip)
    PATROL_THREADS[cam_ip] = MagicMock(is_alive=lambda: True)
    PATROL_FLAGS[cam_ip] = MagicMock(is_set=lambda: False)
    try:
        with pytest.raises(HTTPException) as exc:
            run_focus_optimization(cam_ip)
        assert exc.value.status_code == 409
        assert "stop the patrol" in exc.value.detail

        PATROL_FLAGS[cam_ip] = MagicMock(is_set=lambda: True)
        with pytest.raises(HTTPException) as exc:
            run_focus_optimization(cam_ip)
        assert exc.value.status_code == 409
        assert "stopping" in exc.value.detail
    finally:
        del CAMERA_REGISTRY[cam_ip]
        del PATROL_THREADS[cam_ip]
        del PATROL_FLAGS[cam_ip]


def test_reolink_focus_finder_aborts_before_first_capture():
    camera = ReolinkCamera("reolink-abort", "192.168.1.99", "user", "pwd", "ptz", focus_position=700)
    response = MagicMock(status_code=200)
    response.json.return_value = [{"code": 0, "value": {}}]
    with (
        patch("pyro_camera_api.camera.adapters.reolink.requests.post", return_value=response),
        pytest.raises(FocusAbortedError),
    ):
        camera.focus_finder(should_abort=lambda: True)
    assert camera.focus_position == 700


def test_reolink_focus_finder_abort_mid_sweep_restores_reference(monkeypatch):
    camera = ReolinkCamera("reolink-midabort", "192.168.1.99", "user", "pwd", "ptz", focus_position=700)
    monkeypatch.setattr("pyro_camera_api.camera.adapters.reolink.time.sleep", lambda *_: None)
    response = MagicMock(status_code=200)
    response.json.return_value = [{"code": 0, "value": {}}]
    checks = {"count": 0}

    def abort_after_three_captures():
        checks["count"] += 1
        return checks["count"] > 3

    with (
        patch("pyro_camera_api.camera.adapters.reolink.requests.post", return_value=response),
        patch.object(camera, "capture", return_value=_sharp_image()),
        pytest.raises(FocusAbortedError),
    ):
        camera.focus_finder(should_abort=abort_after_three_captures)
    # The partial sweep (600..620) must not replace the reference
    assert camera.focus_position == 700


def test_reolink_failed_search_without_prior_reference_stores_none(monkeypatch):
    camera = ReolinkCamera("reolink-noref", "192.168.1.99", "user", "pwd", "ptz")
    monkeypatch.setattr("pyro_camera_api.camera.adapters.reolink.time.sleep", lambda *_: None)
    response = MagicMock(status_code=200)
    response.json.return_value = [{"code": 0, "value": {}}]
    with (
        patch("pyro_camera_api.camera.adapters.reolink.requests.post", return_value=response),
        patch.object(camera, "get_focus_level", return_value={"focus": 650, "zoom": 0}),
        patch.object(camera, "capture", return_value=None),
        pytest.raises(RuntimeError),
    ):
        camera.focus_finder()
    # No reference existed before the failed search, none may exist after,
    # so the patrol retries a full calibration instead of fine-tuning
    assert camera.focus_position is None


def test_full_calibration_aborted_mid_search_stores_no_reference():
    cam = _ptz_mock("calib-abort-mid")
    flags = iter([False, True])
    assert full_calibration(cam, should_abort=lambda: next(flags, True)) is None
    assert cam.focus_position is None


def test_reolink_focus_finder_fails_without_valid_capture(monkeypatch):
    camera = ReolinkCamera("reolink-nocapture", "192.168.1.99", "user", "pwd", "ptz", focus_position=700)
    response = MagicMock(status_code=200)
    response.json.return_value = [{"code": 0, "value": {}}]
    monkeypatch.setattr("pyro_camera_api.camera.adapters.reolink.time.sleep", lambda *_: None)
    with (
        patch("pyro_camera_api.camera.adapters.reolink.requests.post", return_value=response),
        patch.object(camera, "capture", return_value=None),
        pytest.raises(RuntimeError),
    ):
        camera.focus_finder()
    # The blurry sweep start (600) must not become the reference
    assert camera.focus_position == 700
