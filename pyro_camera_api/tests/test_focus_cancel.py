# Copyright (C) 2022-2026, Pyronear.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.


import numpy as np
import pytest
from fastapi import HTTPException
from PIL import Image

from pyro_camera_api.api import routes_focus
from pyro_camera_api.api.routes_focus import run_focus_optimization
from pyro_camera_api.camera.adapters.mock import MockCamera
from pyro_camera_api.camera.adapters.reolink import ReolinkCamera
from pyro_camera_api.camera.base import FocusAbortedError
from pyro_camera_api.camera.focus_manager import cancel_focus_and_wait, focus_abort_requested
from pyro_camera_api.camera.registry import CAMERA_REGISTRY, FOCUS_CANCEL_EVENTS, MOVE_LOCKS


def _test_image(size=(64, 64)):
    rng = np.random.default_rng(0)
    arr = rng.integers(0, 255, (size[1], size[0], 3), dtype=np.uint8)
    return Image.fromarray(arr, "RGB")


class OfflineReolink(ReolinkCamera):
    """Reolink adapter with network calls stubbed out for tests."""

    def __init__(self, **kwargs):
        super().__init__(
            camera_id=kwargs.pop("camera_id", "reolink-test"),
            ip_address="192.0.2.1",
            username="user",
            password="pwd",  # noqa: S106
            **kwargs,
        )
        self.focus_history = []

    def set_manual_focus(self, position: int):
        self.focus_position = position
        self.focus_history.append(position)

    def capture(self, patrol_id=None, timeout=2):
        _ = patrol_id, timeout
        return _test_image()

    def get_focus_level(self):
        return {"focus": 720, "zoom": 0}

    def start_zoom_focus(self, position: int):
        _ = position


@pytest.fixture(autouse=True)
def fast_sleep(monkeypatch):
    monkeypatch.setattr("pyro_camera_api.camera.adapters.reolink.time.sleep", lambda _s: None)


# ---------------------------------------------------------------------------
# cancel_focus_and_wait
# ---------------------------------------------------------------------------


def test_cancel_focus_and_wait_free_camera():
    cam_id = "cancel-free"
    assert cancel_focus_and_wait(cam_id, timeout=0.2)
    # The event stays set so no focus search can start before the stream
    # pipeline is registered; the caller clears it afterwards.
    assert FOCUS_CANCEL_EVENTS[cam_id].is_set()
    FOCUS_CANCEL_EVENTS[cam_id].clear()


def test_cancel_focus_and_wait_busy_camera():
    cam_id = "cancel-busy"
    lock = MOVE_LOCKS[cam_id]
    assert lock.acquire(blocking=False)
    try:
        assert not cancel_focus_and_wait(cam_id, timeout=0.2)
    finally:
        lock.release()
    assert not FOCUS_CANCEL_EVENTS[cam_id].is_set()


def test_focus_abort_requested_on_cancel_event():
    cam_id = "abort-event"
    assert not focus_abort_requested(cam_id)
    FOCUS_CANCEL_EVENTS[cam_id].set()
    try:
        assert focus_abort_requested(cam_id)
    finally:
        FOCUS_CANCEL_EVENTS[cam_id].clear()


# ---------------------------------------------------------------------------
# focus_finder abort behavior
# ---------------------------------------------------------------------------


def test_reolink_focus_finder_aborts_and_restores_focus(monkeypatch):
    cam = OfflineReolink(focus_position=700)
    captures: list[int] = []

    def abort_after_three() -> bool:
        return len(captures) >= 3

    real_capture = cam.capture

    def counting_capture(patrol_id=None, timeout=2):
        captures.append(1)
        return real_capture(patrol_id=patrol_id, timeout=timeout)

    monkeypatch.setattr(cam, "capture", counting_capture)

    with pytest.raises(FocusAbortedError):
        cam.focus_finder(should_abort=abort_after_three)

    # Lens back on the pre-search position, reference untouched
    assert cam.focus_history[-1] == 700
    assert cam.focus_position == 700


def test_reolink_focus_finder_abort_restores_unclamped_focus():
    # A configured focus outside the [600, 900] search interval must be
    # restored as-is, not clamped, so the lens matches the cached reference
    cam = OfflineReolink(focus_position=950)

    with pytest.raises(FocusAbortedError):
        cam.focus_finder(should_abort=lambda: True)

    assert cam.focus_history[-1] == 950
    assert cam.focus_position == 950


def test_reolink_focus_finder_abort_without_prior_reference():
    cam = OfflineReolink(focus_position=None)

    with pytest.raises(FocusAbortedError):
        cam.focus_finder(should_abort=lambda: True)

    # No reference must be stored by an aborted search
    assert cam.focus_position is None


def test_reolink_focus_finder_completes_without_abort():
    cam = OfflineReolink(focus_position=700)
    best = cam.focus_finder()
    assert cam.focus_position == best


def test_mock_focus_finder_aborts():
    cam = MockCamera(camera_id="mock-abort", cam_type="ptz")
    with pytest.raises(FocusAbortedError):
        cam.focus_finder(should_abort=lambda: True)


# ---------------------------------------------------------------------------
# /focus/focus_finder route
# ---------------------------------------------------------------------------


@pytest.fixture
def registered_mock_camera():
    cam_id = "route-focus"
    cam = MockCamera(camera_id=cam_id, cam_type="ptz", cam_poses=[0, 1], focus_position=700)
    cam._cached_image = _test_image()
    CAMERA_REGISTRY[cam_id] = cam
    yield cam_id, cam
    CAMERA_REGISTRY.pop(cam_id, None)


def test_route_refuses_when_stream_active(registered_mock_camera, monkeypatch):
    cam_id, _cam = registered_mock_camera
    monkeypatch.setattr(routes_focus, "stream_is_active", lambda _ip: True)
    with pytest.raises(HTTPException) as exc:
        run_focus_optimization(cam_id)
    assert exc.value.status_code == 409


def test_route_refuses_when_camera_busy(registered_mock_camera):
    cam_id, _cam = registered_mock_camera
    lock = MOVE_LOCKS[cam_id]
    assert lock.acquire(blocking=False)
    try:
        with pytest.raises(HTTPException) as exc:
            run_focus_optimization(cam_id)
    finally:
        lock.release()
    assert exc.value.status_code == 409


def test_route_aborts_when_cancel_event_set(registered_mock_camera):
    cam_id, _cam = registered_mock_camera
    FOCUS_CANCEL_EVENTS[cam_id].set()
    try:
        with pytest.raises(HTTPException) as exc:
            run_focus_optimization(cam_id)
    finally:
        FOCUS_CANCEL_EVENTS[cam_id].clear()
    assert exc.value.status_code == 409
    assert not MOVE_LOCKS[cam_id].locked()


def test_route_runs_and_releases_lock(registered_mock_camera):
    cam_id, _cam = registered_mock_camera
    result = run_focus_optimization(cam_id)
    assert result["best_focus_position"] == 700
    assert not MOVE_LOCKS[cam_id].locked()
