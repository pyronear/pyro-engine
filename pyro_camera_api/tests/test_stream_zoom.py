# Copyright (C) 2022-2026, Pyronear.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.


import threading
from unittest.mock import MagicMock, patch

from pyro_camera_api.services import stream

STOP = "pyro_camera_api.services.stream._stop_first_running_stream"
REGISTRY = "pyro_camera_api.services.stream.CAMERA_REGISTRY"


def _camera(zoom):
    cam = MagicMock()
    if isinstance(zoom, Exception):
        cam.get_focus_level.side_effect = zoom
    else:
        cam.get_focus_level.return_value = {"focus": 336, "zoom": zoom}
    return cam


def test_every_stop_path_restores_the_pre_stream_zoom():
    """/stop_stream, replacement in start_stream and the idle stopper all go
    through stop_any_running_stream, so a hand-set zoom survives each of them."""
    cam = _camera(7)
    with patch.dict(REGISTRY, {"cam": cam}, clear=True), patch(STOP, return_value="cam"):
        stream.save_pre_stream_zoom("cam")
        assert stream.stop_any_running_stream(app=None) == "cam"
    cam.start_zoom_focus.assert_called_once_with(position=7)
    assert "cam" not in stream._pre_stream_zoom


def test_unknown_pre_stream_zoom_falls_back_to_zero():
    cam = _camera(OSError("unreachable"))
    with patch.dict(REGISTRY, {"cam": cam}, clear=True), patch(STOP, return_value="cam"):
        stream.save_pre_stream_zoom("cam")
        stream.stop_any_running_stream(app=None)
    cam.start_zoom_focus.assert_called_once_with(position=0)


def test_nothing_stopped_means_no_zoom_command():
    cam = _camera(7)
    with patch.dict(REGISTRY, {"cam": cam}, clear=True), patch(STOP, return_value=None):
        assert stream.stop_any_running_stream(app=None) is None
    cam.start_zoom_focus.assert_not_called()


def test_stop_and_restore_hold_the_stream_lock():
    """Otherwise a concurrent start_stream could save a zoom this stop pops."""
    held = []

    def probe_lock(**_):
        t = threading.Thread(target=lambda: held.append(not stream.STREAM_LOCK.acquire(blocking=False)))
        t.start()
        t.join()

    cam = MagicMock()
    cam.start_zoom_focus.side_effect = probe_lock
    with patch.dict(REGISTRY, {"cam": cam}, clear=True), patch(STOP, return_value="cam"):
        stream.stop_any_running_stream(app=None)
    assert held == [True]
