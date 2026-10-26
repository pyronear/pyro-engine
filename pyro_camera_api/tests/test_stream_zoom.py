# Copyright (C) 2022-2026, Pyronear.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.


from unittest.mock import MagicMock, patch

from pyro_camera_api.api import routes_stream

STOP = "pyro_camera_api.api.routes_stream.stop_any_running_stream"
REGISTRY = "pyro_camera_api.api.routes_stream.CAMERA_REGISTRY"


def test_stop_stream_restores_the_pre_stream_zoom():
    """A static varifocal camera at a hand-set zoom must get it back, not 0."""
    cam = MagicMock()
    cam.get_focus_level.return_value = {"focus": 336, "zoom": 7}
    with patch.dict(REGISTRY, {"cam": cam}, clear=True), patch(STOP, return_value="cam"):
        routes_stream._pre_stream_zoom["cam"] = routes_stream._read_zoom("cam")
        routes_stream.stop_stream(MagicMock())
    cam.start_zoom_focus.assert_called_once_with(position=7)


def test_stop_stream_falls_back_to_zero():
    """Unknown pre-stream zoom keeps the historical reset to 0."""
    cam = MagicMock()
    cam.get_focus_level.side_effect = OSError("unreachable")
    with patch.dict(REGISTRY, {"cam": cam}, clear=True), patch(STOP, return_value="cam"):
        routes_stream._pre_stream_zoom["cam"] = routes_stream._read_zoom("cam")
        routes_stream.stop_stream(MagicMock())
    cam.start_zoom_focus.assert_called_once_with(position=0)


def test_stop_stream_holds_the_startup_lock():
    """Otherwise a concurrent start_stream could save a zoom this stop pops."""
    held = []

    def stop(_app):
        held.append(routes_stream._START_STREAM_LOCK.locked())
        return "cam"

    with patch.dict(REGISTRY, {"cam": MagicMock()}, clear=True), patch(STOP, side_effect=stop):
        routes_stream.stop_stream(MagicMock())
    assert held == [True]
    assert not routes_stream._START_STREAM_LOCK.locked()
