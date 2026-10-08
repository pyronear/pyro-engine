# Copyright (C) 2022-2026, Pyronear.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.


from unittest.mock import MagicMock, patch

import requests

from pyro_camera_api.camera.adapters.reolink import ReolinkCamera

POST = "pyro_camera_api.camera.adapters.reolink._session.post"


def _reply(status=200, code=0, zoom_pos=0):
    """A GetZoomFocus response as the camera would send it."""
    resp = MagicMock()
    resp.status_code = status
    zoom = {} if zoom_pos is None else {"pos": zoom_pos}
    resp.json.return_value = [{"code": code, "value": {"ZoomFocus": {"focus": {"pos": 336}, "zoom": zoom}}}]
    return resp


def _camera(cam_type="static"):
    return ReolinkCamera(
        camera_id="cam",
        ip_address="192.168.1.10",
        username="user",
        password="pwd",  # noqa: S106
        cam_type=cam_type,
    )


def test_static_camera_with_a_varifocal_lens_can_zoom():
    """A bullet camera does not pan, which says nothing about its optics."""
    cam = _camera(cam_type="static")
    with patch(POST, return_value=_reply(zoom_pos=0)):
        assert cam.has_motorised_lens() is True


def test_fixed_lens_camera_reports_no_zoom_position():
    cam = _camera(cam_type="static")
    with patch(POST, return_value=_reply(zoom_pos=None)):
        assert cam.has_motorised_lens() is False


def test_a_rejected_probe_is_not_cached():
    """A non-zero Reolink code is not evidence about the camera's optics."""
    cam = _camera(cam_type="static")
    with patch(POST, side_effect=[_reply(code=-9), _reply(zoom_pos=0)]) as post:
        assert cam.has_motorised_lens() is False
        assert cam.has_motorised_lens() is True
        assert post.call_count == 2


def test_unreachable_camera_is_treated_as_fixed_lens():
    """Probing must not raise: a camera that cannot be asked keeps its commands
    from being sent rather than taking the whole call down."""
    cam = _camera()
    with patch(POST, side_effect=OSError("unreachable")):
        assert cam.has_motorised_lens() is False


def test_an_unanswered_probe_is_not_cached():
    """A transport failure says nothing about the optics. Caching it would
    strand a varifocal camera as fixed-lens for the rest of the process."""
    cam = _camera(cam_type="static")
    with patch(POST, side_effect=OSError("unreachable")):
        assert cam.has_motorised_lens() is False
    with patch(POST, return_value=_reply(zoom_pos=0)):
        assert cam.has_motorised_lens() is True


def test_an_http_error_is_not_cached_either():
    cam = _camera(cam_type="static")
    with patch(POST, return_value=_reply(status=500)):
        assert cam.has_motorised_lens() is False
    with patch(POST, return_value=_reply(zoom_pos=4)):
        assert cam.has_motorised_lens() is True


def test_capability_is_probed_once():
    """Zoom and focus commands are frequent; the lens cannot grow a motor."""
    cam = _camera()
    with patch(POST, return_value=_reply(zoom_pos=4)) as post:
        cam.has_motorised_lens()
        cam.has_motorised_lens()
        assert post.call_count == 1


def test_zoom_command_is_sent_to_a_static_varifocal_camera():
    """The regression this change is about: the command used to be dropped for
    every static camera, silently returning None with no request made."""
    cam = _camera(cam_type="static")
    with (
        patch(POST, return_value=_reply(zoom_pos=0)) as post,
        patch.object(ReolinkCamera, "_handle_response", return_value="ok"),
    ):
        assert cam.start_zoom_focus(32) == "ok"
        # one for the probe, one for the command
        assert post.call_count == 2
        payload = post.call_args.kwargs["json"][0]
        assert payload["param"]["ZoomFocus"]["pos"] == 32
        assert payload["param"]["ZoomFocus"]["op"] == "ZoomPos"


def test_zoom_command_is_not_sent_to_a_fixed_lens_camera():
    cam = _camera(cam_type="static")
    with patch(POST, return_value=_reply(zoom_pos=None)) as post:
        assert cam.start_zoom_focus(32) is None
        # only the probe
        assert post.call_count == 1


def test_ptz_camera_is_never_probed():
    """PTZ cameras keep sending zoom commands unconditionally, as before."""
    cam = _camera(cam_type="ptz")
    with patch(POST, side_effect=OSError("unreachable")) as post:
        assert cam.has_motorised_lens() is True
        assert post.call_count == 0


def test_a_malformed_reply_does_not_raise():
    """A 200 with a body we did not expect must stay inconclusive, not raise
    out of start_zoom_focus and turn the route's 400 into a 500."""
    not_json = MagicMock(status_code=200)
    not_json.json.side_effect = ValueError("Expecting value")
    no_zoom_focus = MagicMock(status_code=200)
    no_zoom_focus.json.return_value = [{"code": 0, "value": {}}]
    cam = _camera()
    for reply in (not_json, no_zoom_focus):
        with patch(POST, return_value=reply):
            assert cam.has_motorised_lens() is False
    with patch(POST, return_value=_reply(zoom_pos=0)):
        assert cam.has_motorised_lens() is True


def test_a_failing_probe_warns_once():
    """Failed probes are retried on every command, the log must not fill up."""
    cam = _camera()
    with (
        patch(POST, return_value=_reply(code=1)),
        patch("pyro_camera_api.camera.adapters.reolink.logger") as log,
    ):
        for _ in range(3):
            cam.has_motorised_lens()
    assert log.warning.call_count == 1
    assert log.debug.call_count == 2


def test_a_hung_camera_times_out_instead_of_blocking():
    """The probe runs under the stream lock: it must be bounded, and a
    timeout must stay inconclusive rather than raise."""
    cam = _camera()
    with patch(POST, side_effect=requests.Timeout("read timed out")) as post:
        assert cam.has_motorised_lens() is False
    assert post.call_args.kwargs["timeout"] > 0
