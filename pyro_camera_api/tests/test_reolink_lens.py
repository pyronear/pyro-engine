# Copyright (C) 2022-2026, Pyronear.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.


from unittest.mock import MagicMock, patch

from pyro_camera_api.camera.adapters.reolink import ReolinkCamera

POST = "pyro_camera_api.camera.adapters.reolink.requests.post"


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


def test_a_camera_that_rejects_the_command_is_a_settled_answer():
    """A non-zero Reolink code is the camera answering that it does not serve
    GetZoomFocus. Re-probing it would cost a request on every command forever."""
    cam = _camera(cam_type="static")
    with patch(POST, return_value=_reply(code=-9)) as post:
        assert cam.has_motorised_lens() is False
        assert cam.has_motorised_lens() is False
        assert post.call_count == 1


def test_unreachable_camera_is_treated_as_fixed_lens():
    """Probing must not raise: a camera that cannot be asked keeps its commands
    from being sent rather than taking the whole call down."""
    cam = _camera()
    with patch(POST, side_effect=OSError("unreachable")):
        assert cam.has_motorised_lens() is False


def test_an_unanswered_probe_is_not_cached():
    """A transport failure says nothing about the optics. Caching it would
    strand a PTZ camera as fixed-lens for the rest of the process."""
    cam = _camera(cam_type="ptz")
    with patch(POST, side_effect=OSError("unreachable")):
        assert cam.has_motorised_lens() is False
    with patch(POST, return_value=_reply(zoom_pos=0)):
        assert cam.has_motorised_lens() is True


def test_an_http_error_is_not_cached_either():
    cam = _camera(cam_type="ptz")
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
