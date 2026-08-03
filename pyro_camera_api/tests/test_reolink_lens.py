# Copyright (C) 2022-2026, Pyronear.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.


from unittest.mock import patch

from pyro_camera_api.camera.adapters.reolink import ReolinkCamera


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
    with patch.object(ReolinkCamera, "get_focus_level", return_value={"focus": 336, "zoom": 0}):
        assert cam.has_motorised_lens() is True


def test_fixed_lens_camera_reports_no_zoom():
    cam = _camera(cam_type="static")
    with patch.object(ReolinkCamera, "get_focus_level", return_value={"focus": 336, "zoom": None}):
        assert cam.has_motorised_lens() is False


def test_unreachable_camera_is_treated_as_fixed_lens():
    """Probing must not raise: a camera that cannot be asked keeps its commands
    from being sent rather than taking the whole call down."""
    cam = _camera()
    with patch.object(ReolinkCamera, "get_focus_level", side_effect=OSError("unreachable")):
        assert cam.has_motorised_lens() is False


def test_an_inconclusive_probe_is_not_cached():
    """A failed request says nothing about the optics. Caching it would strand a
    PTZ camera as fixed-lens for the rest of the process over one bad answer."""
    cam = _camera(cam_type="ptz")
    with patch.object(ReolinkCamera, "get_focus_level", return_value=None):
        assert cam.has_motorised_lens() is False
    with patch.object(ReolinkCamera, "get_focus_level", return_value={"focus": 1, "zoom": 0}):
        assert cam.has_motorised_lens() is True


def test_capability_is_probed_once():
    """Zoom and focus commands are frequent; the lens cannot grow a motor."""
    cam = _camera()
    with patch.object(ReolinkCamera, "get_focus_level", return_value={"focus": 1, "zoom": 4}) as probe:
        cam.has_motorised_lens()
        cam.has_motorised_lens()
        assert probe.call_count == 1


def test_zoom_command_is_sent_to_a_static_varifocal_camera():
    """The regression this change is about: the command used to be dropped for
    every static camera, silently returning None with no request made."""
    cam = _camera(cam_type="static")
    with (
        patch.object(ReolinkCamera, "get_focus_level", return_value={"focus": 1, "zoom": 0}),
        patch("pyro_camera_api.camera.adapters.reolink.requests.post") as post,
        patch.object(ReolinkCamera, "_handle_response", return_value="ok"),
    ):
        assert cam.start_zoom_focus(32) == "ok"
        assert post.call_count == 1
        payload = post.call_args.kwargs["json"][0]
        assert payload["param"]["ZoomFocus"]["pos"] == 32
        assert payload["param"]["ZoomFocus"]["op"] == "ZoomPos"


def test_zoom_command_is_not_sent_to_a_fixed_lens_camera():
    cam = _camera(cam_type="static")
    with (
        patch.object(ReolinkCamera, "get_focus_level", return_value={"focus": 1, "zoom": None}),
        patch("pyro_camera_api.camera.adapters.reolink.requests.post") as post,
    ):
        assert cam.start_zoom_focus(32) is None
        assert post.call_count == 0
