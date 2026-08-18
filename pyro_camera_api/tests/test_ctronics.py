# Copyright (C) 2022-2026, Pyronear.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

"""CTronics adapter tests.

Real-camera test procedure
===========================

The default tests use a fake ONVIF client and do not contact a camera. They are
safe to run in CI or on Vercel. To run the integration tests against a real
camera, follow these steps from the ``pyro_camera_api`` directory:

1. Install the project and test dependencies::

    uv sync

2. Set the camera connection variables. The ONVIF port defaults to 8080::

    export CTRONICS_IP="192.168.1.XX"
    export CTRONICS_USER="admin"
    export CTRONICS_PASSWORD="your-password"
    export CTRONICS_ONVIF_PORT="8080"

   Optional variables are ``CTRONICS_HTTP_PORT`` (default ``80``),
   ``CTRONICS_ONVIF_PROTOCOL`` (default ``http``),
   ``CTRONICS_ONVIF_PROFILE``, and ``CTRONICS_SNAPSHOT_PATH`` (default
   ``/tmpfs/snap.jpg``).

3. Run the safe real-camera smoke test. It checks snapshot capture, ONVIF
   discovery, and preset listing without moving the camera::

    PYTHONPATH=pyro_camera_api CTRONICS_TEST_REAL=1 uv run pytest pyro_camera_api/tests/test_ctronics.py -v

   add "-s --log-cli-level=INFO" for more log infos during test

4. Test a short PTZ movement followed immediately by Stop. This physically
   moves the camera::

    PYTHONPATH=pyro_camera_api CTRONICS_TEST_PTZ=1 CTRONICS_TEST_DIRECTION=Right uv run pytest pyro_camera_api/tests/test_ctronics.py -v

   Valid directions are ``Left``, ``Right``, ``Up``, ``Down``, ``UpLeft``,
   ``UpRight``, ``DownLeft``, ``DownRight``, ``ZoomIn``, and ``ZoomOut``.

5. Test a preset move. Replace ``1`` with an existing ONVIF preset token or
   index configured on the camera::

    PYTHONPATH=pyro_camera_api CTRONICS_TEST_PRESET=1 CTRONICS_TEST_PRESET_ID=1 uv run pytest pyro_camera_api/tests/test_ctronics.py -v

6. Test one relative focus step. Use ``focusin`` for ``+`` and ``focusout`` for ``-``::

    PYTHONPATH=pyro_camera_api CTRONICS_TEST_FOCUS=1 CTRONICS_TEST_FOCUS_ACTION=focusout CTRONICS_TEST_FOCUS_SPEED=45 uv run pytest pyro_camera_api/tests/test_ctronics.py -v

     * ``CTRONICS_TEST_FOCUS_ACTION``: ``focusin`` moves focus toward near and
         ``focusout`` moves it toward far. The test sends ``focusstop`` immediately
         afterward to stop the movement.
     * ``CTRONICS_TEST_FOCUS_SPEED``: integer speed sent as ``-speed``.

     The HTTP endpoint is configurable with ``focus_path`` and defaults to
     ``/web/cgi-bin/hi3510/ptzctrl.cgi``. Authentication defaults to HTTP
     Digest and can be changed with ``CTRONICS_FOCUS_AUTH`` to ``basic`` or
     ``none``. The focus action is relative, so there is no absolute focus
     minimum or maximum value in this API; repeat the command to move farther.

7. Run the focus finder only when a focus sweep is acceptable. It moves the
   focus through several positions and may take a while::

    PYTHONPATH=pyro_camera_api CTRONICS_TEST_FOCUS_FINDER=1 uv run pytest pyro_camera_api/tests/test_ctronics.py -v

8. Test reboot separately. The camera will restart and temporarily disconnect::

    PYTHONPATH=pyro_camera_api CTRONICS_TEST_REBOOT=1 uv run pytest pyro_camera_api/tests/test_ctronics.py -v

Run the complete local test file without hardware with::

    PYTHONPATH=pyro_camera_api uv run pytest pyro_camera_api/tests/test_ctronics.py -v
"""

import logging
import os
import sys
import time
from io import BytesIO
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image
from requests.auth import HTTPDigestAuth

from pyro_camera_api.camera.adapters.ctronics import CTronicsCamera
from pyro_camera_api.camera.base import FocusAbortedError, FocusMixin, PTZMixin


class DictToAttr:
    """Convertit récursivement un dictionnaire en objet avec accès par attribut."""

    def __init__(self, data):
        for key, value in data.items():
            if isinstance(value, dict):
                setattr(self, key, DictToAttr(value))
            else:
                setattr(self, key, value)


class FakeOnvifService:
    def __init__(self):
        self.calls = []
        self.presets = [SimpleNamespace(token="0", Name="home"), SimpleNamespace(token="1", Name="west")]
        self.focus_position = 0.5

    def create_type(self, name):
        return SimpleNamespace(_type=name)

    def GetProfiles(self):
        return [SimpleNamespace(token="profile-1", VideoSourceConfiguration=SimpleNamespace(SourceToken="video-1"))]

    def ContinuousMove(self, request):
        if isinstance(request.Velocity, dict):
            request.Velocity = DictToAttr(request.Velocity)
        self.calls.append(("ContinuousMove", request))

    def Stop(self, request):
        self.calls.append(("Stop", request))

    def GetPresets(self, request):
        self.calls.append(("GetPresets", request))
        return self.presets

    def GotoPreset(self, request):
        self.calls.append(("GotoPreset", request))

    def SetPreset(self, request):
        self.calls.append(("SetPreset", request))
        return SimpleNamespace(token=request.PresetToken or "new")

    def Move(self, request):
        self.calls.append(("Move", request))
        focus = request.Focus
        if isinstance(focus, dict):
            self.focus_position = focus["Absolute"]["Position"]
        else:
            self.focus_position = focus.Absolute.Position

    def GetStatus(self, request):
        self.calls.append(("GetStatus", request))
        return SimpleNamespace(FocusStatus20=SimpleNamespace(Position=self.focus_position))

    def GetImagingSettings(self, request):
        self.calls.append(("GetImagingSettings", request))
        return SimpleNamespace(Focus=SimpleNamespace(AutoFocusMode="AUTO"))

    def SetImagingSettings(self, request):
        self.calls.append(("SetImagingSettings", request))


class FakeOnvifCamera:
    def __init__(self, host, port, username, password, wsdl_dir=None):
        self.args = (host, port, username, password, wsdl_dir)
        self.media = FakeOnvifService()
        self.ptz = FakeOnvifService()
        self.imaging = FakeOnvifService()
        self.devicemgmt = MagicMock()

    def create_media_service(self):
        return self.media

    def create_ptz_service(self):
        return self.ptz

    def create_imaging_service(self):
        return self.imaging


@pytest.fixture
def fake_onvif():
    module = ModuleType("onvif")
    module.ONVIFCamera = FakeOnvifCamera # type: ignore[attr-defined]
    with patch.dict(sys.modules, {"onvif": module}):
        yield


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
    assert get.call_args.args[0] == ("http://192.0.2.10:80/tmpfs/snap.jpg?usr=user&pwd=secret")


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


def test_onvif_connection_uses_configured_port_and_profile(fake_onvif):
    camera = CTronicsCamera("cam", "192.0.2.10", "user", "secret", cam_type="ptz", onvif_port=8080)

    camera._ensure_onvif()

    assert camera._onvif_camera.args[:4] == ("192.0.2.10", 8080, "user", "secret")
    assert camera.onvif_profile_token == "profile-1"


@pytest.mark.parametrize(
    ("operation", "axis"),
    [
        ("Left", "x"),
        ("Right", "x"),
        ("Up", "y"),
        ("Down", "y"),
        ("UpLeft", "xy"),
        ("UpRight", "xy"),
        ("DownLeft", "xy"),
        ("DownRight", "xy"),
        ("ZoomIn", "zoom"),
        ("ZoomOut", "zoom"),
    ],
)
def test_move_camera_maps_operations_to_onvif(fake_onvif, operation, axis):
    camera = CTronicsCamera("cam", "192.0.2.10", "user", "secret", cam_type="ptz")

    camera.move_camera(operation, speed=32)

    call_name, request = camera._ptz_service.calls[-1]
    assert call_name == "ContinuousMove"
    assert request.ProfileToken == "profile-1"
    assert request.Velocity.PanTilt.x == (
        -0.5 if operation in {"Left", "UpLeft", "DownLeft"} else 0.5 if "Right" in operation else 0
    )
    assert request.Velocity.PanTilt.y == (
        0.5
        if operation in {"Up", "UpLeft", "UpRight"}
        else -0.5
        if operation in {"Down", "DownLeft", "DownRight"}
        else 0
    )
    assert request.Velocity.Zoom.x == (-0.5 if operation == "ZoomOut" else 0.5 if operation == "ZoomIn" else 0)


def test_stop_preset_and_azimuth_tracking(fake_onvif):
    camera = CTronicsCamera(
        "cam", "192.0.2.10", "user", "secret", cam_type="ptz", cam_poses=[0, 1], cam_azimuths=[0, 90]
    )

    camera.move_camera("ToPos", idx=1)
    camera.move_camera("Stop")

    assert camera.get_azimuth() == 90.0
    assert [call[0] for call in camera._ptz_service.calls if call[0] in {"GetPresets", "GotoPreset", "Stop"}] == [
        "GetPresets",
        "GotoPreset",
        "Stop",
    ]


def test_preset_focus_autofocus_and_reboot_use_onvif(fake_onvif):
    camera = CTronicsCamera("cam", "192.0.2.10", "user", "secret", cam_type="ptz")

    presets = camera.get_ptz_preset()
    camera.set_ptz_preset(idx=2, name="tower")
    camera.set_manual_focus(250)
    focus = camera.get_focus_level()
    autofocus = camera.get_auto_focus()
    camera.set_auto_focus(disable=True)
    camera.start_zoom_focus(300)
    assert camera.reboot_camera() is True

    assert len(presets) == 2
    assert focus["focus"] == 250
    assert autofocus["mode"] == "AUTO"
    camera._onvif_camera.devicemgmt.SystemReboot.assert_called_once_with()


@patch("pyro_camera_api.camera.adapters.ctronics.requests.get")
def test_ctronics_focus_plus_and_minus_use_hi3510_cgi(mock_get):
    response = MagicMock(status_code=200, text="OK")
    response.raise_for_status.return_value = None
    mock_get.return_value = response
    camera = CTronicsCamera("cam", "192.168.1.2", "user", "secret", focus_speed=45, focus_auth="none")

    assert camera.focus_plus() is True
    assert camera.focus_minus(speed=30) is True
    assert camera.stop_focus() is True

    assert mock_get.call_args_list[0].args[0] == (
        "http://192.168.1.2:80/web/cgi-bin/hi3510/ptzctrl.cgi?-step=0&-act=focusin&-speed=45"
    )
    assert mock_get.call_args_list[1].args[0] == (
        "http://192.168.1.2:80/web/cgi-bin/hi3510/ptzctrl.cgi?-step=0&-act=focusout&-speed=30"
    )
    assert mock_get.call_args_list[2].args[0] == (
        "http://192.168.1.2:80/web/cgi-bin/hi3510/ptzctrl.cgi?-step=0&-act=stop&-speed=45"
    )


@patch("pyro_camera_api.camera.adapters.ctronics.requests.get")
def test_ctronics_focus_uses_digest_auth_by_default(mock_get):
    response = MagicMock(status_code=200, text="OK")
    response.raise_for_status.return_value = None
    mock_get.return_value = response
    camera = CTronicsCamera("cam", "192.168.1.2", "user", "secret")

    camera.focus_minus()

    auth = mock_get.call_args.kwargs["auth"]
    assert isinstance(auth, HTTPDigestAuth)


def test_focus_finder_honors_abort_without_hardware(fake_onvif):
    camera = CTronicsCamera("cam", "192.0.2.10", "user", "secret", cam_type="ptz")
    camera.focus_position = 500

    with pytest.raises(FocusAbortedError):
        camera.focus_finder(should_abort=lambda: True)


def _real_camera() -> CTronicsCamera:
    camera = CTronicsCamera(
        "ctronics-real",
        os.environ["CTRONICS_IP"],
        os.environ["CTRONICS_USER"],
        os.environ["CTRONICS_PASSWORD"],
        port=int(os.getenv("CTRONICS_HTTP_PORT", "80")),
        cam_type="ptz",
        onvif_port=int(os.getenv("CTRONICS_ONVIF_PORT", "8080")),
        onvif_protocol=os.getenv("CTRONICS_ONVIF_PROTOCOL", "http"),
        onvif_profile_token=os.getenv("CTRONICS_ONVIF_PROFILE"),
        focus_auth=os.getenv("CTRONICS_FOCUS_AUTH", "digest"),
        snapshot_path=os.getenv("CTRONICS_SNAPSHOT_PATH", "/tmpfs/snap.jpg"),
    )
    return camera


@pytest.mark.skipif(
    os.getenv("CTRONICS_TEST_REAL") != "1",
    reason="Set CTRONICS_TEST_REAL=1 to run against a physical camera",
)
def test_real_ctronics_capture_and_onvif_discovery():
    camera = _real_camera()

    image = camera.capture()
    assert image is not None
    camera._ensure_onvif()
    assert camera.get_ptz_preset() is not None


@pytest.mark.skipif(
    os.getenv("CTRONICS_TEST_PTZ") != "1",
    reason="Set CTRONICS_TEST_PTZ=1 to move a physical camera",
)
def test_real_ctronics_ptz_move_and_stop():
    camera = _real_camera()
    camera.move_camera(os.getenv("CTRONICS_TEST_DIRECTION", "Right"), speed=1)
    time.sleep(1)
    camera.move_camera("Stop")


@pytest.mark.skipif(
    os.getenv("CTRONICS_TEST_PRESET") != "1",
    reason="Set CTRONICS_TEST_PRESET=1 to move to a physical-camera preset",
)
def test_real_ctronics_preset_and_azimuth():
    camera = _real_camera()
    preset_id = int(os.environ["CTRONICS_TEST_PRESET_ID"])
    logging.basicConfig(level=logging.INFO)
    print(f"[CTronics test] Requesting preset id/index={preset_id}", flush=True)
    presets = camera.get_ptz_preset() or []
    # print(
    #     "[CTronics test] Available presets: "
    #     + repr(
    #         [
    #             {
    #                 "token": getattr(preset, "token", None),
    #                 "name": getattr(preset, "Name", getattr(preset, "name", None)),
    #             }
    #             for preset in presets
    #         ]
    #     ),
    #     flush=True,
    # )
    camera.move_camera("ToPos", idx=preset_id)
    print(f"[CTronics test] GotoPreset completed for id/index={preset_id}", flush=True)
    assert camera.get_azimuth() is None or 0 <= camera.get_azimuth() < 360


@pytest.mark.skipif(
    os.getenv("CTRONICS_TEST_FOCUS") != "1",
    reason="Set CTRONICS_TEST_FOCUS=1 to change focus on a physical camera",
)
def test_real_ctronics_focus():
    camera = _real_camera()
    action = os.getenv("CTRONICS_TEST_FOCUS_ACTION", "focusout")
    speed = int(os.getenv("CTRONICS_TEST_FOCUS_SPEED", "45"))
    print(f"[CTronics test] Sending focus action={action}, speed={speed}", flush=True)
    try:
        assert camera.move_focus(action, speed=speed) is True
        time.sleep(1)
    finally:
        assert camera.stop_focus(speed=speed) is True


@pytest.mark.skipif(
    os.getenv("CTRONICS_TEST_FOCUS_FINDER") != "1",
    reason="Set CTRONICS_TEST_FOCUS_FINDER=1 to run the focus sweep on a physical camera",
)
def test_real_ctronics_focus_finder():
    result = _real_camera().focus_finder(save_images=False)
    assert isinstance(result, int)


@pytest.mark.skipif(
    os.getenv("CTRONICS_TEST_REBOOT") != "1",
    reason="Set CTRONICS_TEST_REBOOT=1 to reboot a physical camera",
)
def test_real_ctronics_reboot():
    assert _real_camera().reboot_camera() is True
