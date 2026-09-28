# Copyright (C) 2022-2026, Pyronear.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

# Adapter tests driven by real ISAPI answers captured on a DS-2SF8C442MXG1-ELWY/26
# (tests/fixtures/hikvision). Only the HTTP layer (_request) is faked.

import re
import xml.etree.ElementTree as ET
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from pyro_camera_api.camera.adapters.hikvision import HikvisionCamera

FIXTURES = Path(__file__).parent / "fixtures" / "hikvision"
NS = "{http://www.hikvision.com/ver20/XMLSchema}"

# specialNo list declared by the same camera in its PTZ capabilities.
SPECIAL_NO = "33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,90,92,93,94,95,96,97,98,99,100,101,102,103,104,105"
CAPABILITIES_XML = (
    '<PTZChanelCap xmlns="http://www.hikvision.com/ver20/XMLSchema">'
    "<AbsoluteZoomPositionSpace><ZRange><Min>10</Min><Max>420</Max></ZRange></AbsoluteZoomPositionSpace>"
    f'<PresetNameCap><specialNo opt="{SPECIAL_NO}"/></PresetNameCap>'
    "</PTZChanelCap>"
).encode()


def _fixture(name: str) -> str:
    return (FIXTURES / name).read_text()


def _response(status: int = 200, text: str = "") -> MagicMock:
    resp = MagicMock()
    resp.status_code = status
    resp.text = text
    resp.content = text.encode()
    return resp


class FakeISAPI:
    """Stand-in for HikvisionCamera._request: answers GETs by path, records every call."""

    def __init__(self, routes: dict[str, str]):
        self.routes = routes
        self.calls: list[tuple[str, str, dict]] = []

    def __call__(self, method: str, path: str, **kwargs):
        self.calls.append((method, path, kwargs))
        if method == "GET":
            return _response(200, self.routes[path]) if path in self.routes else _response(404)
        return _response(200)

    def puts(self) -> list[tuple[str, dict]]:
        return [(path, kwargs) for method, path, kwargs in self.calls if method == "PUT"]


def _make_cam(routes: dict[str, str], **kwargs) -> tuple[HikvisionCamera, FakeISAPI]:
    cam = HikvisionCamera(
        camera_id="203.0.113.30",
        ip_address="203.0.113.30",
        username="admin",
        password="pwd",  # noqa: S106
        disable_osd=False,
        **kwargs,
    )
    fake = FakeISAPI(routes)
    cam._request = fake  # type: ignore[method-assign]
    return cam, fake


ABSOLUTE_EX = "/ISAPI/PTZCtrl/channels/1/absoluteEx"
CAPABILITIES = "/ISAPI/PTZCtrl/channels/1/capabilities"
PRESETS = "/ISAPI/PTZCtrl/channels/1/presets"
OVERLAYS = "/ISAPI/System/Video/inputs/channels/1/overlays"


# ---------------------------------------------------------------------------
# PTZ status
# ---------------------------------------------------------------------------


def test_ptz_status_parses_decimal_degrees_and_zoom_ratio():
    cam, _ = _make_cam({ABSOLUTE_EX: _fixture("absolute_ex.xml")})

    st = cam.get_ptz_status()

    # absoluteEx carries decimal degrees and the zoom ratio directly, no tenths.
    assert st["azimuth_deg"] == pytest.approx(214.21)
    assert st["elevation_deg"] == pytest.approx(8.09)
    assert st["zoom_ratio"] == pytest.approx(1.0)
    assert st["focus_raw"] == 25110
    assert st["real_azimuth_deg"] == pytest.approx(214.21)


def test_azimuth_offset_maps_camera_frame_to_real_world():
    cam, _ = _make_cam({ABSOLUTE_EX: _fixture("absolute_ex.xml")}, azimuth_offset_deg=14.21)

    assert cam.get_azimuth() == pytest.approx(200.0)


def test_azimuth_offset_wraps_below_zero():
    cam, _ = _make_cam({ABSOLUTE_EX: _fixture("absolute_ex.xml")}, azimuth_offset_deg=300.0)

    # 214.21 - 300 = -85.79, i.e. 274.21 once wrapped.
    assert cam.get_azimuth() == pytest.approx(274.21)


# ---------------------------------------------------------------------------
# Absolute moves
# ---------------------------------------------------------------------------


def test_move_absolute_clamps_zoom_to_the_camera_range():
    cam, fake = _make_cam({CAPABILITIES: CAPABILITIES_XML.decode()})

    cam.move_absolute(azimuth_deg=370.0, elevation_deg=-40.0, zoom=100.0)

    ((path, kwargs),) = fake.puts()
    assert path == ABSOLUTE_EX
    body = ET.fromstring(kwargs["data"])
    assert float(body.find(f"{NS}azimuth").text) == pytest.approx(10.0)
    # Elevation is clamped to the safe tilt range, zoom to the 42x read from capabilities.
    assert float(body.find(f"{NS}elevation").text) == pytest.approx(-15.0)
    assert float(body.find(f"{NS}absoluteZoom").text) == pytest.approx(42.0)


# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------


def _preset_list() -> list[tuple[int, str]]:
    root = ET.fromstring(_fixture("presets.xml"))
    return [(int(p.find(f"{NS}id").text), p.find(f"{NS}presetName").text) for p in root.iter(f"{NS}PTZPreset")]


def test_reserved_ids_match_the_function_presets_the_camera_lists():
    # The camera lists its function keys (Auto-flip, Remote reboot, Call OSD
    # menu, ...) as presets too: every one of them must be blocked, and none of
    # the user positions.
    cam, _ = _make_cam({CAPABILITIES: CAPABILITIES_XML.decode()})
    presets = _preset_list()
    user_ids = {pid for pid, name in presets if name == "nord"}
    function_ids = {pid for pid, _ in presets} - user_ids

    assert function_ids == cam.reserved_preset_ids
    assert not user_ids & cam.reserved_preset_ids


def test_get_ptz_preset_returns_the_raw_list():
    cam, _ = _make_cam({PRESETS: _fixture("presets.xml")})

    assert cam.get_ptz_preset() == _fixture("presets.xml")


def test_goto_user_preset_without_azimuths_recalls_it_on_the_camera():
    cam, fake = _make_cam({CAPABILITIES: CAPABILITIES_XML.decode()})
    cam.wait_until_stationary = MagicMock(return_value={"azimuth_deg": 156.9})  # type: ignore[method-assign]

    cam.move_camera("ToPos", idx=3)

    assert [path for path, _ in fake.puts()] == ["/ISAPI/PTZCtrl/channels/1/presets/3/goto"]
    cam.wait_until_stationary.assert_called_once()


@pytest.mark.parametrize(
    ("preset_id", "name"), [(92, "Set manual limits"), (94, "Remote reboot"), (95, "Call OSD menu")]
)
def test_function_presets_are_never_sent(preset_id, name):
    cam, fake = _make_cam({CAPABILITIES: CAPABILITIES_XML.decode()})
    assert (preset_id, name) in _preset_list()

    with pytest.raises(ValueError, match="reserves it"):
        cam.move_camera("ToPos", idx=preset_id)
    with pytest.raises(ValueError, match="reserves it"):
        cam.set_ptz_preset(idx=preset_id)
    assert fake.puts() == []


# ---------------------------------------------------------------------------
# OSD
# ---------------------------------------------------------------------------


def test_disable_osd_sends_back_an_already_clean_document_unchanged():
    # This model has no PTZInfoOverlay block and ships with date and channel
    # name already off: the document must go back byte for byte.
    document = _fixture("overlays.xml")
    cam, fake = _make_cam({OVERLAYS: document})

    assert cam.disable_ptz_osd() is True
    ((path, kwargs),) = fake.puts()
    assert path == OVERLAYS
    assert kwargs["data"].decode() == document


def test_disable_osd_flips_only_the_overlay_flags():
    enabled = _fixture("overlays.xml").replace("<enabled>false</enabled>", "<enabled>true</enabled>")
    cam, fake = _make_cam({OVERLAYS: enabled})

    assert cam.disable_ptz_osd() is True
    ((_, kwargs),) = fake.puts()
    sent = kwargs["data"].decode()
    assert "<enabled>true</enabled>" not in sent
    # Everything but the flags is untouched.
    assert re.sub(r"<enabled>\w+</enabled>", "", sent) == re.sub(r"<enabled>\w+</enabled>", "", enabled)
