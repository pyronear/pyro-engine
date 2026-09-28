# Copyright (C) 2022-2026, Pyronear.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

from unittest.mock import MagicMock

import pytest
import requests

from pyro_camera_api.camera.adapters.hikvision import (
    DEFAULT_ZOOM_MAX,
    FALLBACK_RESERVED_PRESET_IDS,
    HikvisionCamera,
)

# Excerpt of GET /ISAPI/PTZCtrl/channels/1/capabilities on a DS-2SF8C442MXG1-ELWY/26.
# ZRange is in tenths: 10-420 means 1x-42x.
CAPABILITIES_XML = b"""<?xml version="1.0" encoding="UTF-8"?>
<PTZChanelCap version="2.0" xmlns="http://www.hikvision.com/ver20/XMLSchema">
<AbsolutePanTiltPositionSpace>
<XRange><Min>0</Min><Max>3600</Max></XRange>
<YRange><Min>-200</Min><Max>900</Max></YRange>
</AbsolutePanTiltPositionSpace>
<AbsoluteZoomPositionSpace>
<ZRange><Min>10</Min><Max>420</Max></ZRange>
</AbsoluteZoomPositionSpace>
<ContinuousZoomSpace>
<ZRange><Min>-100</Min><Max>100</Max></ZRange>
</ContinuousZoomSpace>
<PresetNameCap version="2.0" xmlns="http://www.hikvision.com/ver20/XMLSchema">
<presetNameSupport>true</presetNameSupport>
<specialNo opt="33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,90,92,93,94,95,96,97,98,99,100,101,102,103,104,105"/>
</PresetNameCap>
</PTZChanelCap>
"""


def _make_cam(**kwargs) -> HikvisionCamera:
    return HikvisionCamera(
        camera_id="203.0.113.20",
        ip_address="203.0.113.20",
        username="admin",
        password="pwd",  # noqa: S106
        disable_osd=False,
        **kwargs,
    )


def _response(status: int, content: bytes = b"") -> MagicMock:
    resp = MagicMock()
    resp.status_code = status
    resp.content = content
    return resp


def test_zoom_max_read_from_capabilities():
    cam = _make_cam()
    cam._request = MagicMock(return_value=_response(200, CAPABILITIES_XML))

    assert cam.zoom_max == pytest.approx(42.0)
    # Cached after a successful read.
    assert cam.zoom_max == pytest.approx(42.0)
    assert cam._request.call_count == 1
    assert cam._request.call_args[0][1] == "/ISAPI/PTZCtrl/channels/1/capabilities"


def test_zoom_max_config_wins_over_capabilities():
    cam = _make_cam(zoom_max=25)
    cam._request = MagicMock(return_value=_response(200, CAPABILITIES_XML))

    assert cam.zoom_max == 25.0
    cam._request.assert_not_called()


@pytest.mark.parametrize(
    "outcome",
    [
        requests.ConnectionError("offline"),
        requests.Timeout("slow"),
        _response(503),
    ],
)
def test_zoom_max_transient_failure_falls_back_and_retries(outcome):
    cam = _make_cam()
    if isinstance(outcome, Exception):
        cam._request = MagicMock(side_effect=outcome)
    else:
        cam._request = MagicMock(return_value=outcome)

    assert cam.zoom_max == DEFAULT_ZOOM_MAX

    # Not cached: once the camera answers, the real range is used.
    cam._request = MagicMock(return_value=_response(200, CAPABILITIES_XML))
    assert cam.zoom_max == pytest.approx(42.0)


@pytest.mark.parametrize(
    "outcome",
    [
        _response(404),
        _response(401),
        _response(200, b"not xml"),
        _response(200, b"<PTZChanelCap><maxPresetNum>300</maxPresetNum></PTZChanelCap>"),
        _response(
            200,
            b"<PTZChanelCap><AbsoluteZoomPositionSpace><ZRange><Min>0</Min><Max>420</Max></ZRange>"
            b"</AbsoluteZoomPositionSpace></PTZChanelCap>",
        ),
    ],
)
def test_zoom_max_permanent_failure_is_cached(outcome):
    cam = _make_cam()
    cam._request = MagicMock(return_value=outcome)

    assert cam.zoom_max == DEFAULT_ZOOM_MAX
    assert cam.zoom_max == DEFAULT_ZOOM_MAX
    # The camera is asked once only.
    assert cam._request.call_count == 1


def test_zoom_level_64_maps_to_camera_max():
    cam = _make_cam()
    cam._request = MagicMock(return_value=_response(200, CAPABILITIES_XML))
    cam.get_ptz_status = MagicMock(return_value={"azimuth_deg": 0.0, "elevation_deg": 0.0, "zoom_ratio": 1.0})
    cam.move_absolute = MagicMock()

    assert cam.start_zoom_focus(64) == {"zoom_raw": pytest.approx(42.0), "zoom_ratio": pytest.approx(42.0)}


def test_reserved_presets_read_from_capabilities():
    cam = _make_cam()
    cam._request = MagicMock(return_value=_response(200, CAPABILITIES_XML))

    reserved = cam.reserved_preset_ids
    # Declared by the camera, and absent from the old hardcoded list.
    assert {90, 92, 93, 95} <= reserved
    assert 50 not in reserved
    assert 1 not in reserved

    with pytest.raises(ValueError, match="reserves it"):
        cam._reject_reserved_preset(95, "move to")
    cam._reject_reserved_preset(5, "move to")


def test_capabilities_fetched_once_for_zoom_and_presets():
    cam = _make_cam()
    cam._request = MagicMock(return_value=_response(200, CAPABILITIES_XML))

    assert cam.zoom_max == pytest.approx(42.0)
    assert 95 in cam.reserved_preset_ids
    assert cam._request.call_count == 1


def test_reserved_presets_fallback_retries_while_unreachable():
    cam = _make_cam()
    cam._request = MagicMock(side_effect=requests.ConnectionError("offline"))

    assert cam.reserved_preset_ids == FALLBACK_RESERVED_PRESET_IDS

    cam._request = MagicMock(return_value=_response(200, CAPABILITIES_XML))
    assert 50 not in cam.reserved_preset_ids


def test_reserved_presets_fallback_cached_when_capabilities_missing():
    cam = _make_cam()
    cam._request = MagicMock(return_value=_response(404))

    assert cam.reserved_preset_ids == FALLBACK_RESERVED_PRESET_IDS
    assert cam.reserved_preset_ids == FALLBACK_RESERVED_PRESET_IDS
    assert cam._request.call_count == 1


def test_wide_fov_defaults_when_not_configured():
    from pyro_camera_api.camera.adapters.hikvision import DEFAULT_WIDE_FOV_DEG

    assert _make_cam().wide_fov_deg == DEFAULT_WIDE_FOV_DEG
    assert _make_cam(wide_fov_deg=[59.0, 34.2]).wide_fov_deg == (59.0, 34.2)
