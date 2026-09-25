# Copyright (C) 2022-2026, Pyronear.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

from unittest.mock import MagicMock

import pytest
import requests

from pyro_camera_api.camera.adapters.hikvision import DEFAULT_ZOOM_MAX, HikvisionCamera

# Excerpt of GET /ISAPI/PTZCtrl/channels/1/capabilities on a Hikvision dome.
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
