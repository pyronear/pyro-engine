# Copyright (C) 2022-2026, Pyronear.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.


import base64
from io import BytesIO
from unittest.mock import MagicMock, patch

from PIL import Image

from pyro_camera_api.camera.adapters.rest import RestSnapshotCamera


def _jpeg_bytes(color=(10, 20, 30)):
    buf = BytesIO()
    Image.new("RGB", (8, 8), color).save(buf, format="JPEG")
    return buf.getvalue()


def _response(status=200, content=b"", json_body=None):
    resp = MagicMock()
    resp.status_code = status
    resp.content = content
    if json_body is not None:
        resp.json.return_value = json_body
    if status >= 400:
        resp.raise_for_status.side_effect = Exception(f"HTTP {status}")
    else:
        resp.raise_for_status.return_value = None
    return resp


def test_capture_raw_image():
    cam = RestSnapshotCamera("cam", "https://host/snap", response="image")
    with patch.object(cam._session, "get", return_value=_response(content=_jpeg_bytes())) as get:
        img = cam.capture()
    assert isinstance(img, Image.Image)
    assert img.size == (8, 8)
    # Custom headers are forwarded on the request.
    get.assert_called_once()


def test_capture_json_base64_vigilant_shape():
    payload = base64.b64encode(_jpeg_bytes()).decode()
    body = {"success": True, "data": payload}
    cam = RestSnapshotCamera(
        "cam",
        "https://vigilant.cat/api/v1/cameras/uuid/snap",
        headers={"Authorization": "Bearer secret"},
        response="json",
        json_path="data",
        encoding="base64",
    )
    with patch.object(cam._session, "get", return_value=_response(json_body=body)):
        img = cam.capture()
    assert isinstance(img, Image.Image)
    assert img.size == (8, 8)


def test_capture_json_base64_strips_data_uri_prefix():
    payload = "data:image/jpeg;base64," + base64.b64encode(_jpeg_bytes()).decode()
    cam = RestSnapshotCamera("cam", "https://host/snap", response="json", json_path="result.image")
    with patch.object(cam._session, "get", return_value=_response(json_body={"result": {"image": payload}})):
        img = cam.capture()
    assert isinstance(img, Image.Image)


def test_capture_json_url_refetch():
    cam = RestSnapshotCamera("cam", "https://host/meta", response="json", json_path="url", encoding="url")
    meta = _response(json_body={"url": "https://cdn/host/frame.jpg"})
    image = _response(content=_jpeg_bytes())
    with patch.object(cam._session, "get", side_effect=[meta, image]) as get:
        img = cam.capture()
    assert isinstance(img, Image.Image)
    assert get.call_count == 2


def test_url_refetch_same_origin_forwards_auth_header():
    cam = RestSnapshotCamera(
        "cam",
        "https://host/meta",
        headers={"Authorization": "Bearer secret"},
        response="json",
        json_path="url",
        encoding="url",
    )
    meta = _response(json_body={"url": "https://host/frames/1.jpg"})  # same origin
    image = _response(content=_jpeg_bytes())
    with patch.object(cam._session, "get", side_effect=[meta, image]) as get:
        cam.capture()
    _, kwargs = get.call_args  # second (nested) call
    assert kwargs["headers"].get("Authorization") == "Bearer secret"


def test_url_refetch_cross_origin_strips_auth_header():
    cam = RestSnapshotCamera(
        "cam",
        "https://host/meta",
        headers={"Authorization": "Bearer secret", "User-Agent": "pyro"},
        response="json",
        json_path="url",
        encoding="url",
    )
    meta = _response(json_body={"url": "https://cdn.other.com/1.jpg"})  # different origin
    image = _response(content=_jpeg_bytes())
    with patch.object(cam._session, "get", side_effect=[meta, image]) as get:
        cam.capture()
    _, kwargs = get.call_args
    assert "Authorization" not in kwargs["headers"]
    assert kwargs["headers"].get("User-Agent") == "pyro"  # non-sensitive header kept


def test_capture_returns_none_on_http_error():
    cam = RestSnapshotCamera("cam", "https://host/snap", retries=0)
    with patch.object(cam._session, "get", return_value=_response(status=500, content=b"<html>error</html>")):
        assert cam.capture() is None


def test_capture_retries_then_succeeds():
    cam = RestSnapshotCamera("cam", "https://host/snap", retries=2)
    ok = _response(content=_jpeg_bytes())
    with patch.object(cam._session, "get", side_effect=[Exception("boom"), ok]) as get:
        img = cam.capture()
    assert isinstance(img, Image.Image)
    assert get.call_count == 2


def test_capture_gives_up_after_all_attempts():
    cam = RestSnapshotCamera("cam", "https://host/snap", retries=2)
    with patch.object(cam._session, "get", side_effect=Exception("boom")) as get:
        assert cam.capture() is None
    assert get.call_count == 3  # retries + 1


def test_redaction_masks_secrets():
    cam = RestSnapshotCamera("cam", "https://host/snap?token=abc123", headers={"Authorization": "Bearer secret"})
    assert cam._redact_headers(cam.headers)["Authorization"] == "***"
    assert "abc123" not in cam._redact_url(cam.url)
