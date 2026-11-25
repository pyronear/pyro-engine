# Copyright (C) 2022-2026, Pyronear.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

from io import BytesIO

import numpy as np
import pytest
from fastapi import HTTPException
from PIL import Image

from pyro_camera_api.api.routes_cameras import get_frame, get_latest_image, list_frames
from pyro_camera_api.camera import base
from pyro_camera_api.camera.adapters.mock import MockCamera
from pyro_camera_api.camera.registry import CAMERA_REGISTRY


def _jpeg(seed=0, size=(64, 48), fmt="JPEG"):
    pixels = np.random.default_rng(seed).integers(0, 255, (size[1], size[0], 3), dtype=np.uint8)
    buffer = BytesIO()
    Image.fromarray(pixels).save(buffer, format=fmt, quality=50)
    return buffer.getvalue()


@pytest.fixture
def cam():
    camera = MockCamera(camera_id="store-cam", cam_type="ptz", cam_poses=[0])
    CAMERA_REGISTRY["store-cam"] = camera
    yield camera
    CAMERA_REGISTRY.pop("store-cam", None)


def test_camera_jpeg_is_stored_and_served_untouched(cam):
    raw = _jpeg()
    frame = cam.store_image(0, base.open_snapshot(raw))
    assert frame.jpeg == raw
    response = get_latest_image("store-cam", 0, quality=None, max_age_s=300)
    assert response.body == raw
    assert response.headers["x-frame-id"] == frame.id
    # An explicit quality still re-encodes
    assert get_latest_image("store-cam", 0, quality=95, max_age_s=300).body != raw


def test_other_images_are_encoded_once(cam):
    # A PNG answer and a derived image carry no camera JPEG: they are stored as JPEG all the same
    png = base.open_snapshot(_jpeg(fmt="PNG"))
    resized = base.open_snapshot(_jpeg()).resize((32, 24))
    for image in (png, resized):
        assert not hasattr(image, "original_jpeg")
        assert Image.open(BytesIO(cam.store_image(0, image).jpeg)).format == "JPEG"


def test_truncated_jpeg_is_refused():
    assert base.open_snapshot(_jpeg(size=(4, 4))).size == (4, 4)  # smaller than the reduced decode
    with pytest.raises(OSError, match="truncated"):
        base.open_snapshot(_jpeg(size=(640, 480))[:-2000])


def test_buffer_keeps_the_last_captures_of_each_pose(cam, monkeypatch):
    monkeypatch.setattr(base, "FRAME_BUFFER_SIZE", 3)
    frames = [cam.store_image(0, base.open_snapshot(_jpeg(seed))) for seed in range(5)]
    assert [f.id for f in cam.frames(0)] == [f.id for f in frames[-3:]]
    assert [f["id"] for f in list_frames("store-cam", 0)["frames"]] == [f.id for f in frames[-3:]]
    assert cam.latest_frames()[0] is frames[-1]
    with pytest.raises(HTTPException) as expired:
        get_frame("store-cam", 0, frames[0].id, index=-1, crop=None, width=None, quality=90)
    assert expired.value.status_code == 404
    # By position from the end: -1 is the last capture, -3 the oldest one kept
    by_index = {i: get_frame("store-cam", 0, None, index=i, crop=None, width=None, quality=90) for i in (-1, -3)}
    assert by_index[-1].headers["x-frame-id"] == frames[-1].id
    assert by_index[-3].headers["x-frame-id"] == frames[-3].id
    with pytest.raises(HTTPException) as too_far:
        get_frame("store-cam", 0, None, index=-4, crop=None, width=None, quality=90)
    assert too_far.value.status_code == 404
    cam.clear_frames()
    assert cam.frames(0) == []


def test_frame_by_id_as_stored_cropped_or_resized(cam):
    raw = _jpeg(size=(640, 480))
    frame = cam.store_image(0, base.open_snapshot(raw))

    def size(**kwargs):
        params = {"index": -1, "crop": None, "width": None, "quality": 90, **kwargs}
        return Image.open(BytesIO(get_frame("store-cam", 0, frame.id, **params).body)).size

    assert get_frame("store-cam", 0, frame.id, index=-1, crop=None, width=None, quality=90).body == raw
    assert size(width=320) == (320, 240)
    assert size(crop="100,50,300,150") == (200, 100)
    assert size(crop="100,50,300,150", width=100) == (100, 50)
    with pytest.raises(HTTPException) as outside:
        size(crop="600,0,700,100")
    assert outside.value.status_code == 422
