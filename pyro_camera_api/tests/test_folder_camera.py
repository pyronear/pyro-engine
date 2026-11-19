# Copyright (C) 2022-2026, Pyronear.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.


import time
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image

from pyro_camera_api.api.routes_cameras import router
from pyro_camera_api.camera.adapters.folder import FolderCamera
from pyro_camera_api.camera.registry import CAMERA_REGISTRY, build_camera_object


def _write_frame(tmp_path, pose_id, age_s, color=(10, 20, 30), platform_id=42):
    captured_at = (datetime.now(timezone.utc) - timedelta(seconds=age_s)).replace(microsecond=0)
    directory = tmp_path / str(platform_id) / str(pose_id)
    directory.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), color).save(directory / f"{captured_at:%Y%m%dT%H%M%SZ}.jpg", format="JPEG")
    return captured_at


def _ptz_camera(tmp_path, ignore_frames_older_than_s=1200.0):
    return FolderCamera(
        "cam",
        str(tmp_path),
        platform_id=42,
        pose_ids=[317, 318],
        cam_type="ptz",
        cam_poses=[8, 9],
        ignore_frames_older_than_s=ignore_frames_older_than_s,
    )


def test_poll_serves_frames_oldest_first_once(tmp_path):
    first = _write_frame(tmp_path, 317, 300)
    second = _write_frame(tmp_path, 317, 60)
    cam = _ptz_camera(tmp_path)

    cam.poll_frames()
    assert cam.last_images[8].info["captured_at"] == first.isoformat()

    # An unread frame holds the slot
    cam.poll_frames()
    assert cam.last_images[8].info["captured_at"] == first.isoformat()

    cam.last_images.pop(8)
    cam.poll_frames()
    assert cam.last_images[8].info["captured_at"] == second.isoformat()

    # Nothing new: the pose stays empty instead of serving the same frame again
    cam.last_images.pop(8)
    cam.poll_frames()
    assert 8 not in cam.last_images


def test_poll_skips_frames_over_max_age(tmp_path):
    _write_frame(tmp_path, 317, 5000)
    cam = _ptz_camera(tmp_path)

    cam.poll_frames()

    assert cam.last_images == {}


def test_poll_keeps_poses_separate(tmp_path):
    _write_frame(tmp_path, 318, 60)
    cam = _ptz_camera(tmp_path)

    cam.poll_frames()

    assert list(cam.last_images) == [9]


@pytest.mark.parametrize(
    "name",
    ["20261030T084051Z.jpg.tmp", "CAM~8_20261030084051_P194T-2.8Z15.9.jpg", "20261399T084051Z.jpg", "notes.txt"],
)
def test_poll_ignores_other_files(tmp_path, name):
    directory = tmp_path / "42" / "317"
    directory.mkdir(parents=True)
    (directory / name).write_bytes(b"partial")
    cam = _ptz_camera(tmp_path, ignore_frames_older_than_s=10**10)

    cam.poll_frames()

    assert cam.last_images == {}


def test_poll_moves_past_unreadable_frame(tmp_path):
    good = _write_frame(tmp_path, 317, 60)
    bad = (datetime.now(timezone.utc) - timedelta(seconds=120)).strftime("%Y%m%dT%H%M%SZ")
    (tmp_path / "42" / "317" / f"{bad}.jpg").write_bytes(b"not an image")
    cam = _ptz_camera(tmp_path)

    cam.poll_frames()

    assert cam.last_images[8].info["captured_at"] == good.isoformat()


def test_poll_handles_missing_folder(tmp_path):
    cam = _ptz_camera(tmp_path)

    cam.poll_frames()

    assert cam.last_images == {}


def test_static_camera_uses_static_pose(tmp_path):
    _write_frame(tmp_path, 317, 60)
    cam = FolderCamera("cam", str(tmp_path), platform_id=42, pose_ids=[317])

    cam.poll_frames()

    assert list(cam.last_images) == [-1]
    assert cam.capture() is not None


def test_capture_returns_latest_without_changing_poll(tmp_path):
    first = _write_frame(tmp_path, 317, 300)
    latest = _write_frame(tmp_path, 317, 60)
    cam = _ptz_camera(tmp_path)

    assert cam.capture(patrol_id=8).info["captured_at"] == latest.isoformat()

    cam.poll_frames()
    assert cam.last_images[8].info["captured_at"] == first.isoformat()


def test_capture_without_frame_or_with_unknown_pose(tmp_path):
    _write_frame(tmp_path, 317, 5000)
    cam = _ptz_camera(tmp_path)

    assert cam.capture(patrol_id=8) is None
    assert cam.capture(patrol_id=99) is None


def test_registry_builds_folder_camera(tmp_path):
    conf = {
        "adapter": "folder",
        "type": "ptz",
        "id": 42,
        "poses": [8],
        "pose_ids": [317],
        "folder_path": str(tmp_path),
        "ignore_frames_older_than_s": 60,
    }
    cam = build_camera_object("cam", conf)

    assert isinstance(cam, FolderCamera)
    assert cam.ignore_frames_older_than_s == 60
    assert cam.pose_dirs == {8: tmp_path / "42" / "317"}


@pytest.mark.parametrize(
    "missing",
    [
        {"folder_path": None},
        {"id": None},
        {"pose_ids": []},
        {"poses": [8, 9]},
    ],
)
def test_registry_rejects_incomplete_config(tmp_path, missing):
    conf = {"adapter": "folder", "type": "ptz", "id": 42, "poses": [8], "pose_ids": [317], "folder_path": str(tmp_path)}

    assert build_camera_object("cam", {**conf, **missing}) is None


def test_latest_image_serves_folder_frame_once_with_capture_time(tmp_path, monkeypatch):
    captured_at = _write_frame(tmp_path, 317, 60)
    cam = _ptz_camera(tmp_path)
    cam.poll_frames()
    monkeypatch.setitem(CAMERA_REGISTRY, "cam", cam)
    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)

    first = client.get("/latest_image", params={"camera_ip": "cam", "pose": 8})
    assert first.status_code == 200
    assert first.headers["X-Captured-At"] == captured_at.isoformat()

    assert client.get("/latest_image", params={"camera_ip": "cam", "pose": 8}).status_code == 204


def test_latest_image_keeps_frame_for_other_adapters(monkeypatch):
    cam = MagicMock(consume_on_read=False)
    cam.last_images = {-1: Image.new("RGB", (8, 8))}
    cam.last_image_at = {-1: time.monotonic()}
    monkeypatch.setitem(CAMERA_REGISTRY, "other", cam)
    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)

    for _ in range(2):
        response = client.get("/latest_image", params={"camera_ip": "other", "pose": -1})
        assert response.status_code == 200
        assert "X-Captured-At" not in response.headers
