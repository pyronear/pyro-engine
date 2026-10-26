# Copyright (C) 2022-2026, Pyronear.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

import threading
import time

from PIL import Image

from pyro_camera_api.api import routes_health
from pyro_camera_api.api.routes_cameras import get_latest_image
from pyro_camera_api.api.routes_health import health
from pyro_camera_api.camera import patrol
from pyro_camera_api.camera.adapters.mock import MockCamera
from pyro_camera_api.camera.registry import CAMERA_REGISTRY


def test_latest_image_refuses_stale_frame(monkeypatch):
    # A hung patrol thread leaves an old frame in the cache: it must not be served
    cam_id = "stale-cam"
    cam = MockCamera(camera_id=cam_id, cam_type="ptz", cam_poses=[0])
    CAMERA_REGISTRY[cam_id] = cam
    try:
        assert get_latest_image(cam_id, 0, quality=95, max_age_s=300).status_code == 204

        cam.store_image(0, Image.new("RGB", (8, 8)))
        assert get_latest_image(cam_id, 0, quality=95, max_age_s=300).media_type == "image/jpeg"

        monkeypatch.setattr("time.monotonic", lambda: cam.last_image_at[0] + 301)
        assert get_latest_image(cam_id, 0, quality=95, max_age_s=300).status_code == 204
    finally:
        CAMERA_REGISTRY.pop(cam_id, None)


def test_watchdog_flags_only_hung_patrol_threads(monkeypatch):
    # A live thread silent for too long is hung, a finished one is not
    hung, done = threading.Event(), threading.Event()
    threads = {
        "hung": threading.Thread(target=hung.wait, daemon=True),
        "busy": threading.Thread(target=hung.wait, daemon=True),
        "done": threading.Thread(target=done.set),
    }
    for thread in threads.values():
        thread.start()
    threads["done"].join()
    monkeypatch.setattr(patrol, "PATROL_THREADS", threads)
    now = time.monotonic()
    monkeypatch.setattr(
        patrol, "PATROL_PROGRESS", {"hung": now - patrol.STALL_LIMIT - 1, "busy": now, "done": now - 3600}
    )
    try:
        assert patrol.stalled_patrols() == ["hung"]
    finally:
        hung.set()


def test_health_reports_stalled_patrols(monkeypatch):
    assert health() == {"status": "ok"}
    monkeypatch.setattr(routes_health, "stalled_patrols", lambda: ["192.168.1.12"])
    assert health().status_code == 503
