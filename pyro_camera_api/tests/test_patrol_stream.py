# Copyright (C) 2022-2026, Pyronear.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.


import threading

from PIL import Image

from pyro_camera_api.camera import patrol
from pyro_camera_api.camera.adapters.mock import MockCamera
from pyro_camera_api.camera.registry import CAMERA_REGISTRY


def test_patrol_loop_pauses_while_streaming(monkeypatch):
    cam_id = "patrol-stream"
    cam = MockCamera(camera_id=cam_id, cam_type="ptz", cam_poses=[0, 1])
    cam._cached_image = Image.new("RGB", (8, 8))
    CAMERA_REGISTRY[cam_id] = cam
    try:
        monkeypatch.setattr(patrol, "STREAM_CHECK_INTERVAL", 0.01)
        monkeypatch.setattr(patrol.time, "sleep", lambda _s: None)

        streaming = {"on": True}
        monkeypatch.setattr(patrol, "is_camera_streaming", lambda _ip: streaming["on"])

        moved = threading.Event()
        real_move = cam.move_camera

        def tracking_move(*args, **kwargs):
            moved.set()
            return real_move(*args, **kwargs)

        monkeypatch.setattr(cam, "move_camera", tracking_move)

        stop = threading.Event()
        thread = threading.Thread(target=patrol.patrol_loop, args=(cam_id, stop), daemon=True)
        thread.start()

        # While a stream is active the patrol must not move the camera
        assert not moved.wait(0.3)

        # As soon as the stream stops, the patrol resumes on its own
        streaming["on"] = False
        assert moved.wait(2.0)

        stop.set()
        thread.join(timeout=2.0)
        assert not thread.is_alive()
    finally:
        CAMERA_REGISTRY.pop(cam_id, None)
