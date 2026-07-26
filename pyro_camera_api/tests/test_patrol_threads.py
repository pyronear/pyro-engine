# Copyright (C) 2022-2026, Pyronear.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.


import threading
from typing import Optional

from PIL import Image

from pyro_camera_api.camera import patrol
from pyro_camera_api.camera.base import BaseCamera
from pyro_camera_api.camera.registry import CAMERA_REGISTRY, PATROL_FLAGS, PATROL_THREADS


class _FakeStaticCamera(BaseCamera):
    def __init__(self):
        super().__init__(camera_id="fake", cam_type="static")

    def capture(self, **kwargs) -> Optional[Image.Image]:
        _ = kwargs  # unused, signature imposed by BaseCamera
        return Image.new("RGB", (8, 8), (255, 200, 200))


def _cleanup(camera_ip, extra_threads=()):
    flag = PATROL_FLAGS.pop(camera_ip, None)
    if flag is not None:
        flag.set()
    thread = PATROL_THREADS.pop(camera_ip, None)
    for thr in (thread, *extra_threads):
        if thr is not None and thr.is_alive():
            thr.join(timeout=2)
    CAMERA_REGISTRY.pop(camera_ip, None)
    patrol.FAILURE_COUNT.pop(camera_ip, None)
    patrol.SKIP_UNTIL.pop(camera_ip, None)


def test_start_patrol_thread_idempotent_when_running():
    camera_ip = "10.0.0.98"
    CAMERA_REGISTRY[camera_ip] = _FakeStaticCamera()
    release = threading.Event()
    running = threading.Thread(target=release.wait, daemon=True)
    running.start()
    PATROL_THREADS[camera_ip] = running
    PATROL_FLAGS[camera_ip] = threading.Event()  # not set: genuinely running

    try:
        status, loop_type = patrol.start_patrol_thread(camera_ip)
        assert status == "already_running"
        assert loop_type == "static"
        assert PATROL_THREADS[camera_ip] is running
    finally:
        release.set()
        _cleanup(camera_ip)


def test_start_patrol_thread_replaces_zombie_thread(monkeypatch):
    """Regression: a thread stuck after stop_patrol must not block restarts.

    Before the fix, start_patrol saw the old thread alive and returned
    already_running without clearing the set stop flag, leaving the patrol
    dead forever while its images went stale.
    """
    monkeypatch.setattr(patrol, "STOPPING_THREAD_JOIN_TIMEOUT", 0.05)
    camera_ip = "10.0.0.99"
    CAMERA_REGISTRY[camera_ip] = _FakeStaticCamera()

    release = threading.Event()
    zombie = threading.Thread(target=release.wait, daemon=True)
    zombie.start()
    old_flag = threading.Event()
    old_flag.set()  # stop was requested but the thread never exited
    PATROL_THREADS[camera_ip] = zombie
    PATROL_FLAGS[camera_ip] = old_flag

    try:
        status, loop_type = patrol.start_patrol_thread(camera_ip)
        assert status == "started"
        assert loop_type == "static"
        assert PATROL_THREADS[camera_ip] is not zombie
        assert PATROL_THREADS[camera_ip].is_alive()
        assert PATROL_FLAGS[camera_ip] is not old_flag
        assert not PATROL_FLAGS[camera_ip].is_set()
    finally:
        release.set()
        _cleanup(camera_ip, extra_threads=(zombie,))
