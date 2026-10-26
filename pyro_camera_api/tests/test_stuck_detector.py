# Copyright (C) 2022-2026, Pyronear.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

import threading
import time
from collections import deque
from unittest.mock import MagicMock

import numpy as np
import pytest
from PIL import Image

from pyro_camera_api.camera import stuck_detector as sd
from pyro_camera_api.camera.adapters.mock import MockCamera
from pyro_camera_api.camera.registry import CAMERA_REGISTRY


def _noise(seed):
    return Image.fromarray(np.random.default_rng(seed).integers(0, 255, (64, 64, 3), dtype=np.uint8))


def test_detect_no_fresh_image():
    prev = sd.snapshot({0: _noise(0)}, {0: 1.0})
    assert sd.detect(prev, prev, []) == "no_fresh_image"
    # A camera that never produced an image is caught too
    assert sd.detect({}, {}, []) == "no_fresh_image"


def test_detect_frozen_same_pixels_stored_again():
    # Each pose keeps its own frozen image: the Epinay August case, not seen by pHash
    prev = sd.snapshot({0: _noise(0), 1: _noise(1)}, {0: 1.0, 1: 1.0})
    snap = sd.snapshot({0: _noise(0), 1: _noise(1)}, {0: 2.0, 1: 1.0})
    assert sd.detect(prev, snap, []) == "frozen"


def test_detect_healthy_and_not_moving():
    prev = sd.snapshot({0: _noise(0), 1: _noise(1)}, {0: 1.0, 1: 1.0})
    images = {0: _noise(2), 1: _noise(3)}
    snap = sd.snapshot(images, {0: 2.0, 1: 2.0})
    assert sd.detect(prev, snap, list(images.values())) is None
    # Live pixels but every pose shows the same view: the turret does not move
    same_view = [_noise(2), _noise(2)]
    assert sd.detect(prev, snap, same_view) == "not_moving"


def test_reboot_budget():
    attempts: deque[float] = deque()
    assert sd.reboot_allowed(attempts, 0.0)
    attempts.append(0.0)
    assert not sd.reboot_allowed(attempts, sd.REBOOT_COOLDOWN - 1)
    for t in (sd.REBOOT_COOLDOWN, 2 * sd.REBOOT_COOLDOWN):
        assert sd.reboot_allowed(attempts, t)
        attempts.append(t)
    assert not sd.reboot_allowed(attempts, 10 * sd.REBOOT_COOLDOWN)
    # The oldest attempt leaves the 24 h window
    assert sd.reboot_allowed(attempts, sd.DAY + 1)


@pytest.mark.parametrize("allow_still", [False, True])
def test_frozen_camera_is_rebooted_once_unless_allowed(monkeypatch, allow_still):
    cam_id = f"frozen-{allow_still}"
    cam = MockCamera(camera_id=cam_id, cam_type="static")
    cam.reboot_camera = MagicMock(return_value=True)
    CAMERA_REGISTRY[cam_id] = cam
    monkeypatch.setattr(sd, "CHECK_INTERVAL", 0.02)
    monkeypatch.setattr(sd, "ALLOW_STILL_CAMERA", allow_still)
    monkeypatch.setattr(sd, "_patrol_is_running", lambda _ip: True)
    monkeypatch.setattr(sd, "is_camera_streaming", lambda _ip: False)

    stop = threading.Event()
    frozen = _noise(0)

    def frozen_patrol():
        while not stop.wait(0.005):
            cam.store_image(-1, frozen)

    threads = [
        threading.Thread(target=frozen_patrol, daemon=True),
        threading.Thread(target=sd.stuck_check_loop, args=(cam_id, stop), daemon=True),
    ]
    try:
        for thread in threads:
            thread.start()
        time.sleep(0.5)
    finally:
        stop.set()
        for thread in threads:
            thread.join(timeout=2.0)
        CAMERA_REGISTRY.pop(cam_id, None)

    # The cooldown keeps it to a single attempt
    assert cam.reboot_camera.call_count == (0 if allow_still else 1)
    assert sd.recovery_status(cam_id)["reboots_24h"] == (0 if allow_still else 1)
