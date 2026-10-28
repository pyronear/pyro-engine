# Copyright (C) 2022-2026, Pyronear.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.


"""
Stuck-camera detector and recovery.

Every CHECK_INTERVAL seconds, compare the images the patrol stored for a camera
with the ones seen at the previous check:

- no_fresh_image: no pose got a new image, snapshots are failing.
- frozen: every pose that got a new image has the exact same pixels as before.
  A live sensor never returns the same bytes twice, a frozen camera does.
- not_moving: all PTZ poses look alike, the turret is stuck on one view.
  Thresholds were calibrated on real sdis-77 captures: stuck-episode max
  pairwise Hamming <= 6, working-patrol min pairwise Hamming >= 17.

After CONSECUTIVE_HITS_BEFORE_REBOOT checks in a row, reboot the camera through
its API, at most MAX_REBOOTS_PER_DAY attempts at least REBOOT_COOLDOWN apart.
A camera that does not answer at all is power-cycled by the Shelly watchdog
(watchdog/shelly/main_pi), not here.

ALLOW_STILL_CAMERA=true ignores frozen and not_moving, to work with a camera
that does not move (at home, or a mock). Never set it on a station.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

from pyro_camera_api.camera import patrol
from pyro_camera_api.camera.registry import CAMERA_REGISTRY, PATROL_FLAGS, PATROL_THREADS
from pyro_camera_api.services.stream import is_camera_streaming

logger = logging.getLogger(__name__)

CHECK_INTERVAL = 5 * 60.0  # seconds between checks
STUCK_MAX_HAMMING = 10  # max pairwise distance below which we suspect stuck
CONSECUTIVE_HITS_BEFORE_REBOOT = 2
MIN_POSES_FOR_CHECK = 2
# Fog / low-light scenes collapse to near-uniform gray and produce unstable pHashes.
# Skip the check when the mean per-image gray variance falls below this threshold.
# Calibrated on sun_test data: foggy rounds mean variance <= 332, stuck rounds mean >= 854.
MIN_MEAN_VARIANCE_FOR_CHECK = 500.0
# Same limits as the Shelly watchdog. Attempts are wall-clock times kept in STATE_FILE, so a
# container restart (healthcheck, update) does not hand out a fresh budget.
MAX_REBOOTS_PER_DAY = 3
REBOOT_COOLDOWN = 30 * 60.0
DAY = 24 * 3600.0
STATE_FILE = os.getenv("RECOVERY_STATE_FILE", "data/camera_recovery.json")
# The static loop skips a camera 30 min after two failed captures: give a rebooting camera
# this long to come back before its captures count again.
REBOOT_GRACE = 90.0

ALLOW_STILL_CAMERA = os.getenv("ALLOW_STILL_CAMERA", "false").strip().lower() in ("1", "true", "yes", "on")

# Per camera recovery state, exposed by /patrol_status. The lock covers the attempts deque,
# appended by the detector thread and read by the API.
RECOVERY_STATE: Dict[str, Dict[str, Any]] = {}
_LOCK = threading.Lock()

# pose -> (time the image was stored, md5 of its pixels)
Snapshot = Dict[int, Tuple[float, bytes]]


def _phash(img: Image.Image, hash_size: int = 8, highfreq_factor: int = 4) -> np.ndarray:
    """Classic pHash: downscale to grayscale, DCT, threshold low-freq block on its median."""
    img_size = hash_size * highfreq_factor
    gray = img.convert("L").resize((img_size, img_size), Image.Resampling.LANCZOS)
    arr = np.asarray(gray, dtype=np.float32)
    dct = cv2.dct(arr)
    low = dct[:hash_size, :hash_size]
    med = np.median(low[1:, 1:])
    return (low > med).flatten()


def _hamming(a: np.ndarray, b: np.ndarray) -> int:
    return int(np.count_nonzero(a != b))


def _mean_gray_variance(images: List[Image.Image]) -> float:
    return float(np.mean([np.asarray(im.convert("L"), dtype=np.float32).var() for im in images]))


def _max_pairwise_hamming(images: List[Image.Image]) -> int:
    hashes = [_phash(im) for im in images]
    n = len(hashes)
    return max(_hamming(hashes[i], hashes[j]) for i in range(n) for j in range(i + 1, n))


def _patrol_is_running(camera_ip: str) -> bool:
    thr = PATROL_THREADS.get(camera_ip)
    flag = PATROL_FLAGS.get(camera_ip)
    return bool(thr and thr.is_alive() and flag and not flag.is_set())


def snapshot(images: Dict[int, Optional[Image.Image]], stored_at: Dict[int, float]) -> Snapshot:
    return {
        pose: (stored_at[pose], hashlib.md5(image.tobytes(), usedforsecurity=False).digest())
        for pose, image in images.items()
        if image is not None and pose in stored_at
    }


def detect(prev: Snapshot, snap: Snapshot, ptz_images: List[Image.Image]) -> Optional[str]:
    """Return the problem seen since the previous check, or None when the camera looks fine."""
    advanced = [pose for pose, (ts, _) in snap.items() if pose not in prev or prev[pose][0] != ts]
    if not advanced:
        return "no_fresh_image"
    if all(pose in prev and prev[pose][1] == snap[pose][1] for pose in advanced):
        return "frozen"
    if (
        len(ptz_images) >= MIN_POSES_FOR_CHECK
        and _mean_gray_variance(ptz_images) >= MIN_MEAN_VARIANCE_FOR_CHECK
        and _max_pairwise_hamming(ptz_images) < STUCK_MAX_HAMMING
    ):
        return "not_moving"
    return None


def reboot_allowed(attempts: Deque[float], now: float) -> bool:
    """Budget on reboot attempts, a rejected or timed-out request may still reboot the camera."""
    while attempts and now - attempts[0] > DAY:
        attempts.popleft()
    return len(attempts) < MAX_REBOOTS_PER_DAY and (not attempts or now - attempts[-1] >= REBOOT_COOLDOWN)


def load_attempts(camera_ip: str) -> Deque[float]:
    """Reboot attempts of the last 24 h saved by a previous run, none when the file is unusable."""
    try:
        saved = json.loads(Path(STATE_FILE).read_text()).get(camera_ip, [])
    except (OSError, ValueError, AttributeError):
        return deque()
    now = time.time()
    return deque(float(t) for t in saved if now - float(t) <= DAY)


def save_attempts(camera_ip: str, attempts: Deque[float]) -> None:
    try:
        path = Path(STATE_FILE)
        try:
            state = json.loads(path.read_text())
        except (OSError, ValueError):
            state = {}
        state[camera_ip] = list(attempts)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(state))
    except (OSError, TypeError) as exc:
        logger.warning("[%s] Cannot save reboot attempts to %s: %s", camera_ip, STATE_FILE, exc)


def recovery_status(camera_ip: str) -> Dict[str, Any]:
    state = RECOVERY_STATE.get(camera_ip)
    if state is None:
        return {"status": None, "reboot_attempts_24h": 0, "last_reboot_attempt": None}
    now = time.time()
    with _LOCK:
        attempts = list(state["attempts"])
    return {
        "status": state["status"],
        "reboot_attempts_24h": sum(1 for t in attempts if now - t <= DAY),
        "last_reboot_attempt": state["last_reboot_attempt"],
    }


def stuck_check_loop(camera_ip: str, stop_flag: threading.Event) -> None:
    cam = CAMERA_REGISTRY[camera_ip]
    can_reboot = hasattr(cam, "reboot_camera")
    is_ptz = getattr(cam, "cam_type", "static") == "ptz"
    state: Dict[str, Any] = {"status": "ok", "attempts": load_attempts(camera_ip), "last_reboot_attempt": None}
    RECOVERY_STATE[camera_ip] = state

    logger.info(
        "[%s] Stuck detector started (interval=%ds, consecutive=%d, reboot=%s)",
        camera_ip,
        int(CHECK_INTERVAL),
        CONSECUTIVE_HITS_BEFORE_REBOOT,
        "api" if can_reboot else "unsupported",
    )

    prev: Optional[Snapshot] = None
    hits = 0

    while not stop_flag.wait(CHECK_INTERVAL):
        if not _patrol_is_running(camera_ip) or is_camera_streaming(camera_ip):
            # Night, autofocus or a live stream: no image is expected, start over from a new baseline
            prev, hits = None, 0
            state["status"] = "idle"
            continue

        # One copy, the patrol thread keeps storing images while we read them
        images = dict(cam.last_images)
        snap = snapshot(images, dict(cam.last_image_at))
        if prev is None:
            prev = snap
            state["status"] = "ok"
            continue

        ptz_images = [img for pose, img in images.items() if is_ptz and pose != -1 and img is not None]
        try:
            problem = detect(prev, snap, ptz_images)
        except Exception as exc:
            logger.warning("[%s] Stuck check failed: %s", camera_ip, exc)
            continue
        prev = snap

        if problem in ("frozen", "not_moving") and ALLOW_STILL_CAMERA:
            logger.warning("[%s] Camera looks %s, ignored because ALLOW_STILL_CAMERA is on", camera_ip, problem)
            problem = None

        state["status"] = problem or "ok"
        if problem is None:
            if hits:
                logger.info("[%s] Stuck detector cleared", camera_ip)
            hits = 0
            continue

        hits += 1
        logger.warning("[%s] Camera %s (hit %d/%d)", camera_ip, problem, hits, CONSECUTIVE_HITS_BEFORE_REBOOT)
        if hits < CONSECUTIVE_HITS_BEFORE_REBOOT:
            continue
        if not can_reboot:
            logger.error("[%s] Camera %s and its adapter cannot reboot it", camera_ip, problem)
            continue
        with _LOCK:
            allowed = reboot_allowed(state["attempts"], time.time())
        if not allowed:
            # Logged once per episode: hits keep counting silently until the cooldown ends
            if hits == CONSECUTIVE_HITS_BEFORE_REBOOT:
                logger.error(
                    "[%s] Camera %s, reboot budget spent (%d per 24 h, %d min apart)",
                    camera_ip,
                    problem,
                    MAX_REBOOTS_PER_DAY,
                    int(REBOOT_COOLDOWN // 60),
                )
            continue

        logger.error("[%s] Rebooting camera, %s", camera_ip, problem)
        with _LOCK:
            state["attempts"].append(time.time())
            save_attempts(camera_ip, state["attempts"])
        state["last_reboot_attempt"] = datetime.now(timezone.utc).isoformat()
        ok = False
        try:
            ok = bool(cam.reboot_camera())
        except Exception as exc:
            # The exception text carries the request URL, with the camera password in it
            logger.error("[%s] Reboot raised %s", camera_ip, type(exc).__name__)
        if ok:
            for pose in list(cam.last_images):
                cam.last_images[pose] = None
            # Let the camera boot before its failed captures count towards the 30 min skip
            patrol.FAILURE_COUNT[camera_ip] = 0
            patrol.SKIP_UNTIL[camera_ip] = time.time() + REBOOT_GRACE
        else:
            logger.error("[%s] Reboot command rejected by camera", camera_ip)
        prev, hits = None, 0

    logger.info("[%s] Stuck detector exited cleanly", camera_ip)
