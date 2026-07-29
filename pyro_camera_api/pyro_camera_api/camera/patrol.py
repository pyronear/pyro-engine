# Copyright (C) 2022-2026, Pyronear.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.


from __future__ import annotations

import logging
import threading
import time
from typing import Any, Dict

from pyro_camera_api.camera.registry import CAMERA_REGISTRY

logger = logging.getLogger(__name__)

# backoff settings for static cameras
MAX_FAILS_BEFORE_SKIP = 2  # skip after 2 consecutive failures
SKIP_DURATION = 30 * 60.0  # skip for 30 minutes

# per camera state for the static loop
FAILURE_COUNT: Dict[str, int] = {}
SKIP_UNTIL: Dict[str, float] = {}


def _is_thread_alive(obj: object) -> bool:
    try:
        thr = getattr(obj, "_thread", None)
        return isinstance(thr, threading.Thread) and thr.is_alive()
    except Exception:
        return False


def is_stream_running_for(app: Any, camera_ip: str) -> bool:
    """
    True if a pipeline or an ffmpeg restream is active for this camera.
    Expects app.state.stream_workers and app.state.stream_processes to be set in lifespan.
    """
    try:
        workers = getattr(app.state, "stream_workers", {})
        procs = getattr(app.state, "stream_processes", {})
    except Exception:
        return False

    # three worker pipeline
    p = workers.get(camera_ip)
    if p is not None:
        try:
            if _is_thread_alive(p.decoder) and _is_thread_alive(p.encoder):
                return True
        except Exception as exc:
            logger.debug("Could not check anonymizer pipeline state for %s: %s", camera_ip, exc)

    # plain ffmpeg restream
    proc = procs.get(camera_ip)
    if proc is not None:
        try:
            if proc.poll() is None:
                return True
        except Exception as exc:
            logger.debug("Could not check ffmpeg process state for %s: %s", camera_ip, exc)

    return False


def patrol_loop(camera_ip: str, stop_flag: threading.Event) -> None:
    cam = CAMERA_REGISTRY[camera_ip]
    poses = getattr(cam, "cam_poses", []) or []

    if not poses:
        logger.warning("[%s] No poses defined, exiting patrol loop", camera_ip)
        return

    logger.info("[%s] Starting patrol cycle with %d poses", camera_ip, len(poses))

    while not stop_flag.is_set():
        start_time = time.time()
        captured = 0
        failed = 0

        for pose in poses:
            if stop_flag.is_set():
                break

            try:
                cam.move_camera("ToPos", idx=pose, speed=50)
                logger.debug("[%s] Moved to pose %s", camera_ip, pose)
                time.sleep(1.5)

                image = cam.capture()
                if image:
                    cam.last_images[pose] = image
                    captured += 1
                    logger.debug("[%s] Stored image for pose %s", camera_ip, pose)
                else:
                    failed += 1
                    logger.debug("[%s] Capture returned no image for pose %s", camera_ip, pose)

            except Exception as exc:
                failed += 1
                logger.error("[%s] Error at pose %s: %s", camera_ip, pose, exc)
                continue

        try:
            cam.move_camera("ToPos", idx=poses[0], speed=50)
            logger.debug("[%s] Returned to pose %s", camera_ip, poses[0])
        except Exception as exc:
            logger.warning("[%s] Failed to return to first pose: %s", camera_ip, exc)

        if getattr(cam, "focus_position", None) is not None:
            try:
                if cam.focus_position is not None:
                    cam.set_manual_focus(cam.focus_position)
                logger.debug("[%s] Restored manual focus to %s", camera_ip, cam.focus_position)
            except Exception as exc:
                logger.warning("[%s] Failed to restore focus: %s", camera_ip, exc)

        elapsed = time.time() - start_time
        logger.info(
            "[%s] Patrol cycle: captured=%d/%d failed=%d duration=%.1fs",
            camera_ip,
            captured,
            len(poses),
            failed,
            elapsed,
        )
        sleep_time = max(0.0, 30.0 - elapsed)
        stop_flag.wait(sleep_time)

    logger.info("[%s] Patrol loop exited cleanly", camera_ip)


def static_loop(camera_ip: str, stop_flag: threading.Event) -> None:
    cam = CAMERA_REGISTRY[camera_ip]
    logger.info("[%s] Starting static camera loop", camera_ip)

    # init per camera state
    FAILURE_COUNT.setdefault(camera_ip, 0)
    SKIP_UNTIL.setdefault(camera_ip, 0.0)

    settle_until = 0.0

    while not stop_flag.is_set():
        now = time.time()

        # skip window: entering it is already logged once, so stay quiet while it lasts
        if now < SKIP_UNTIL[camera_ip]:
            left = int(SKIP_UNTIL[camera_ip] - now)
            logger.debug("[%s] Skipped for %ds due to previous failures", camera_ip, left)
        else:
            try:
                # capture with internal timeout handled by RTSP or URL adapters
                image = cam.capture()
                now = time.time()

                # after a reconnect, wait a bit before storing
                opened_at = getattr(cam, "_opened_at", 0.0)
                if opened_at:
                    settle_until = opened_at + 1.0

                if image and now >= settle_until:
                    cam.last_images[-1] = image
                    # Steady state is silent, only the recovery is worth an INFO line.
                    if FAILURE_COUNT[camera_ip]:
                        logger.info(
                            "[%s] Capture recovered after %d failures", camera_ip, FAILURE_COUNT[camera_ip]
                        )
                    else:
                        logger.debug("[%s] Updated static image (pose -1)", camera_ip)
                    # success reset failure counter and clear skip
                    FAILURE_COUNT[camera_ip] = 0
                    SKIP_UNTIL[camera_ip] = 0.0
                else:
                    FAILURE_COUNT[camera_ip] += 1
                    logger.warning(
                        "[%s] Capture returned no image, failures=%d",
                        camera_ip,
                        FAILURE_COUNT[camera_ip],
                    )
                    if FAILURE_COUNT[camera_ip] >= MAX_FAILS_BEFORE_SKIP:
                        SKIP_UNTIL[camera_ip] = time.time() + SKIP_DURATION
                        logger.error("[%s] Entering skip window for %ds", camera_ip, int(SKIP_DURATION))

            except Exception as exc:
                FAILURE_COUNT[camera_ip] += 1
                logger.warning(
                    "[%s] Error capturing static image: %s, failures=%d",
                    camera_ip,
                    exc,
                    FAILURE_COUNT[camera_ip],
                )
                if FAILURE_COUNT[camera_ip] >= MAX_FAILS_BEFORE_SKIP:
                    SKIP_UNTIL[camera_ip] = time.time() + SKIP_DURATION
                    logger.error("[%s] Entering skip window for %ds", camera_ip, int(SKIP_DURATION))

        # sleep 30 seconds or exit early
        if stop_flag.wait(30.0):
            break

    logger.info("[%s] Static camera loop exited cleanly", camera_ip)
