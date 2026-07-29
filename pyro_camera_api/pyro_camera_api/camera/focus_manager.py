# Copyright (C) 2022-2026, Pyronear.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.


"""Two-stage autofocus orchestration.

Stage 1: full_calibration runs the adapter's focus_finder search once (at
patrol startup, or on demand via the /focus/focus_finder route) and stores the
result in cam.focus_position, which acts as the per-camera reference.

Stage 2: fine_adjustment probes a few positions around the reference between
patrol rounds and moves the reference only when a candidate is clearly sharper.

Both stages are skipped while a stream is active for the camera and abort
mid-run if one starts, so live viewers never see a focus sweep. The per-camera
MOVE_LOCKS serialize them against manual PTZ commands.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

import cv2
import numpy as np
from PIL import Image

from pyro_camera_api.camera.base import BaseCamera, FocusMixin, PTZMixin
from pyro_camera_api.camera.registry import MOVE_LOCKS
from pyro_camera_api.services.stream import (
    get_app_for_stream,
    get_processes,
    get_workers,
    is_pipeline_running,
    is_process_running,
)

logger = logging.getLogger(__name__)

# Seconds between two fine adjustments of the same camera
FINE_TUNE_INTERVAL = 30 * 60.0
# Offsets probed around the reference focus position
FINE_TUNE_OFFSETS = (-4, -2, 2, 4)
# Relative sharpness gain required before moving the reference
FINE_TUNE_MIN_GAIN = 1.10
# Seconds to wait after a focus move before capturing
FOCUS_SETTLE_TIME = 2.0
# Seconds to let the turret settle on the calibration pose before capturing
POSE_SETTLE_TIME = 3.0


def stream_is_active(camera_id: str) -> bool:
    """True if a live pipeline or ffmpeg restream is running for this camera."""
    app = get_app_for_stream()
    if app is None:
        return False
    try:
        if is_pipeline_running(get_workers(app).get(camera_id)):
            return True
        return is_process_running(get_processes(app).get(camera_id))
    except Exception as exc:
        logger.debug("Could not check stream state for %s: %s", camera_id, exc)
        return False


def measure_sharpness(image: Image.Image) -> float:
    """Variance of the Laplacian over the grayscale image."""
    arr = np.array(image.convert("L"))
    return float(cv2.Laplacian(arr, cv2.CV_64F).var())


def _move_to_calibration_pose(cam: BaseCamera) -> None:
    """
    Point the camera at its calibration pose (second preset when available).

    Both stages measure sharpness on this pose so the reference is always
    optimized for the same scene. Move failures are logged, not fatal.
    """
    if not isinstance(cam, PTZMixin):
        return
    poses = getattr(cam, "cam_poses", []) or []
    if not poses:
        return
    pose = poses[1] if len(poses) > 1 else poses[0]
    try:
        cam.move_camera("ToPos", idx=pose, speed=50)
        time.sleep(POSE_SETTLE_TIME)
    except Exception as exc:
        logger.warning("[%s] Could not move to calibration pose %s: %s", cam.camera_id, pose, exc)


def full_calibration(cam: BaseCamera, save_images: bool = False) -> Optional[int]:
    """
    Run the adapter's full focus search and store the result as reference.

    The camera is moved to its calibration pose first, under the same lock,
    so no other PTZ command can interleave. Returns the best focus position,
    or None when the calibration was skipped (unsupported camera, active
    stream, or camera busy).
    """
    if not isinstance(cam, FocusMixin) or cam.cam_type == "static":
        return None

    if stream_is_active(cam.camera_id):
        logger.info("[%s] Skipping focus calibration, stream active", cam.camera_id)
        return None

    lock = MOVE_LOCKS[cam.camera_id]
    if not lock.acquire(blocking=False):
        logger.info("[%s] Skipping focus calibration, camera busy", cam.camera_id)
        return None
    try:
        _move_to_calibration_pose(cam)
        best = cam.focus_finder(save_images=save_images, should_abort=lambda: stream_is_active(cam.camera_id))
        logger.info("[%s] Focus calibration done, reference=%s", cam.camera_id, best)
        return int(best)
    finally:
        lock.release()


def fine_adjustment(cam: BaseCamera) -> Optional[int]:
    """
    Probe a few positions around the reference focus and keep the sharpest.

    The reference (cam.focus_position) moves only when a candidate beats it by
    FINE_TUNE_MIN_GAIN, so noise in the sharpness measure does not make the
    focus drift. Returns the reference in use, or None when skipped.
    """
    if not isinstance(cam, FocusMixin) or cam.cam_type == "static":
        return None

    if cam.focus_position is None:
        return None
    reference = int(cam.focus_position)

    if stream_is_active(cam.camera_id):
        return None

    lock = MOVE_LOCKS[cam.camera_id]
    if not lock.acquire(blocking=False):
        return None
    try:

        def probe(position: int) -> Optional[float]:
            cam.set_manual_focus(position)
            time.sleep(FOCUS_SETTLE_TIME)
            image = cam.capture()
            if image is None:
                return None
            return measure_sharpness(image)

        _move_to_calibration_pose(cam)

        ref_score = probe(reference)
        if ref_score is None:
            logger.warning("[%s] Fine adjustment skipped, no image at reference", cam.camera_id)
            return None

        best_pos, best_score = reference, ref_score
        for offset in FINE_TUNE_OFFSETS:
            if stream_is_active(cam.camera_id):
                logger.info("[%s] Fine adjustment aborted, stream started", cam.camera_id)
                break
            candidate = reference + offset
            score = probe(candidate)
            if score is not None and score > best_score:
                best_pos, best_score = candidate, score

        if best_pos != reference and best_score >= ref_score * FINE_TUNE_MIN_GAIN:
            logger.info(
                "[%s] Fine adjustment moved reference %s -> %s (sharpness %.2f -> %.2f)",
                cam.camera_id,
                reference,
                best_pos,
                ref_score,
                best_score,
            )
            reference = best_pos

        return reference
    finally:
        # Always leave the camera on a validated reference, even when a probe
        # raised mid-run and left the motor on an unvalidated candidate.
        cam.focus_position = reference
        try:
            cam.set_manual_focus(reference)
        except Exception as exc:
            logger.warning("[%s] Could not restore reference focus %s: %s", cam.camera_id, reference, exc)
        lock.release()
