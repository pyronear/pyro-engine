# Copyright (C) 2022-2026, Pyronear.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.


"""
PTZ stuck-camera detector.

Every CHECK_INTERVAL seconds, compute pairwise pHash distances across the
most recent per-pose images produced by the patrol loop. A turret that has
frozen returns near-identical frames for all poses, giving a very small
maximum pairwise distance. After CONSECUTIVE_HITS_BEFORE_REBOOT consecutive
low-distance checks, reboot the camera.

Thresholds were calibrated on real sdis-77 captures: stuck-episode max
pairwise Hamming <= 6, working-patrol min pairwise Hamming >= 17.
"""

from __future__ import annotations

import logging
import threading
from typing import Dict, List

import cv2
import numpy as np
from PIL import Image

from pyro_camera_api.camera.registry import CAMERA_REGISTRY, PATROL_FLAGS, PATROL_THREADS

logger = logging.getLogger(__name__)

CHECK_INTERVAL = 30 * 60.0  # seconds between checks
INITIAL_DELAY = 3 * 60.0  # delay before the first check, lets patrol populate last_images
STUCK_MAX_HAMMING = 10  # max pairwise distance below which we suspect stuck
CONSECUTIVE_HITS_BEFORE_REBOOT = 2
MIN_POSES_FOR_CHECK = 3

CONSECUTIVE_HITS: Dict[str, int] = {}


def _phash(img: Image.Image, hash_size: int = 8, highfreq_factor: int = 4) -> np.ndarray:
    """Classic pHash: downscale to grayscale, DCT, threshold low-freq block on its median."""
    img_size = hash_size * highfreq_factor
    gray = img.convert("L").resize((img_size, img_size), Image.LANCZOS)
    arr = np.asarray(gray, dtype=np.float32)
    dct = cv2.dct(arr)
    low = dct[:hash_size, :hash_size]
    med = np.median(low[1:, 1:])
    return (low > med).flatten()


def _hamming(a: np.ndarray, b: np.ndarray) -> int:
    return int(np.count_nonzero(a != b))


def _max_pairwise_hamming(images: List[Image.Image]) -> int:
    hashes = [_phash(im) for im in images]
    n = len(hashes)
    return max(_hamming(hashes[i], hashes[j]) for i in range(n) for j in range(i + 1, n))


def _patrol_is_running(camera_ip: str) -> bool:
    thr = PATROL_THREADS.get(camera_ip)
    flag = PATROL_FLAGS.get(camera_ip)
    return bool(thr and thr.is_alive() and flag and not flag.is_set())


def stuck_check_loop(camera_ip: str, stop_flag: threading.Event) -> None:
    cam = CAMERA_REGISTRY[camera_ip]

    if not hasattr(cam, "reboot_camera"):
        logger.info("[%s] Stuck detector disabled: adapter does not support reboot", camera_ip)
        return

    logger.info(
        "[%s] Stuck detector started (initial=%ds, interval=%ds, threshold=%d, consecutive=%d)",
        camera_ip,
        int(INITIAL_DELAY),
        int(CHECK_INTERVAL),
        STUCK_MAX_HAMMING,
        CONSECUTIVE_HITS_BEFORE_REBOOT,
    )

    CONSECUTIVE_HITS[camera_ip] = 0
    next_delay = INITIAL_DELAY

    while not stop_flag.wait(next_delay):
        next_delay = CHECK_INTERVAL
        if not _patrol_is_running(camera_ip):
            logger.info("[%s] Stuck check skipped: patrol not running", camera_ip)
            CONSECUTIVE_HITS[camera_ip] = 0
            continue

        images = [im for pose, im in cam.last_images.items() if pose != -1 and im is not None]
        if len(images) < MIN_POSES_FOR_CHECK:
            logger.info(
                "[%s] Stuck check skipped: only %d pose images available",
                camera_ip,
                len(images),
            )
            continue

        try:
            max_dist = _max_pairwise_hamming(images)
        except Exception as exc:
            logger.warning("[%s] Stuck check failed: %s", camera_ip, exc)
            continue

        logger.info(
            "[%s] Stuck check: max pHash distance=%d across %d poses (threshold=%d)",
            camera_ip,
            max_dist,
            len(images),
            STUCK_MAX_HAMMING,
        )

        if max_dist < STUCK_MAX_HAMMING:
            CONSECUTIVE_HITS[camera_ip] += 1
            logger.warning(
                "[%s] Possible stuck PTZ: max pHash distance=%d across %d poses (hit %d/%d)",
                camera_ip,
                max_dist,
                len(images),
                CONSECUTIVE_HITS[camera_ip],
                CONSECUTIVE_HITS_BEFORE_REBOOT,
            )
            if CONSECUTIVE_HITS[camera_ip] >= CONSECUTIVE_HITS_BEFORE_REBOOT:
                logger.error(
                    "[%s] Rebooting camera due to stuck PTZ detection (max distance=%d)",
                    camera_ip,
                    max_dist,
                )
                try:
                    cam.reboot_camera()
                    cam.last_images.clear()
                except Exception as exc:
                    logger.error("[%s] Reboot failed: %s", camera_ip, exc)
                CONSECUTIVE_HITS[camera_ip] = 0
            else:
                # confirm the hit quickly rather than waiting a full interval
                next_delay = INITIAL_DELAY
        else:
            if CONSECUTIVE_HITS[camera_ip] > 0:
                logger.info(
                    "[%s] Stuck detector cleared: max distance=%d",
                    camera_ip,
                    max_dist,
                )
            CONSECUTIVE_HITS[camera_ip] = 0

    logger.info("[%s] Stuck detector exited cleanly", camera_ip)
