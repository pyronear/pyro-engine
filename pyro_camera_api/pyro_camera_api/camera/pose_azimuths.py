# Copyright (C) 2022-2026, Pyronear.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.


"""Resolve the local pose -> azimuth mapping from the platform API.

credentials.json no longer carries azimuths, only local pose presets and
platform pose ids. The platform stores each pose's azimuth together with its
``patrol_id`` (the local pose preset index), so the mapping is fetched with the
camera's own token from ``GET /api/v1/poses/`` and applied to the adapters that
dead-reckon their azimuth (Reolink-style). An ``azimuths`` list present in
credentials.json still takes precedence (legacy configs).
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Dict, List

import requests

from pyro_camera_api.camera.base import PTZMixin
from pyro_camera_api.camera.registry import CAMERA_REGISTRY, RAW_CONFIG

logger = logging.getLogger(__name__)

RETRY_INTERVAL_S = 60.0


def fetch_pose_azimuths(api_url: str, token: str, timeout: float = 10.0) -> Dict[int, float]:
    """Fetch the authenticated camera's active poses and map patrol_id -> azimuth."""
    url = api_url.rstrip("/") + "/api/v1/poses/"
    resp = requests.get(url, headers={"Authorization": f"Bearer {token}"}, timeout=timeout)
    resp.raise_for_status()
    mapping: Dict[int, float] = {}
    for pose in resp.json():
        patrol_id = pose.get("patrol_id")
        if patrol_id is None:
            logger.warning("Pose %s has no patrol_id, skipping", pose.get("id"))
            continue
        mapping[int(patrol_id)] = float(pose["azimuth"]) % 360.0
    return mapping


def resolve_camera_azimuths(cam: PTZMixin, mapping: Dict[int, float]) -> bool:
    """Set cam.cam_azimuths aligned with cam.cam_poses from the fetched mapping.

    Returns False (and leaves the camera untouched) when any local pose has no
    azimuth in the mapping, so a partial list can never desynchronize the
    pose-index-based lookup used by the adapters.
    """
    poses = cam.cam_poses
    missing = [p for p in poses if p not in mapping]
    if missing:
        logger.warning("Poses %s have no azimuth in the platform API (mapping: %s)", missing, mapping)
        return False
    cam.cam_azimuths = [mapping[p] for p in poses]
    return True


def _cameras_needing_azimuths() -> List[str]:
    """Tracked-azimuth PTZ cameras with poses but no azimuth mapping yet."""
    out = []
    for key, cam in CAMERA_REGISTRY.items():
        if not isinstance(cam, PTZMixin) or cam.azimuth_source != "tracked":
            continue
        if not cam.cam_poses or cam.cam_azimuths:
            continue
        out.append(key)
    return out


def azimuth_sync_loop(stop_flag: threading.Event) -> None:
    """Background loop: fetch pose azimuths until every camera is resolved.

    Runs at startup and retries every RETRY_INTERVAL_S so an unreachable
    platform API at boot only delays azimuth availability instead of failing
    camera registration. Exits once nothing is left to resolve.
    """
    api_url = os.environ.get("API_URL", "")
    if not api_url:
        if _cameras_needing_azimuths():
            logger.warning("API_URL not set; pose azimuths cannot be fetched, azimuth will stay unknown")
        return

    no_token: set[str] = set()
    while not stop_flag.is_set():
        pending = [k for k in _cameras_needing_azimuths() if k not in no_token]
        if not pending:
            logger.info("Azimuth sync complete")
            return

        for key in pending:
            cam = CAMERA_REGISTRY[key]
            token = RAW_CONFIG.get(key, {}).get("token")
            if not token:
                logger.warning("[%s] No token in credentials.json; cannot fetch pose azimuths", key)
                no_token.add(key)
                continue
            try:
                mapping = fetch_pose_azimuths(api_url, token)
            except Exception as exc:
                logger.warning("[%s] Failed to fetch pose azimuths, will retry: %s", key, exc)
                continue
            if isinstance(cam, PTZMixin) and resolve_camera_azimuths(cam, mapping):
                logger.info(
                    "[%s] Pose azimuths resolved from platform API: %s",
                    key,
                    dict(zip(cam.cam_poses, cam.cam_azimuths)),
                )

        stop_flag.wait(RETRY_INTERVAL_S)
