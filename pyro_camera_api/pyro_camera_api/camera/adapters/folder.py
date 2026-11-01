# Copyright (C) 2022-2026, Pyronear.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.


from __future__ import annotations

import logging
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image

from pyro_camera_api.camera.base import BaseCamera

__all__ = ["FolderCamera"]

logger = logging.getLogger(__name__)

FRAME_NAME = re.compile(r"\d{8}T\d{6}Z\.jpg")
FRAME_TIME_FORMAT = "%Y%m%dT%H%M%SZ"

# Pose key used by static cameras in last_images, as in the static loop
STATIC_POSE = -1


class FolderCamera(BaseCamera):
    """
    Camera whose images are deposited on disk instead of being fetched from a device.

    Images are read from {folder_path}/{platform_id}/{pose_id}/{YYYYMMDDTHHMMSSZ}.jpg,
    with the capture time in UTC. Files must appear atomically (written under another
    name, then renamed). The folder is only ever read.

    Each frame is handed over once: poll_frames() fills an empty pose slot with the
    oldest unread frame, and /latest_image empties the slot when it serves it. The
    read position is kept in memory, so a restart reads again what is under max_age_s.
    """

    consume_on_read = True

    def __init__(
        self,
        camera_id: str,
        folder_path: str,
        platform_id: int,
        pose_ids: List[int],
        cam_type: str = "static",
        cam_poses: Optional[List[int]] = None,
        max_age_s: float = 1200.0,
    ) -> None:
        super().__init__(camera_id=camera_id, cam_type=cam_type)
        self.max_age_s = max_age_s
        self.cam_poses = cam_poses or []

        root = Path(folder_path) / str(platform_id)
        if cam_type == "ptz":
            self.pose_dirs = {pose: root / str(pose_id) for pose, pose_id in zip(self.cam_poses, pose_ids, strict=True)}
        else:
            self.pose_dirs = {STATIC_POSE: root / str(pose_ids[0])}

        self._last_read: Dict[int, datetime] = {}

    @staticmethod
    def _list_frames(directory: Path, oldest: datetime) -> List[Tuple[datetime, Path]]:
        """Return the frames of a pose captured at or after oldest, oldest first."""
        try:
            names = [path.name for path in directory.iterdir()]
        except OSError as exc:
            logger.debug("Cannot list %s: %s", directory, exc)
            return []

        frames = []
        for name in names:
            if not FRAME_NAME.fullmatch(name):
                continue
            try:
                captured_at = datetime.strptime(name[:-4], FRAME_TIME_FORMAT).replace(tzinfo=timezone.utc)
            except ValueError:
                continue
            if captured_at >= oldest:
                frames.append((captured_at, directory / name))
        frames.sort()
        return frames

    def _load_frame(self, path: Path, captured_at: datetime) -> Optional[Image.Image]:
        try:
            with Image.open(path) as source:
                image = source.convert("RGB")
        except Exception as exc:
            logger.error("[%s] Could not read %s: %s", self.camera_id, path, exc)
            return None
        image.info["captured_at"] = captured_at.isoformat()
        return image

    def _oldest_allowed(self) -> datetime:
        return datetime.now(timezone.utc) - timedelta(seconds=self.max_age_s)

    def poll_frames(self) -> None:
        """Fill each empty pose slot with the oldest frame that was not read yet."""
        oldest = self._oldest_allowed()
        for pose, directory in self.pose_dirs.items():
            if pose in self.last_images:
                continue
            last_read = self._last_read.get(pose)
            for captured_at, path in self._list_frames(directory, oldest):
                if last_read is not None and captured_at <= last_read:
                    continue
                self._last_read[pose] = captured_at
                image = self._load_frame(path, captured_at)
                if image is not None:
                    self.store_image(pose, image)
                    logger.info("[%s] Stored frame %s for pose %s", self.camera_id, path.name, pose)
                    break

    def capture(self, patrol_id: Optional[int] = None, **kwargs) -> Optional[Image.Image]:
        """Return the most recent frame of a pose, without changing what poll_frames() serves."""
        _ = kwargs
        pose = patrol_id if patrol_id is not None else next(iter(self.pose_dirs), None)
        directory = self.pose_dirs.get(pose) if pose is not None else None
        if directory is None:
            logger.error("[%s] Unknown pose %s", self.camera_id, patrol_id)
            return None

        for captured_at, path in reversed(self._list_frames(directory, self._oldest_allowed())):
            image = self._load_frame(path, captured_at)
            if image is not None:
                return image

        logger.warning("[%s] No frame under %ss in %s", self.camera_id, self.max_age_s, directory)
        return None
