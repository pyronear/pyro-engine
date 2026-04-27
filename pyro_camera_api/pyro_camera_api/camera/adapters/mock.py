# Copyright (C) 2022-2026, Pyronear.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.


from __future__ import annotations

import logging
import zipfile
from io import BytesIO
from operator import itemgetter
from threading import Lock
from typing import Optional

import requests
from PIL import Image

from pyro_camera_api.camera.base import BaseCamera, FocusMixin, PTZMixin

__all__ = ["MockCamera"]

logger = logging.getLogger(__name__)

DEFAULT_FAKE_IMAGE_URL = "https://github.com/user-attachments/files/27112977/41_croix-augas-02_2025-07-14_11-24-31.zip"

_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


class MockCamera(BaseCamera, PTZMixin, FocusMixin):
    """
    Mock camera adapter for development and testing.

    The source URL can point to a single image or to a ZIP archive containing
    an ``images/`` folder. In the ZIP case, the adapter extracts every image
    in that folder, sorts them by filename, caches them in memory, and loops
    over the sequence on successive capture calls. PTZ and focus methods are
    no-ops that only log, allowing the rest of the system to run without
    real hardware.
    """

    def __init__(
        self,
        camera_id: str,
        image_url: str = DEFAULT_FAKE_IMAGE_URL,
        cam_type: str = "static",
        cam_poses: Optional[list[int]] = None,
        cam_azimuths: Optional[list[int]] = None,
        focus_position: Optional[int] = None,
    ) -> None:
        super().__init__(camera_id=camera_id, cam_type=cam_type)
        self.image_url = image_url
        self._cached_images: list[Image.Image] = []
        self._index = 0
        self._lock = Lock()

        self.cam_poses = cam_poses or []
        self.cam_azimuths = cam_azimuths or []
        self.focus_position = focus_position

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_zip_images(self, content: bytes) -> list[Image.Image]:
        images: list[tuple[str, Image.Image]] = []
        with zipfile.ZipFile(BytesIO(content)) as zf:
            for info in zf.infolist():
                if info.is_dir():
                    continue
                name = info.filename
                # Only pick files inside an "images/" folder, at any depth
                parts = name.split("/")
                if "images" not in parts[:-1]:
                    continue
                if not name.lower().endswith(_IMAGE_EXTENSIONS):
                    continue
                with zf.open(info) as fh:
                    img = Image.open(BytesIO(fh.read())).convert("RGB")
                images.append((parts[-1], img))
        images.sort(key=itemgetter(0))
        return [img for _, img in images]

    def _ensure_images(self) -> None:
        if self._cached_images:
            return

        try:
            logger.info("MockCamera %s downloading from %s", self.camera_id, self.image_url)
            resp = requests.get(self.image_url, timeout=15)
            resp.raise_for_status()
            content = resp.content

            is_zip = self.image_url.lower().endswith(".zip") or content[:2] == b"PK"
            if is_zip:
                self._cached_images = self._load_zip_images(content)
                if not self._cached_images:
                    logger.error("MockCamera %s zip contains no images/ entries", self.camera_id)
                    return
                logger.info(
                    "MockCamera %s cached %d images from zip, first size=%s",
                    self.camera_id,
                    len(self._cached_images),
                    self._cached_images[0].size,
                )
            else:
                img = Image.open(BytesIO(content)).convert("RGB")
                self._cached_images = [img]
                logger.info("MockCamera %s cached single image, size=%s", self.camera_id, img.size)
        except Exception as exc:
            logger.error("MockCamera %s failed to download image(s): %s", self.camera_id, exc)
            self._cached_images = []

    # ------------------------------------------------------------------
    # BaseCamera implementation
    # ------------------------------------------------------------------

    def capture(self, **kwargs) -> Optional[Image.Image]:
        _ = kwargs  # unused
        self._ensure_images()
        if not self._cached_images:
            return None
        with self._lock:
            img = self._cached_images[self._index % len(self._cached_images)]
            self._index = (self._index + 1) % len(self._cached_images)
        return img.copy()

    # ------------------------------------------------------------------
    # PTZMixin implementation (no op)
    # ------------------------------------------------------------------

    def move_camera(self, operation: str, speed: int = 20, idx: int = 0) -> None:
        logger.info(
            "MockCamera %s move_camera called, op=%s speed=%s idx=%s (no op)",
            self.camera_id,
            operation,
            speed,
            idx,
        )

    # ------------------------------------------------------------------
    # FocusMixin implementation (fake but compatible)
    # ------------------------------------------------------------------

    def set_manual_focus(self, position: int) -> None:
        self.focus_position = position
        logger.info("MockCamera %s set_manual_focus(%s) (no op)", self.camera_id, position)

    def focus_finder(self, save_images: bool = False, retry_depth: int = 0) -> int:
        _ = save_images
        _ = retry_depth
        if self.focus_position is None:
            self.focus_position = 720
        logger.info("MockCamera %s focus_finder -> %s (fake)", self.camera_id, self.focus_position)
        return int(self.focus_position)

    # ------------------------------------------------------------------
    # Extra helpers to satisfy existing routes (no op)
    # ------------------------------------------------------------------

    def set_auto_focus(self, disable: bool):
        logger.info("MockCamera %s set_auto_focus(disable=%s) (no op)", self.camera_id, disable)
        return {"status": "ok", "disable": disable}

    def get_focus_level(self):
        focus = self.focus_position if self.focus_position is not None else 720
        zoom = 0
        logger.info("MockCamera %s get_focus_level -> focus=%s zoom=%s (fake)", self.camera_id, focus, zoom)
        return {"focus": focus, "zoom": zoom}

    def start_zoom_focus(self, position: int):
        logger.info("MockCamera %s start_zoom_focus(%s) (no op)", self.camera_id, position)
        return {"status": "ok", "position": position}
