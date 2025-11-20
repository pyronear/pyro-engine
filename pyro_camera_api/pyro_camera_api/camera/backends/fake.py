# Copyright (C) 2022-2025, Pyronear.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.


from __future__ import annotations

import logging
from io import BytesIO
from typing import Optional

import requests
from PIL import Image

from pyro_camera_api.camera.base import BaseCamera, FocusMixin, PTZMixin

__all__ = ["FakeCamera"]

logger = logging.getLogger(__name__)

DEFAULT_FAKE_IMAGE_URL = "https://github.com/pyronear/pyro-engine/releases/download/v0.1.1/fire_sample_image.jpg"


class FakeCamera(BaseCamera, PTZMixin, FocusMixin):
    """
    Simple fake camera backend.

    It downloads one reference image once,
    then returns that image for every capture call.
    Control and focus methods are no ops that only log.
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
        self._cached_image: Optional[Image.Image] = None

        # PTZ and focus metadata used by patrol and API routes
        self.cam_poses: list[int] = cam_poses or []
        self.cam_azimuths: list[int] = cam_azimuths or []
        self.focus_position: Optional[int] = focus_position

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_image(self) -> None:
        """Download the reference image once and cache it."""
        if self._cached_image is not None:
            return

        try:
            logger.info("FakeCamera %s downloading image from %s", self.camera_id, self.image_url)
            resp = requests.get(self.image_url, timeout=5)
            resp.raise_for_status()
            self._cached_image = Image.open(BytesIO(resp.content)).convert("RGB")
            logger.info("FakeCamera %s cached image, size=%s", self.camera_id, self._cached_image.size)
        except Exception as exc:
            logger.error("FakeCamera %s failed to download image: %s", self.camera_id, exc)
            self._cached_image = None

    # ------------------------------------------------------------------
    # BaseCamera implementation
    # ------------------------------------------------------------------

    def capture(self, **kwargs) -> Optional[Image.Image]:
        """
        Always returns the same cached image.

        Extra keyword arguments are ignored.
        """
        _ = kwargs  # unused
        self._ensure_image()
        if self._cached_image is None:
            return None
        # Return a copy to avoid callers mutating the cached instance
        return self._cached_image.copy()

    # ------------------------------------------------------------------
    # PTZMixin implementation (no op)
    # ------------------------------------------------------------------

    def move_camera(self, operation: str, speed: int = 20, idx: int = 0) -> None:
        """
        No op PTZ call, only logs the intent.
        """
        logger.info(
            "FakeCamera %s move_camera called, op=%s speed=%s idx=%s (no op)",
            self.camera_id,
            operation,
            speed,
            idx,
        )

    # ------------------------------------------------------------------
    # FocusMixin implementation (fake but compatible)
    # ------------------------------------------------------------------

    def set_manual_focus(self, position: int) -> None:
        """
        Remember the requested focus position, no real hardware call.
        """
        self.focus_position = position
        logger.info("FakeCamera %s set_manual_focus(%s) (no op)", self.camera_id, position)

    def focus_finder(self, save_images: bool = False, retry_depth: int = 0) -> int:
        """
        Fake focus search returns a fixed focus position.
        """
        _ = save_images
        _ = retry_depth
        if self.focus_position is None:
            self.focus_position = 720
        logger.info("FakeCamera %s focus_finder -> %s (fake)", self.camera_id, self.focus_position)
        return int(self.focus_position)

    # ------------------------------------------------------------------
    # Extra helpers to satisfy existing routes (no op)
    # ------------------------------------------------------------------

    def set_auto_focus(self, disable: bool):
        logger.info("FakeCamera %s set_auto_focus(disable=%s) (no op)", self.camera_id, disable)
        return {"status": "ok", "disable": disable}

    def get_focus_level(self):
        focus = self.focus_position if self.focus_position is not None else 720
        zoom = 0
        logger.info("FakeCamera %s get_focus_level -> focus=%s zoom=%s (fake)", self.camera_id, focus, zoom)
        return {"focus": focus, "zoom": zoom}

    def start_zoom_focus(self, position: int):
        logger.info("FakeCamera %s start_zoom_focus(%s) (no op)", self.camera_id, position)
        return {"status": "ok", "position": position}
