# Copyright (C) 2022-2026, Pyronear.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.


from __future__ import annotations

import itertools
import os
import threading
import time
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from io import BytesIO
from typing import Deque, Dict, List, Optional

from PIL import Image

# Captures kept per pose, as compressed JPEG. A decoded 4K frame takes about 33 MB, its JPEG about 1 MB.
FRAME_BUFFER_SIZE = max(1, int(os.getenv("FRAME_BUFFER_SIZE", "16")))
# Frame ids stay unique and ordered across restarts: start time of this process, then a counter.
_BOOT = int(time.time())
_SEQ = itertools.count(1)


@dataclass(frozen=True)
class Frame:
    """One stored capture: the JPEG as the camera sent it when it sent one, else encoded once."""

    id: str
    jpeg: bytes
    stored_at: float  # time.monotonic(), so stale frames can be refused
    captured_at: str  # UTC, ISO 8601

    def decode(self) -> Image.Image:
        image = Image.open(BytesIO(self.jpeg))
        image.load()
        return image if image.mode == "RGB" else image.convert("RGB")


def open_snapshot(raw: bytes) -> Image.Image:
    """Open the bytes a camera answered with, as an RGB image.

    An RGB JPEG is checked with a reduced-size decode and returned without decoding it at full size
    (Pillow decodes on first pixel access). The image carries its bytes as ``original_jpeg`` so
    ``store_image`` keeps them as they are. A plain attribute, so a resized or anonymized copy does not inherit it.
    """
    image = Image.open(BytesIO(raw))
    if image.format != "JPEG" or image.mode != "RGB":
        return image.convert("RGB")
    check = Image.open(BytesIO(raw))
    check.draft("RGB", (check.width // 8, check.height // 8))
    check.load()  # raises on a truncated or corrupt file, as the full decode did
    image.original_jpeg = raw  # ty: ignore[unresolved-attribute]
    return image


class FocusAbortedError(Exception):
    """Raised inside focus_finder when the caller requests an early abort."""


class BaseCamera(ABC):
    """
    Abstract base class for all camera types.

    Every concrete camera must implement capture.
    """

    def __init__(self, camera_id: str, cam_type: str = "static") -> None:
        """
        Args:
            camera_id: Logical identifier used in the registry and API.
            cam_type: Simple label for the kind of camera,
                      examples "static", "ptz", "rtsp".
        """
        self.camera_id = camera_id
        self.cam_type = cam_type
        # Last captures of each pose, oldest first. PTZ cameras use their pose ids, static cameras use -1.
        self._frames: Dict[int, Deque[Frame]] = {}
        self._frames_lock = threading.Lock()

    def store_image(self, pose: int, image: Image.Image) -> Frame:
        jpeg = getattr(image, "original_jpeg", None)
        if jpeg is None:
            buffer = BytesIO()
            image.save(buffer, format="JPEG", quality=95)
            jpeg = buffer.getvalue()
        frame = Frame(
            id=f"{_BOOT}-{next(_SEQ)}",
            jpeg=jpeg,
            stored_at=time.monotonic(),
            captured_at=datetime.now(timezone.utc).isoformat(),
        )
        with self._frames_lock:
            self._frames.setdefault(pose, deque(maxlen=FRAME_BUFFER_SIZE)).append(frame)
        return frame

    def frames(self, pose: int) -> List[Frame]:
        """Stored captures of a pose, oldest first."""
        with self._frames_lock:
            return list(self._frames.get(pose, ()))

    def latest_frames(self) -> Dict[int, Frame]:
        """Last capture of every pose that has one."""
        with self._frames_lock:
            return {pose: frames[-1] for pose, frames in self._frames.items() if frames}

    def clear_frames(self) -> None:
        with self._frames_lock:
            self._frames.clear()

    @abstractmethod
    def capture(self, *, patrol_id: Optional[int] = None) -> Optional[Image.Image]:
        """
        Capture a frame and return it as a PIL Image or None on failure.

        Keyword arguments are adapter specific:
        Reolink may accept patrol_id,
        RTSP may accept timeout,
        URL snapshot usually no arguments.
        """
        ...


# Continuous operations that involve the pan axis. Starting one of these makes
# a dead-reckoned azimuth stale until the caller computes the displacement or a
# preset move provides a fresh reference.
PAN_OPERATIONS = frozenset({"Left", "Right", "UpLeft", "UpRight", "DownLeft", "DownRight"})


class PTZMixin(ABC):
    """
    Capability mixin for cameras that support pan tilt zoom controls.

    Use isinstance(camera, PTZMixin) to check support.
    """

    # "tracked": azimuth is dead-reckoned server-side from commanded moves.
    # "hardware": azimuth is read back from the camera itself.
    azimuth_source: str = "tracked"

    # Local pose presets and their real-world azimuths, index-aligned. The
    # azimuths come from credentials.json (legacy) or are fetched from the
    # platform API at startup (see camera.pose_azimuths).
    current_azimuth: Optional[float]
    cam_poses: List[int]
    cam_azimuths: List[float]

    # Seconds the API keeps the camera locked after a fire-and-forget preset
    # move so concurrent commands get rejected while the camera travels.
    # 0 when the adapter blocks (or completes instantly) on preset moves.
    preset_move_hold_s: float = 0.0

    @abstractmethod
    def move_camera(self, operation: str, speed: int = 20, idx: int = 0) -> None:
        """
        Perform a PTZ operation.

        Args:
            operation: Operation name understood by the adapter,
                       examples "Left", "Right", "Up", "Down", "Stop", "ToPos".
            speed: adapter specific speed value.
            idx: Preset index for operations that use a preset.
        """
        ...

    @abstractmethod
    def get_azimuth(self) -> Optional[float]:
        """
        Return the camera's current real-world azimuth in degrees [0, 360),
        or None when unknown (e.g. after boot, before any preset move).
        """
        ...


class FocusMixin(ABC):
    """
    Capability mixin for cameras that support manual focus control.

    Use isinstance(camera, FocusMixin) to check support.
    """

    @abstractmethod
    def set_manual_focus(self, position: int) -> None:
        """Set manual focus to a specific position."""
        ...

    @abstractmethod
    def get_focus_level(self) -> Optional[dict]:
        """
        Retrieve current focus and zoom information.

        Expected shape for dict (Reolink)
          { "focus": int | None, "zoom": int | None }
        """
        ...
