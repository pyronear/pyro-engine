# Copyright (C) 2022-2026, Pyronear.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.


from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Optional

from PIL import Image


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
        # Dictionary for storing latest images
        # PTZ cameras can use pose -> image
        # Static cameras can use -1 -> image
        self.last_images: Dict[int, Optional[Image.Image]] = {}

    @abstractmethod
    def capture(self, **kwargs) -> Optional[Image.Image]:
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

    # Reference focus position found by calibration, None until known
    focus_position: Optional[int] = None

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

    def focus_finder(
        self,
        save_images: bool = False,
        retry_depth: int = 0,
        should_abort: Optional[Callable[[], bool]] = None,
    ) -> int:
        """
        Run the adapter's autofocus search and return the best focus position.

        should_abort, when provided, is polled between capture steps; when it
        fires the adapter must restore its pre-search focus and raise
        FocusAbortedError so the caller does not store a reference.
        Adapters without a real search keep this default, which raises
        NotImplementedError and excludes them via supports_focus_search().
        """
        raise NotImplementedError
