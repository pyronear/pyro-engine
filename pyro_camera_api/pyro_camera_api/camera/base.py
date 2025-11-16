# pyro_camera_api/camera/base.py

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Optional

from PIL import Image


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
        self.last_images: Dict[int, Image.Image] = {}

    @abstractmethod
    def capture(self, **kwargs) -> Optional[Image.Image]:
        """
        Capture a frame and return it as a PIL Image or None on failure.

        Keyword arguments are backend specific:
        Reolink may accept pos_id,
        RTSP may accept timeout,
        URL snapshot usually no arguments.
        """
        ...


class PTZMixin(ABC):
    """
    Capability mixin for cameras that support pan tilt zoom controls.

    Use isinstance(camera, PTZMixin) to check support.
    """

    @abstractmethod
    def move_camera(self, operation: str, speed: int = 20, idx: int = 0) -> None:
        """
        Perform a PTZ operation.

        Args:
            operation: Operation name understood by the backend,
                       examples "Left", "Right", "Up", "Down", "Stop", "ToPos".
            speed: Backend specific speed value.
            idx: Preset index for operations that use a preset.
        """
        ...


class FocusMixin(ABC):
    """
    Capability mixin for cameras that support manual focus control.

    Use isinstance(camera, FocusMixin) to check support.
    """

    @abstractmethod
    def set_manual_focus(self, position: int) -> None:
        """
        Set manual focus to a specific position.

        Args:
            position: Backend specific focus position value.
        """
        ...

    @abstractmethod
    def focus_finder(self, save_images: bool = False, retry_depth: int = 0) -> int:
        """
        Run a focus search algorithm and return best focus position.

        Args:
            save_images: If true, implementation may save intermediate images.
            retry_depth: Optional parameter for implementations that want retries.

        Returns:
            Best focus position found as an integer.
        """
        ...
