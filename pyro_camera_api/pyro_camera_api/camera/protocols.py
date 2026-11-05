# Copyright (C) 2022-2026, Pyronear.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

"""Types for optional adapter capabilities checked by the camera routes."""

from typing import Callable, Protocol


class FocusControl(Protocol):
    focus_position: int | None

    def get_focus_level(self) -> dict | None: ...

    def set_manual_focus(self, position: int) -> object: ...

    def set_auto_focus(self, disable: bool) -> object: ...

    def start_zoom_focus(self, position: int) -> object: ...

    def focus_finder(
        self,
        save_images: bool = False,
        retry_depth: int = 0,
        should_abort: Callable[[], bool] | None = None,
    ) -> int: ...


class RelativePTZ(Protocol):
    def move_relative_deg(self, delta_azimuth_deg: float, delta_elevation_deg: float = 0.0) -> dict: ...

    def get_ptz_status(self) -> dict: ...


class PresetControl(Protocol):
    def get_ptz_preset(self) -> object: ...

    def set_ptz_preset(self, idx: int | None = None) -> object: ...


class Rebootable(Protocol):
    def reboot_camera(self) -> bool: ...
