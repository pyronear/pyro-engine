# Copyright (C) 2022-2026, Pyronear.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.


from __future__ import annotations

import logging
import operator
import pathlib
import time
from io import BytesIO
from typing import Any, List, Optional

import cv2
import numpy as np
import requests
import urllib3
from PIL import Image

from pyro_camera_api.camera.base import PAN_OPERATIONS, BaseCamera, FocusMixin, PTZMixin

__all__ = ["ReolinkCamera"]

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

logger = logging.getLogger(__name__)


class ReolinkCamera(BaseCamera, PTZMixin, FocusMixin):
    """
    A controller class for interacting with Reolink cameras.
    """

    # ToPos is fire-and-forget with no completion feedback: hold the per-camera
    # lock for a conservative travel time so concurrent commands get a 409.
    preset_move_hold_s = 5.0

    def __init__(
        self,
        camera_id: str,
        ip_address: str,
        username: str,
        password: str,
        cam_type: str = "ptz",
        cam_poses: Optional[List[int]] = None,
        cam_azimuths: Optional[List[float]] = None,
        protocol: str = "https",
        focus_position: Optional[int] = None,
    ):
        # BaseCamera stores camera_id, cam_type and last_images
        super().__init__(camera_id=camera_id, cam_type=cam_type)
        self.ip_address = ip_address
        self.username = username
        self.password = password
        self.cam_poses = cam_poses if cam_poses is not None else []
        self.cam_azimuths = cam_azimuths if cam_azimuths is not None else []
        self.protocol = protocol
        self.focus_position = focus_position
        # Dead-reckoned real-world azimuth; None until a preset move gives a reference.
        self.current_azimuth: Optional[float] = None
        # An empty azimuth list is normal at boot: the mapping is fetched from
        # the platform API by camera.pose_azimuths. A non-empty mismatched list
        # is a config error that silently disables tracking, hence the warning.
        if self.cam_type == "ptz" and self.cam_azimuths and len(self.cam_poses) != len(self.cam_azimuths):
            logger.warning(
                "[%s] poses (%d) and azimuths (%d) differ in credentials.json; "
                "azimuth tracking will stay unknown until fixed",
                self.ip_address,
                len(self.cam_poses),
                len(self.cam_azimuths),
            )
        self._has_motorised_lens: Optional[bool] = None

    def has_motorised_lens(self) -> bool:
        """Whether this camera's lens can be driven.

        ``cam_type`` describes how the camera is mounted, not what optics it
        carries. A "static" camera is one that does not pan or tilt, which says
        nothing about zoom: Reolink bullets such as the RLC-811A or the P430 sit
        fixed on their mast and still ship a motorised varifocal lens.

        Rather than infer it, ask the camera. ``GetZoomFocus`` reports a zoom
        position only on models that can move the lens, so the answer comes from
        the device itself and holds for any model. The result is cached: it
        cannot change while the camera is running, and this saves a request on
        every zoom or focus command.
        """
        if self._has_motorised_lens is None:
            try:
                self._has_motorised_lens = (self.get_focus_level() or {}).get("zoom") is not None
            except Exception as exc:
                logger.warning("[%s] could not probe lens capability: %s", self.ip_address, exc)
                return False
        return self._has_motorised_lens

    def _build_url(self, command: str) -> str:
        """Constructs a URL for API commands to the camera."""
        return (
            f"{self.protocol}://{self.ip_address}/cgi-bin/api.cgi?"
            f"cmd={command}&user={self.username}&password={self.password}&channel=0"
        )

    def _handle_response(self, response, success_message: str):
        """Handles HTTP responses, logging success or errors based on response data."""
        if response.status_code == 200:
            response_data = response.json()
            if response_data[0]["code"] == 0:
                logger.debug(success_message)
            else:
                logger.error("Error: %s", response_data)
            return response_data
        logger.error("Failed operation: %s, %s", response.status_code, response.text)
        return None

    def capture(self, patrol_id: Optional[int] = None, timeout: int = 2) -> Optional[Image.Image]:
        """
        Captures an image from the camera. Optionally moves the camera to a preset position before capturing.
        """
        if patrol_id is not None:
            self.move_camera("ToPos", idx=int(patrol_id), speed=50)
            time.sleep(1)
        url = self._build_url("Snap")
        logger.debug("Start capture for %s", self.ip_address)

        try:
            response = requests.get(url, verify=False, timeout=timeout)  # nosec: B501
            if response.status_code == 200:
                image_data = BytesIO(response.content)
                return Image.open(image_data).convert("RGB")
            logger.error("Failed to capture image: %s, %s", response.status_code, response.text)
        except requests.RequestException as e:
            logger.error("Request failed: %s", e)
        return None

    def move_camera(self, operation: str, speed: int = 20, idx: int = 0):
        """
        Sends a command to move the camera.

        Raises RuntimeError when the camera rejects the command, so callers
        (and the dead-reckoned azimuth) never assume a move that did not start.
        """
        if operation in PAN_OPERATIONS:
            # Untimed pan motion starts: azimuth is unknown until the caller
            # computes the displacement or a preset move resyncs it.
            self.current_azimuth = None
        url = self._build_url("PtzCtrl")
        data: Any = [
            {"cmd": "PtzCtrl", "action": 0, "param": {"channel": 0, "op": operation, "id": idx, "speed": speed}}
        ]
        response = requests.post(url, json=data, verify=False)  # nosec: B501
        response_data = self._handle_response(response, "PTZ operation successful.")
        try:
            ok = bool(response_data) and response_data[0]["code"] == 0
        except (KeyError, IndexError, TypeError):
            ok = False
        if not ok:
            raise RuntimeError(f"PTZ command '{operation}' rejected by camera {self.ip_address}")
        if operation == "ToPos":
            self._sync_azimuth_from_pose(int(idx))

    def _sync_azimuth_from_pose(self, pose_id: int) -> None:
        """Resync the dead-reckoned azimuth from the pose mapping after an accepted ToPos.

        An accepted preset outside the configured mapping still moves the
        camera, so the previous reference becomes stale and is dropped.
        """
        if pose_id in self.cam_poses and len(self.cam_poses) == len(self.cam_azimuths):
            self.current_azimuth = float(self.cam_azimuths[self.cam_poses.index(pose_id)]) % 360.0
        else:
            self.current_azimuth = None

    def get_azimuth(self) -> Optional[float]:
        """Return the dead-reckoned azimuth, or None when unknown."""
        return self.current_azimuth

    def move_in_seconds(self, s: float, operation: str = "Right", speed: int = 20, save_path: str = "im.jpg"):
        """
        Moves the camera in a specified direction for a specified number of seconds.
        """
        self.move_camera(operation, speed)
        time.sleep(s)
        self.move_camera("Stop")
        time.sleep(1)
        im = self.capture()
        if im is not None and save_path is not None:
            im.save(save_path)

    def get_ptz_preset(self):
        """
        Retrieves the preset positions available for PTZ cameras.
        """
        url = self._build_url("GetPtzPreset")
        data: Any = [{"cmd": "GetPtzPreset", "action": 1, "param": {"channel": 0}}]
        response = requests.post(url, json=data, verify=False)  # nosec: B501
        response_data = self._handle_response(response, "Presets retrieved successfully.")
        if response_data and response_data[0]["code"] == 0:
            return response_data[0].get("value", {}).get("PtzPreset", [])
        return None

    def set_ptz_preset(self, idx: Optional[int] = None):
        """
        Sets a PTZ preset position. If no ID is provided, finds the next available slot.
        """
        if idx is None:
            presets_ptz = self.get_ptz_preset()
            for cfg in presets_ptz:
                if cfg["enable"] == 0:
                    idx = cfg["id"]
                    break
            if idx is None:
                raise ValueError("No available slots for new presets.")

        url = self._build_url("SetPtzPreset")
        name = f"pos{idx}"
        data: Any = [
            {
                "cmd": "SetPtzPreset",
                "action": 0,
                "param": {"PtzPreset": {"channel": 0, "enable": 1, "id": idx, "name": name}},
            }
        ]
        response = requests.post(url, json=data, verify=False)  # nosec: B501
        self._handle_response(response, f"Preset {name} set successfully.")

    def reboot_camera(self) -> bool:
        url = self._build_url("Reboot")
        data = [{"cmd": "Reboot"}]
        response = requests.post(url, json=data, verify=False)  # nosec: B501
        response_data = self._handle_response(response, "Camera reboot initiated successfully.")
        if not response_data:
            return False
        try:
            return response_data[0]["code"] == 0
        except (KeyError, IndexError, TypeError):
            return False

    def get_auto_focus(self):
        url = self._build_url("GetAutoFocus")
        data: Any = [{"cmd": "GetAutoFocus", "action": 1, "param": {"channel": 0}}]
        response = requests.post(url, json=data, verify=False)  # nosec: B501
        return self._handle_response(response, "Fetched AutoFocus settings successfully.")

    def set_auto_focus(self, disable: bool):
        url = self._build_url("SetAutoFocus")
        data: Any = [
            {
                "cmd": "SetAutoFocus",
                "action": 0,
                "param": {"AutoFocus": {"channel": 0, "disable": int(disable)}},
            }
        ]
        response = requests.post(url, json=data, verify=False)  # nosec: B501
        return self._handle_response(response, "Set AutoFocus settings successfully.")

    def start_zoom_focus(self, position: int):
        if self.has_motorised_lens():
            url = self._build_url("StartZoomFocus")
            data: Any = [
                {
                    "cmd": "StartZoomFocus",
                    "action": 0,
                    "param": {"ZoomFocus": {"channel": 0, "pos": position, "op": "ZoomPos"}},
                }
            ]
            response = requests.post(url, json=data, verify=False)  # nosec: B501
            return self._handle_response(response, "Started ZoomFocus successfully.")
        return None

    def set_manual_focus(self, position: int):
        """
        Set manual focus to a specific position.
        """
        if self.has_motorised_lens():
            self.focus_position = position
            url = self._build_url("StartZoomFocus")
            data: Any = [
                {
                    "cmd": "StartZoomFocus",
                    "action": 0,
                    "param": {"ZoomFocus": {"channel": 0, "pos": position, "op": "FocusPos"}},
                }
            ]
            response = requests.post(url, json=data, verify=False)  # nosec: B501
            return self._handle_response(response, f"Manual focus set at position {position}")
        return None

    def get_focus_level(self):
        """Retrieve the current manual focus and zoom positions."""
        url = self._build_url("GetZoomFocus")
        data: Any = [{"cmd": "GetZoomFocus", "action": 0, "param": {"channel": 0}}]
        response = requests.post(url, json=data, verify=False)  # nosec: B501
        result = self._handle_response(response, "Got zoom/focus values")
        if result and result[0]["code"] == 0:
            zoom_focus = result[0]["value"]["ZoomFocus"]
            focus = zoom_focus.get("focus", {}).get("pos")
            zoom = zoom_focus.get("zoom", {}).get("pos")
            return {"focus": focus, "zoom": zoom}
        return None

    def _measure_sharpness(self, pil_image: Image.Image) -> float:
        img = pil_image.convert("L")
        arr = np.array(img)
        laplacian = cv2.Laplacian(arr, cv2.CV_64F)
        return float(laplacian.var())

    def focus_finder(self, save_images: bool = False, retry_depth: int = 0) -> int:
        """
        Perform adaptive exponential hill climb to find best manual focus.
        """
        _ = retry_depth  # unused, kept for signature compatibility

        abs_min = 600
        abs_max = 900

        def clamp_focus(pos: int) -> int:
            return max(abs_min, min(abs_max, pos))

        def capture_and_score(pos: int) -> float:
            pos = clamp_focus(pos)
            self.set_manual_focus(pos)
            time.sleep(2)
            image = self.capture()
            if image is None:
                logger.warning("[%s] No image at focus %s", self.ip_address, pos)
                return 0.0
            score_local = self._measure_sharpness(image)
            logger.info("[%s] Focus %s: Sharpness = %.2f", self.ip_address, pos, score_local)
            if save_images:
                folder = f"focus_debug/{self.ip_address.replace('.', '_')}"
                pathlib.Path(folder).mkdir(exist_ok=True, parents=True)
                image.save(f"{folder}/focus_{pos}.jpg")
            return score_local

        if not self.has_motorised_lens():
            return 720

        if self.focus_position is None:
            self.start_zoom_focus(0)
            time.sleep(0.5)
            focus_info = self.get_focus_level() or {}
            current_focus = focus_info.get("focus", 720)
            logger.info("[%s] Initial focus obtained from camera: %s", self.ip_address, current_focus)
        else:
            current_focus = self.focus_position
            logger.info("[%s] Using existing focus position: %s", self.ip_address, current_focus)

        best_focus = clamp_focus(int(current_focus))
        best_score = capture_and_score(best_focus)

        forward_score = capture_and_score(best_focus + 1)
        backward_score = capture_and_score(best_focus - 1)

        if forward_score > backward_score:
            direction = 1
            next_focus = best_focus + 1
            next_score = forward_score
        else:
            direction = -1
            next_focus = best_focus - 1
            next_score = backward_score

        step = 2
        history = [(best_focus, best_score), (next_focus, next_score)]

        while True:
            test_focus = clamp_focus(next_focus + direction * step)
            score = capture_and_score(test_focus)
            history.append((test_focus, score))
            if score > next_score:
                next_focus = test_focus
                next_score = score
                step *= 2
            else:
                break

        best_focus, best_score = max(history, key=operator.itemgetter(1))
        for fine_step in [3, 1]:
            improved = True
            while improved:
                improved = False
                for offset in (-fine_step, fine_step):
                    candidate = clamp_focus(best_focus + offset)
                    score = capture_and_score(candidate)
                    if score > best_score:
                        best_score = score
                        best_focus = candidate
                        improved = True
                        break

        self.focus_position = best_focus
        self.set_manual_focus(best_focus)
        logger.info(
            "[%s] Final best focus at %s with sharpness %.2f",
            self.ip_address,
            best_focus,
            best_score,
        )
        return best_focus
