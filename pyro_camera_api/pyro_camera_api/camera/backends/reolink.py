# Copyright (C) 2022-2025, Pyronear.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.


from __future__ import annotations

import logging
import os
import time
from io import BytesIO
from typing import List, Optional

import cv2
import numpy as np
import requests
import urllib3
from PIL import Image

from pyro_camera_api.camera.base import BaseCamera, FocusMixin, PTZMixin

__all__ = ["ReolinkCamera"]

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

logger = logging.getLogger(__name__)


class ReolinkCamera(BaseCamera, PTZMixin, FocusMixin):
    """
    A controller class for interacting with Reolink cameras.
    """

    def __init__(
        self,
        camera_id: str,
        ip_address: str,
        username: str,
        password: str,
        cam_type: str = "ptz",
        cam_poses: Optional[List[int]] = None,
        cam_azimuths: Optional[List[int]] = None,
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

    def capture(self, pos_id: Optional[int] = None, timeout: int = 2) -> Optional[Image.Image]:
        """
        Captures an image from the camera. Optionally moves the camera to a preset position before capturing.
        """
        if pos_id is not None:
            self.move_camera("ToPos", idx=int(pos_id), speed=50)
            time.sleep(1)
        url = self._build_url("Snap")
        logger.debug("Start capture for %s", self.ip_address)

        try:
            response = requests.get(url, verify=False, timeout=timeout)  # nosec: B501
            if response.status_code == 200:
                image_data = BytesIO(response.content)
                image = Image.open(image_data).convert("RGB")
                return image
            logger.error("Failed to capture image: %s, %s", response.status_code, response.text)
        except requests.RequestException as e:
            logger.error("Request failed: %s", e)
        return None

    def move_camera(self, operation: str, speed: int = 20, idx: int = 0):
        """
        Sends a command to move the camera.
        """
        url = self._build_url("PtzCtrl")
        data = [{"cmd": "PtzCtrl", "action": 0, "param": {"channel": 0, "op": operation, "id": idx, "speed": speed}}]
        response = requests.post(url, json=data, verify=False)  # nosec: B501
        self._handle_response(response, "PTZ operation successful.")

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
        data = [{"cmd": "GetPtzPreset", "action": 1, "param": {"channel": 0}}]
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
        data = [
            {
                "cmd": "SetPtzPreset",
                "action": 0,
                "param": {"PtzPreset": {"channel": 0, "enable": 1, "id": idx, "name": name}},
            }
        ]
        response = requests.post(url, json=data, verify=False)  # nosec: B501
        self._handle_response(response, f"Preset {name} set successfully.")

    def reboot_camera(self):
        url = self._build_url("Reboot")
        data = [{"cmd": "Reboot"}]
        response = requests.post(url, json=data, verify=False)  # nosec: B501
        return self._handle_response(response, "Camera reboot initiated successfully.")

    def get_auto_focus(self):
        url = self._build_url("GetAutoFocus")
        data = [{"cmd": "GetAutoFocus", "action": 1, "param": {"channel": 0}}]
        response = requests.post(url, json=data, verify=False)  # nosec: B501
        return self._handle_response(response, "Fetched AutoFocus settings successfully.")

    def set_auto_focus(self, disable: bool):
        url = self._build_url("SetAutoFocus")
        data = [
            {
                "cmd": "SetAutoFocus",
                "action": 0,
                "param": {"AutoFocus": {"channel": 0, "disable": int(disable)}},
            }
        ]
        response = requests.post(url, json=data, verify=False)  # nosec: B501
        return self._handle_response(response, "Set AutoFocus settings successfully.")

    def start_zoom_focus(self, position: int):
        if self.cam_type != "static":
            url = self._build_url("StartZoomFocus")
            data = [
                {
                    "cmd": "StartZoomFocus",
                    "action": 0,
                    "param": {"ZoomFocus": {"channel": 0, "pos": position, "op": "ZoomPos"}},
                }
            ]
            response = requests.post(url, json=data, verify=False)  # nosec: B501
            return self._handle_response(response, "Started ZoomFocus successfully.")

    def set_manual_focus(self, position: int):
        """
        Set manual focus to a specific position.
        """
        if self.cam_type != "static":
            self.focus_position = position
            url = self._build_url("StartZoomFocus")
            data = [
                {
                    "cmd": "StartZoomFocus",
                    "action": 0,
                    "param": {"ZoomFocus": {"channel": 0, "pos": position, "op": "FocusPos"}},
                }
            ]
            response = requests.post(url, json=data, verify=False)  # nosec: B501
            return self._handle_response(response, f"Manual focus set at position {position}")

    def get_focus_level(self):
        """Retrieve the current manual focus and zoom positions."""
        url = self._build_url("GetZoomFocus")
        data = [{"cmd": "GetZoomFocus", "action": 0, "param": {"channel": 0}}]
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

        ABS_MIN = 600
        ABS_MAX = 900

        def clamp_focus(pos: int) -> int:
            return max(ABS_MIN, min(ABS_MAX, pos))

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
                os.makedirs(folder, exist_ok=True)
                image.save(f"{folder}/focus_{pos}.jpg")
            return score_local

        if self.cam_type == "static":
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

        best_focus, best_score = max(history, key=lambda x: x[1])
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
