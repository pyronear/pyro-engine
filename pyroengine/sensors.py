# Copyright (C) 2022-2024, Pyronear.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

import logging
import time
from io import BytesIO
from typing import List, Optional

import requests
import urllib3
from PIL import Image

__all__ = ["ReolinkCamera"]

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Configure logging
logging.basicConfig(level=logging.INFO)


class ReolinkCamera:
    """
    A controller class for interacting with Reolink cameras.

    Attributes:
        ip_address (str): IP address of the Reolink camera.
        username (str): Username for accessing the camera.
        password (str): Password for accessing the camera.
        cam_type (str): Type of the camera, e.g., 'static' or 'ptz' (pan-tilt-zoom).
        cam_poses (Optional[List[int]]): List of preset positions for PTZ cameras.
        protocol (str): Protocol used for communication, defaults to 'http'.
        verbose (bool): Flag to enable detailed logging.

    Methods:
        capture(pos_id): Captures an image from the camera. Moves to position `pos_id` if provided.
        move_camera(operation, speed, id): Moves the camera based on the operation type and speed.
        move_in_seconds(s, operation, speed): Moves the camera for a specific duration and then stops.
        get_ptz_preset(): Retrieves preset positions for a PTZ camera.
        set_ptz_preset(id): Sets a PTZ preset position using an ID.
    """

    def __init__(
        self,
        ip_address: str,
        username: str,
        password: str,
        cam_type: str,
        cam_poses: Optional[List[int]] = None,
        protocol: str = "http",
        verbose: str = False,
    ):
        self.ip_address = ip_address
        self.username = username
        self.password = password
        self.cam_type = cam_type
        self.cam_poses = cam_poses if cam_poses is not None else []
        self.protocol = protocol
        self.verbose = verbose

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
                if self.verbose:
                    logging.info(success_message)
            else:
                logging.error(f"Error: {response_data}")
            return response_data
        else:
            logging.error(f"Failed operation: {response.status_code}, {response.text}")

    def capture(self, pos_id: Optional[int] = None, timeout: int = 3, retries: int = 3) -> Optional[Image.Image]:
        """
        Captures an image from the camera. Optionally moves the camera to a preset position before capturing.

        Args:
            pos_id (Optional[int]): The preset position ID to move to before capturing.
            timeout (int): Timeout for the HTTP request.
            retries (int): Number of retries for the HTTP request.

        Returns:
            Image.Image: An image captured from the camera, or None if there was an error.
        """
        if pos_id is not None:
            self.move_camera("ToPos", id=pos_id, speed=20)
            time.sleep(1)

        url = self._build_url("Snap")
        logging.debug("Start capture")

        for attempt in range(retries):
            try:
                logging.debug(f"attempt {attempt}")
                start_time = time.time()
                response = requests.get(url, verify=False, timeout=timeout)
                elapsed_time = time.time() - start_time
                logging.debug(f"Capture attempt {attempt + 1}: Response received in {elapsed_time:.2f} seconds")

                if response.status_code == 200:
                    image_data = BytesIO(response.content)
                    image = Image.open(image_data)
                    return image
                else:
                    logging.error(f"Failed to capture image: {response.status_code}, {response.text}")
            except requests.RequestException as e:
                logging.error(f"Request failed (attempt {attempt + 1}): {e}")

            time.sleep(2)  # Wait a bit before retrying

        logging.error(f"Failed to capture image after {retries} attempts")
        return None

    def move_camera(self, operation: str, speed: int = 1, id: int = 0):
        """
        Sends a command to move the camera.

        Args:
            operation (str): The operation to perform, e.g., 'Left', 'Right'.
            speed (int): The speed of the operation.
            id (int): The ID of the position to move to (relevant for PTZ cameras).
        """
        url = self._build_url("PtzCtrl")
        data = [{"cmd": "PtzCtrl", "action": 0, "param": {"channel": 0, "op": operation, "id": id, "speed": speed}}]
        response = requests.post(url, json=data, verify=False)
        self._handle_response(response, "PTZ operation successful.")

    def move_in_seconds(self, s: int, operation: str = "Right", speed: int = 1):
        """
        Moves the camera in a specified direction for a specified number of seconds.

        Args:
            s (int): Duration in seconds to move the camera.
            operation (str): Direction to move the camera.
            speed (int): Speed of the movement.
        """
        self.move_camera(operation, speed)
        time.sleep(s)
        self.move_camera("Stop")
        time.sleep(1)

    def get_ptz_preset(self):
        """
        Retrieves the preset positions available for PTZ cameras.

        Returns:
            List[Dict]: A list of preset positions and their details if successful, else None.
        """
        url = self._build_url("GetPtzPreset")
        data = [{"cmd": "GetPtzPreset", "action": 1, "param": {"channel": 0}}]
        response = requests.post(url, json=data, verify=False)
        response_data = self._handle_response(response, "Presets retrieved successfully.")
        if response_data[0]["code"] == 0:
            return response_data[0].get("value", {}).get("PtzPreset", [])
        else:
            return response_data

    def set_ptz_preset(self, id: Optional[int] = None):
        """
        Sets a PTZ preset position. If no ID is provided, finds the next available slot.

        Args:
            id (Optional[int]): The preset ID to set. If None, finds an available ID automatically.

        Raises:
            ValueError: If no slots are available for new presets.
        """
        if id is None:
            presets_ptz = self.get_ptz_preset()
            for cfg in presets_ptz:
                if cfg["enable"] == 0:
                    id = cfg["id"]
                    break
            if id is None:
                raise ValueError("No available slots for new presets.")

        url = self._build_url("SetPtzPreset")
        name = f"pos{id}"
        data = [
            {
                "cmd": "SetPtzPreset",
                "action": 0,  # The action code for setting data
                "param": {"PtzPreset": {"channel": 0, "enable": 1, "id": id, "name": name}},
            }
        ]
        response = requests.post(url, json=data, verify=False)
        # Utilizing the shared response handling method
        self._handle_response(response, f"Preset {name} set successfully.")
