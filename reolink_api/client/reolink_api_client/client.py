# Copyright (C) 2025-2025, Pyronear.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.


from io import BytesIO
from typing import Optional

import requests
from PIL import Image


class ReolinkAPIClient:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")

    def list_cameras(self):
        resp = requests.get(f"{self.base_url}/info/cameras")
        resp.raise_for_status()
        return resp.json()

    def get_camera_infos(self):
        resp = requests.get(f"{self.base_url}/info/camera_infos")
        resp.raise_for_status()
        return resp.json()

    def capture_image(self, camera_ip: str, pos_id: Optional[int] = None) -> Image.Image:
        """Triggers a live capture from the camera."""
        params = {"camera_ip": camera_ip}
        if pos_id is not None:
            params["pos_id"] = pos_id
        resp = requests.get(f"{self.base_url}/capture/capture", params=params)
        resp.raise_for_status()
        return Image.open(BytesIO(resp.content))

    def get_latest_image(self, camera_ip: str, pose: int) -> Image.Image:
        """Fetches the last captured image for a given pose from memory."""
        params = {"camera_ip": camera_ip, "pose": pose}
        resp = requests.get(f"{self.base_url}/capture/latest_image", params=params)
        resp.raise_for_status()
        return Image.open(BytesIO(resp.content))

    def move_camera(
        self,
        camera_ip: str,
        direction: Optional[str] = None,
        speed: int = 10,
        pose_id: Optional[int] = None,
        degrees: Optional[float] = None,
    ):
        params = {"camera_ip": camera_ip, "speed": speed}
        if direction:
            params["direction"] = direction
        if pose_id is not None:
            params["pose_id"] = pose_id
        if degrees is not None:
            params["degrees"] = degrees

        resp = requests.post(f"{self.base_url}/control/move", params=params)
        resp.raise_for_status()
        return resp.json()

    def stop_camera(self, camera_ip: str):
        resp = requests.post(f"{self.base_url}/control/stop/{camera_ip}")
        resp.raise_for_status()
        return resp.json()

    def list_presets(self, camera_ip: str):
        resp = requests.get(f"{self.base_url}/control/preset/list", params={"camera_ip": camera_ip})
        resp.raise_for_status()
        return resp.json()

    def set_preset(self, camera_ip: str, idx: Optional[int] = None):
        params = {"camera_ip": camera_ip}
        if idx is not None:
            params["idx"] = idx
        resp = requests.post(f"{self.base_url}/control/preset/set", params=params)
        resp.raise_for_status()
        return resp.json()

    def zoom(self, camera_ip: str, level: int):
        resp = requests.post(f"{self.base_url}/control/zoom/{camera_ip}/{level}")
        resp.raise_for_status()
        return resp.json()

    def set_manual_focus(self, camera_ip: str, position: int):
        resp = requests.post(
            f"{self.base_url}/focus/manual",
            params={"camera_ip": camera_ip, "position": position},
        )
        resp.raise_for_status()
        return resp.json()

    def toggle_autofocus(self, camera_ip: str, disable: bool = True):
        resp = requests.post(
            f"{self.base_url}/focus/set_autofocus",
            params={"camera_ip": camera_ip, "disable": disable},
        )
        resp.raise_for_status()
        return resp.json()

    def get_focus_status(self, camera_ip: str):
        resp = requests.get(
            f"{self.base_url}/focus/status",
            params={"camera_ip": camera_ip},
        )
        resp.raise_for_status()
        return resp.json()

    def run_focus_optimization(self, camera_ip: str, save_images: bool = False):
        resp = requests.post(
            f"{self.base_url}/focus/focus_finder",
            params={"camera_ip": camera_ip, "save_images": save_images},
        )
        resp.raise_for_status()
        return resp.json()

    def start_patrol(self, camera_ip: str):
        resp = requests.post(f"{self.base_url}/patrol/start_patrol", params={"camera_ip": camera_ip})
        resp.raise_for_status()
        return resp.json()

    def stop_patrol(self, camera_ip: str):
        resp = requests.post(f"{self.base_url}/patrol/stop_patrol", params={"camera_ip": camera_ip})
        resp.raise_for_status()
        return resp.json()

    def get_patrol_status(self, camera_ip: str):
        resp = requests.get(f"{self.base_url}/patrol/patrol_status", params={"camera_ip": camera_ip})
        resp.raise_for_status()
        return resp.json()

    def start_stream(self, camera_ip: str):
        resp = requests.post(f"{self.base_url}/stream/start_stream/{camera_ip}")
        resp.raise_for_status()
        return resp.json()

    def stop_stream(self):
        resp = requests.post(f"{self.base_url}/stream/stop_stream")
        resp.raise_for_status()
        return resp.json()

    def get_stream_status(self):
        resp = requests.get(f"{self.base_url}/stream/status")
        resp.raise_for_status()
        return resp.json()

    def is_stream_running(self, camera_ip: str):
        resp = requests.get(f"{self.base_url}/stream/is_stream_running/{camera_ip}")
        resp.raise_for_status()
        return resp.json()
