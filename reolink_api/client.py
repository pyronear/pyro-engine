from io import BytesIO
from typing import Optional

import requests
from PIL import Image


class ReolinkAPIClient:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")

    def list_cameras(self):
        resp = requests.get(f"{self.base_url}/cameras")
        resp.raise_for_status()
        return resp.json()

    def capture_image(self, camera_ip: str, pos_id: Optional[int] = None) -> Image.Image:
        params = {"camera_ip": camera_ip}
        if pos_id is not None:
            params["pos_id"] = pos_id
        resp = requests.get(f"{self.base_url}/capture", params=params)
        resp.raise_for_status()
        return Image.open(BytesIO(resp.content))

    def get_latest_image(self, camera_ip: str, pose: int) -> Image.Image:
        params = {"camera_ip": camera_ip, "pose": pose}
        resp = requests.get(f"{self.base_url}/latest_image", params=params)
        resp.raise_for_status()
        return Image.open(BytesIO(resp.content))

    def move_camera(self, camera_ip: str, operation: str, speed: int = 20, idx: int = 0):
        data = {"camera_ip": camera_ip, "operation": operation, "speed": speed, "idx": idx}
        resp = requests.post(f"{self.base_url}/move", params=data)
        resp.raise_for_status()
        return resp.json()

    def move_in_seconds(self, camera_ip: str, seconds: float, operation: str = "Right", speed: int = 20):
        data = {"camera_ip": camera_ip, "seconds": seconds, "operation": operation, "speed": speed}
        resp = requests.post(f"{self.base_url}/move_in_seconds", params=data)
        resp.raise_for_status()
        return resp.json()

    def manual_focus(self, camera_ip: str, position: int):
        data = {"camera_ip": camera_ip, "position": position}
        resp = requests.post(f"{self.base_url}/focus/manual", params=data)
        resp.raise_for_status()
        return resp.json()

    def toggle_autofocus(self, camera_ip: str, disable: bool = True):
        data = {"camera_ip": camera_ip, "disable": disable}
        resp = requests.post(f"{self.base_url}/focus/autofocus", params=data)
        resp.raise_for_status()
        return resp.json()

    def get_focus_status(self, camera_ip: str):
        params = {"camera_ip": camera_ip}
        resp = requests.get(f"{self.base_url}/focus/status", params=params)
        resp.raise_for_status()
        return resp.json()

    def list_presets(self, camera_ip: str):
        params = {"camera_ip": camera_ip}
        resp = requests.get(f"{self.base_url}/preset/list", params=params)
        resp.raise_for_status()
        return resp.json()

    def set_preset(self, camera_ip: str, idx: Optional[int] = None):
        params = {"camera_ip": camera_ip}
        if idx is not None:
            params["idx"] = idx
        resp = requests.post(f"{self.base_url}/preset/set", params=params)
        resp.raise_for_status()
        return resp.json()

    def run_focus_finder(self, camera_ip: str, save_images: bool = False):
        params = {"camera_ip": camera_ip, "save_images": save_images}
        resp = requests.post(f"{self.base_url}/focus/focus_finder", params=params)
        resp.raise_for_status()
        return resp.json()

    def start_patrol(self, camera_ip: str):
        data = {"camera_ip": camera_ip}
        resp = requests.post(f"{self.base_url}/start_patrol", params=data)
        resp.raise_for_status()
        return resp.json()

    def stop_patrol(self, camera_ip: str):
        data = {"camera_ip": camera_ip}
        resp = requests.post(f"{self.base_url}/stop_patrol", params=data)
        resp.raise_for_status()
        return resp.json()
