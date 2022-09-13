# Copyright (C) 2022, Pyronear.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.

import logging
from io import BytesIO
from typing import List

import requests
import urllib3
from PIL import Image

from pyroengine import Engine

__all__ = ["ReolinkCamera", "SystemController"]

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

logging.basicConfig(format="%(asctime)s | %(levelname)s: %(message)s", level=logging.INFO, force=True)

CAM_URL = "https://{ip_address}/cgi-bin/api.cgi?cmd=Snap&channel=0&rs=wuuPhkmUCeI9WG7C&user={login}&password={password}"


class ReolinkCamera:
    """Implements a camera controller

    Args:
        ip_address: the local IP address to reach the Reolink camera
        login: login to access camera stream
        password: password to access camera stream
    """

    def __init__(self, ip_address: str, login: str, password: str) -> None:
        self.ip_address = ip_address
        self.login = login
        self._url = CAM_URL.format(ip_address=ip_address, login=login, password=password)
        # Check the connection
        assert isinstance(self.capture(), Image.Image)

    def capture(self) -> Image.Image:
        """Retrieves the camera stream"""
        response = requests.get(self._url, verify=False, timeout=3)
        return Image.open(BytesIO(response.content))

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(ip_address={self.ip_address}, login={self.login})"


class SystemController:
    """Implements the full system controller

    Args:
        engine: the image analyzer
        cameras: the cameras to get the visual streams from
    """

    def __init__(self, engine: Engine, cameras: List[ReolinkCamera]) -> None:
        self.engine = engine
        self.cameras = cameras

    def analyze_stream(self, idx: int) -> float:
        assert 0 <= idx < len(self.cameras)
        try:
            img = self.cameras[idx].capture()
            try:
                self.engine.predict(img, self.cameras[idx].ip_address)
            except Exception:
                logging.warning(f"Unable to analyze stream from camera {self.cameras[idx]}")
        except Exception:
            logging.warning(f"Unable to fetch stream from camera {self.cameras[idx]}")

    def run(self):
        """Analyzes all camera streams"""
        for idx in range(len(self.cameras)):
            self.analyze_stream(idx)

    def __repr__(self) -> str:
        repr_str = f"{self.__class__.__name__}("
        for cam in self.cameras:
            repr_str += f"\n\t{repr(cam)},"
        return repr_str + "\n)"
