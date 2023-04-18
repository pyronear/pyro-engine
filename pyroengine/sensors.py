# Copyright (C) 2022-2023, Pyronear.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

from io import BytesIO

import requests
import urllib3
from PIL import Image

__all__ = ["ReolinkCamera"]

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

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
