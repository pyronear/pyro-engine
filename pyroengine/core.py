# Copyright (C) 2022-2025, Pyronear.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.


import logging
import time
from datetime import datetime
from io import BytesIO
from typing import Any, Dict, Optional

import numpy as np
import requests
import urllib3
from PIL import Image

from .engine import Engine

__all__ = ["SystemController", "is_day_time"]

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Configure logging
logging.basicConfig(format="%(asctime)s | %(levelname)s: %(message)s", level=logging.INFO, force=True)


def is_day_time(cache, frame, strategy, delta=0):
    """
    Determine whether it is daytime based on the selected strategy.

    Args:
        cache (Path): Cache folder containing the `sunset_sunrise.txt` file (for time-based strategy).
        frame (PIL.Image): Image frame to analyze (for IR-based strategy).
        strategy (str): Strategy to determine daytime ("time", "ir", or "both").
        delta (int, optional): Tolerance (in seconds) around sunrise/sunset for day/night transition.

    Returns:
        bool: True if it is considered daytime, False otherwise.
    """
    return True
    is_day = True
    if cache and strategy in ["both", "time"]:
        with open(cache.joinpath("sunset_sunrise.txt")) as f:
            lines = f.readlines()
        sunrise = datetime.strptime(lines[0].strip(), "%H:%M")
        sunset = datetime.strptime(lines[1].strip(), "%H:%M")
        now = datetime.strptime(datetime.now().isoformat().split("T")[1][:5], "%H:%M")
        if (now - sunrise).total_seconds() < -delta or (sunset - now).total_seconds() < -delta:
            is_day = False

    if strategy in ["both", "ir"]:
        frame = np.array(frame)
        if np.max(frame[:, :, 0] - frame[:, :, 1]) == 0:
            is_day = False

    return is_day


class SystemController:
    """
    Controller to manage multiple cameras, capture images, and perform detection.

    Attributes:
        engine (Engine): Image detection engine.
        cameras (List[ReolinkCamera]): List of camera instances.
        mediamtx_server_ip (str): IP address of the MediaMTX server (optional).
    """

    def __init__(
        self,
        engine: Engine,
        camera_data: Dict[str, Dict[str, Any]],
        reolink_api_url,
        mediamtx_server_ip: Optional[str] = None,
    ) -> None:
        """
        Initialize the system controller.
        """
        self.engine = engine
        self.camera_data = camera_data
        self.is_day = True
        self.reolink_api_url = reolink_api_url
        self.mediamtx_server_ip = mediamtx_server_ip
        self.last_autofocus: Optional[datetime] = None

        if self.mediamtx_server_ip:
            logging.info(f"Using MediaMTX server IP: {self.mediamtx_server_ip}")
        else:
            logging.info("No MediaMTX server IP provided, skipping levée de doute checks.")

    def main_loop(self, period: int, send_alerts: bool = True) -> None:
        """
        Run the main loop that regularly captures and analyzes camera feeds.

        Args:
            period (int): Interval between analysis loops (in seconds).
            send_alerts (bool, optional): Whether to trigger alerts after analysis.
        """
        while True:
            start_ts = time.time()

            if not self.is_day:
                logging.info("Nighttime detected by at least one camera, sleeping for 1 hour.")
                time.sleep(3600)
            else:
                for ip, cam in self.camera_data.items():
                    token = cam.get("token")
                    headers = {"Authorization": f"Bearer {token}"} if token else {}
                    camera_name = cam["name"]

                    if cam.get("type") == "ptz":
                        for pose in cam.get("poses", []):
                            try:
                                cam_id = f"{ip}_{pose}"
                                response = requests.get(
                                    f"{self.reolink_api_url}/latest_image",
                                    params={"camera_ip": ip, "pose": pose},
                                    headers=headers,
                                    timeout=3,
                                )
                                response.raise_for_status()

                                frame = Image.open(BytesIO(response.content)).convert("RGB")
                                logging.info(f"Captured image for {ip}, pose {pose}")

                                self.is_day = is_day_time(None, frame, "ir")

                                self.engine.predict(frame, cam_id)

                            except requests.HTTPError as e:
                                logging.error(f"❌ HTTP error for {camera_name}, pose {pose}: {e.response.text}")
                            except Exception as e:
                                logging.error(f"❌ Error for {camera_name}, pose {pose}: {e}")

                    else:
                        print("static cam todo")

                loop_time = time.time() - start_ts
                sleep_time = max(period - loop_time, 0)
                logging.info(f"Loop run under {loop_time:.2f} seconds, sleeping for {sleep_time:.2f} seconds")
                time.sleep(sleep_time)
