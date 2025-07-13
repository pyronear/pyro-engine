# Copyright (C) 2022-2025, Pyronear.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.


import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import requests
import urllib3

# Add the parent folder of reolink_api to the import path
sys.path.append(str(Path(__file__).resolve().parent.parent / "reolink_api"))

from reolink_api_client import ReolinkAPIClient

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
        self.mediamtx_server_ip = mediamtx_server_ip
        self.last_autofocus: Optional[datetime] = None
        self.reolink_client = ReolinkAPIClient(reolink_api_url)

        for ip in self.camera_data.keys():
            self.reolink_client.start_patrol(ip)

        if self.mediamtx_server_ip:
            logging.info(f"Using MediaMTX server IP: {self.mediamtx_server_ip}")
        else:
            logging.info("No MediaMTX server IP provided, skipping levée de doute checks.")

    def focus_finder(
        self,
    ):
        now = datetime.now()
        print("focus_finder", self.is_day, self.last_autofocus)
        if self.is_day and (self.last_autofocus is None or (now - self.last_autofocus).total_seconds() > 3600):
            logging.info("🔄 Hourly autofocus triggered after idle period")
            for ip in self.camera_data.keys():
                try:
                    self.reolink_client.stop_patrol(ip)
                    time.sleep(0.5)
                    self.reolink_client.run_focus_optimization(ip)
                    logging.info(f"Autofocus completed for {ip}")
                    self.last_autofocus = now
                    self.reolink_client.start_patrol(ip)

                except Exception as e:
                    logging.error(f"[Failed to run hourly focus finder on camera {ip} : {e}")

    def inference_loop(
        self,
    ):
        for ip, cam in self.camera_data.items():
            camera_name = cam["name"]

            if cam.get("type") == "ptz":
                for pose in cam.get("poses", []):
                    try:
                        cam_id = f"{ip}_{pose}"

                        frame = self.reolink_client.get_latest_image(ip, pose)

                        logging.info(f"Captured image for {ip}, pose {pose}")

                        self.is_day = is_day_time(None, frame, "ir")

                        self.engine.predict(frame, cam_id)

                    except requests.HTTPError as e:
                        logging.error(f"❌ HTTP error for {camera_name}, pose {pose}: {e.response.text}")
                    except Exception as e:
                        logging.error(f"❌ Error for {camera_name}, pose {pose}: {e}")

            else:
                try:
                    cam_id = f"{ip}"

                    frame = self.reolink_client.get_latest_image(ip, -1)
                    logging.info(f"Captured image for {ip}")

                    self.is_day = is_day_time(None, frame, "ir")

                    self.engine.predict(frame, cam_id)

                except requests.HTTPError as e:
                    logging.error(f"❌ HTTP error for {camera_name}: {e.response.text}")
                except Exception as e:
                    logging.error(f"❌ Error for {camera_name}: {e}")

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
                if len(self.engine._alerts) and send_alerts:
                    try:
                        self.engine._process_alerts()
                    except Exception as e:
                        logging.error(f"Error processing alerts: {e}")
                else:
                    # Run Autofocus
                    logging.info("Run focus finder")
                    self.focus_finder()

                # Inference
                self.inference_loop()

                loop_time = time.time() - start_ts
                sleep_time = max(period - loop_time, 0)
                logging.info(f"Loop run under {loop_time:.2f} seconds, sleeping for {sleep_time:.2f} seconds")
                time.sleep(sleep_time)
