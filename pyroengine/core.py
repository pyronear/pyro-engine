# Copyright (C) 2022-2026, Pyronear.

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
from PIL import Image, UnidentifiedImageError

# Add the parent folder of pyro_camera_api to the import path
sys.path.append(str(Path(__file__).resolve().parent.parent / "pyro_camera_api"))

from pyro_camera_api_client.client import PyroCameraAPIClient

from .engine import Engine

__all__ = ["SystemController", "is_day_time"]

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

logging.basicConfig(format="%(asctime)s | %(levelname)s: %(message)s", level=logging.INFO, force=True)
logger = logging.getLogger(__name__)


def is_day_time(cache: Optional[Path], frame: Image.Image, strategy: str, delta: int = 0) -> bool:
    """
    Determine whether it is daytime based on the selected strategy.

    Args:
        cache (Path): Cache folder containing the `sunset_sunrise.txt` file (for time based strategy).
        frame (PIL.Image): Image frame to analyze (for IR based strategy).
        strategy (str): Strategy to determine daytime, one of "time", "ir", or "both".
        delta (int, optional): Tolerance in seconds around sunrise or sunset for day and night transition.

    Returns:
        bool: True if it is considered daytime, False otherwise.
    """
    is_day = True
    if cache and strategy in ["both", "time"]:
        with Path(cache.joinpath("sunset_sunrise.txt")).open() as f:
            lines = f.readlines()
        sunrise = datetime.strptime(lines[0].strip(), "%H:%M")
        sunset = datetime.strptime(lines[1].strip(), "%H:%M")
        now = datetime.strptime(datetime.now().isoformat().split("T")[1][:5], "%H:%M")
        if (now - sunrise).total_seconds() < -delta or (sunset - now).total_seconds() < -delta:
            is_day = False

    if strategy in ["both", "ir"]:
        frame_arr = np.array(frame)
        if np.max(frame_arr[:, :, 0] - frame_arr[:, :, 1]) == 0:
            is_day = False

    return is_day


class SystemController:
    """
    Controller to manage multiple cameras, capture images, and perform detection.
    """

    API_INITIAL_WAIT = 30
    API_RETRY_DELAY = 10
    POST_READY_WAIT = 10

    def __init__(
        self,
        engine: Engine,
        camera_data: Dict[str, Dict[str, Any]],
        pyro_camera_api_url: str,
    ) -> None:
        self.engine = engine
        self.camera_data = camera_data
        self.is_day = True
        # Last focus timestamp seen per camera IP, to re-upload an image after a focus change
        self._last_focus_seen: Dict[str, float] = {}

        # Wait for the camera API to be available
        time.sleep(self.API_INITIAL_WAIT)
        while True:
            try:
                logger.info("Waiting for Pyro Camera API")
                self.camera_api_client = PyroCameraAPIClient(pyro_camera_api_url)
                _ = self.camera_api_client.get_stream_status()
                logger.info("Pyro Camera API client ready")
                break
            except Exception as e:
                logger.error(f"API not ready: {e}")
                time.sleep(self.API_RETRY_DELAY)

        # Optional startup actions, do not fail hard
        for ip in self.camera_data:
            try:
                self.camera_api_client.start_patrol(ip)
            except Exception as e:
                logger.warning(f"Could not start patrol on {ip} at startup, continuing: {e}")

        # Wait and then loop until inference passes once
        time.sleep(self.POST_READY_WAIT)
        while True:
            try:
                logger.info("Waiting for cameras")
                self.inference_loop()
                break
            except Exception as e:
                logger.error(f"Inference failed: {e}")
                time.sleep(self.API_RETRY_DELAY)

    def _any_stream_active(self) -> bool:
        """
        Return True if any stream is active.

        Supports both the new keys active_pipelines and active_ffmpeg
        and the legacy key active_streams used in older APIs and tests.
        """
        try:
            status = self.camera_api_client.get_stream_status()

            # New format with explicit lists
            active_pipelines = status.get("active_pipelines")
            active_ffmpeg = status.get("active_ffmpeg")
            if active_pipelines is not None or active_ffmpeg is not None:
                return bool(active_pipelines or active_ffmpeg)

            # Backward compatible support for legacy field
            active_streams = status.get("active_streams")
            if active_streams is not None:
                try:
                    return int(active_streams) > 0
                except (TypeError, ValueError):
                    return bool(active_streams)

            return False
        except Exception as e:
            logger.error(f"Could not fetch stream status: {e}")
            return False

    def _safe_get_latest_image(self, ip: str, pose: int) -> Optional[Image.Image]:
        try:
            return self.camera_api_client.get_latest_image(ip, pose)
        except UnidentifiedImageError:
            return None
        except Exception as e:
            logger.error(f"Error getting image for {ip} pose {pose}: {e}")
            return None

    def inference_loop(self) -> None:
        """
        Run one inference pass on all cameras and poses.

        This skips processing entirely if a stream is currently active.
        """
        if self._any_stream_active():
            logger.info("Stream detected, skipping inference on all cameras")
            return

        for ip, cam in self.camera_data.items():
            camera_name = cam["name"]

            if cam.get("type") == "ptz":
                for pose in cam.get("poses", []):
                    if self._any_stream_active():
                        logger.info("Stream turned on during loop, stopping inference immediately")
                        return
                    try:
                        cam_id = f"{ip}_{pose}"
                        frame = self._safe_get_latest_image(ip, pose)
                        if frame is not None:
                            logger.info(f"Captured image for {ip}, pose {pose}")
                            self.is_day = is_day_time(None, frame, "ir")
                            self.engine.predict(frame, cam_id)
                    except requests.HTTPError as e:
                        logger.error(
                            f"HTTP error for {camera_name}, pose {pose}: "
                            f"{e.response.text if e.response is not None else e}"
                        )
                    except Exception as e:
                        logger.error(f"Error for {camera_name}, pose {pose}: {e}")

            else:
                if self._any_stream_active():
                    logger.info("Stream turned on during loop, stopping inference immediately")
                    return
                try:
                    cam_id = ip
                    frame = self._safe_get_latest_image(ip, -1)
                    if frame is not None:
                        logger.info(f"Captured image for {ip}")
                        self.is_day = is_day_time(None, frame, "ir")
                        self.engine.predict(frame, cam_id)
                except requests.HTTPError as e:
                    logger.error(f"HTTP error for {camera_name}: {e.response.text if e.response is not None else e}")
                except Exception as e:
                    logger.error(f"Error for {camera_name}: {e}")

    def check_and_restart_patrol(self) -> None:
        """
        Check stream activity and ensure patrol is running when no stream is active.
        """
        try:
            stream_status = self.camera_api_client.get_stream_status()
        except Exception as e:
            logger.error(f"Could not check if stream is running: {e}")
            return

        active_pipelines = stream_status.get("active_pipelines") or []
        active_ffmpeg = stream_status.get("active_ffmpeg") or []
        if not active_pipelines and not active_ffmpeg:
            for ip in self.camera_data:
                try:
                    patrol_status = self.camera_api_client.get_patrol_status(ip)
                    last_focus = float(patrol_status.get("last_focus_time") or 0.0)
                    if last_focus > self._last_focus_seen.get(ip, 0.0):
                        self._last_focus_seen[ip] = last_focus
                        self.engine.mark_periodic_images_stale(ip)
                        logger.info(f"Focus changed on camera {ip}, periodic images will be re-uploaded")
                    if not patrol_status.get("patrol_running", False):
                        self.camera_api_client.start_patrol(ip)
                        logger.info(f"Patrol restarted on camera {ip}")
                except Exception as e:
                    logger.error(f"Could not check or restart patrol on camera {ip}: {e}")

    def main_loop(self, period: int, send_alerts: bool = True) -> None:
        """
        Run the main control loop.

        This loop handles:
        detection alerts,
        patrol management,
        image inference for all cameras.

        Args:
            period (int): Interval between analysis loops in seconds.
            send_alerts (bool, optional): Whether to process detection alerts.
        """
        while True:
            start_ts = time.time()

            if not self.is_day:
                for ip in self.camera_data:
                    try:
                        patrol_status = self.camera_api_client.get_patrol_status(ip)
                        if not patrol_status.get("patrol_running", True):
                            self.camera_api_client.stop_patrol(ip)
                            logger.info(f"Stopped patrol for camera {ip} due to night")
                    except Exception as e:
                        logger.error(f"Failed to stop patrol on camera {ip}: {e}")

                logger.info("Nighttime detected by at least one camera, sleeping for 1 hour")
                time.sleep(3600)

                try:
                    ip = next(iter(self.camera_data.keys()))
                    frame = self.camera_api_client.capture_image(ip)
                    self.is_day = is_day_time(None, frame, "ir")
                    logger.info(f"Re checked day and night using camera {ip}, result is_day={self.is_day}")

                    if self.is_day:
                        logger.info("Day detected, restarting patrols")
                        self.check_and_restart_patrol()
                        time.sleep(30)
                        logger.info("Patrols restarted successfully, waiting 30 seconds before next check")
                except Exception as e:
                    logger.error(f"Failed to check day and night after sleep: {e}")
                    self.is_day = False

            else:
                if len(self.engine._alerts) and send_alerts:
                    try:
                        self.engine._process_alerts()
                    except Exception as e:
                        logger.error(f"Error processing alerts: {e}")

                self.check_and_restart_patrol()
                self.inference_loop()

                loop_time = time.time() - start_ts
                sleep_time = max(period - loop_time, 0)
                logger.info(f"Loop run under {loop_time:.2f} seconds, sleeping for {sleep_time:.2f} seconds")
                time.sleep(sleep_time)
