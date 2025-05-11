# Copyright (C) 2022-2025, Pyronear.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

import asyncio
import logging
import time
from datetime import datetime
from typing import Any, List, Optional

import aiohttp
import numpy as np
import urllib3

from .engine import Engine
from .sensors import ReolinkCamera

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
    if strategy in ["both", "time"]:
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


async def capture_camera_image(
    camera: ReolinkCamera,
    image_queue: asyncio.Queue,
    server_ip: Optional[str] = None
) -> bool:
    """
    Capture an image from a camera and enqueue it for analysis.

    Args:
        camera (ReolinkCamera): Camera object to capture an image from.
        image_queue (asyncio.Queue): Queue to store captured images.
        server_ip (str, optional): IP address of the live stream API (for levée de doute checks).

    Returns:
        bool: True if it is daytime according to the captured image, False otherwise.
    """
    cam_id = camera.ip_address
    try:
        if server_ip:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"http://127.0.0.1:8081/is_stream_running/{cam_id}") as resp:
                    data = await resp.json()
                    if data.get("running"):
                        logging.info(f"{cam_id} Camera is streaming, skipping capture.")
                        return True

        if camera.cam_type == "ptz":
            for idx, pose_id in enumerate(camera.cam_poses):
                cam_id = f"{camera.ip_address}_{pose_id}"
                frame = camera.capture()
                next_pos_id = camera.cam_poses[(idx + 1) % len(camera.cam_poses)]
                camera.move_camera("ToPos", idx=int(next_pos_id), speed=50)
                if frame is not None:
                    await image_queue.put((cam_id, frame))
                    await asyncio.sleep(0)
                    if not is_day_time(None, frame, "ir"):
                        return False
        else:
            frame = camera.capture()
            if frame is not None:
                await image_queue.put((cam_id, frame))
                await asyncio.sleep(0)
                if not is_day_time(None, frame, "ir"):
                    return False
    except Exception as e:
        logging.exception(f"Error during image capture from camera {cam_id}: {e}")
    return True


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
    cameras: List[ReolinkCamera],
    mediamtx_server_ip: Optional[str] = None
) -> None:
        """
        Initialize the system controller.

        Args:
            engine (Engine): Image analysis engine.
            cameras (List[ReolinkCamera]): List of camera objects.
            mediamtx_server_ip (str, optional): IP address of the MediaMTX server.
        """
        self.engine = engine
        self.cameras = cameras
        self.is_day = True
        self.mediamtx_server_ip = mediamtx_server_ip

        if self.mediamtx_server_ip:
            logging.info(f"Using MediaMTX server IP: {self.mediamtx_server_ip}")
        else:
            logging.info("No MediaMTX server IP provided, skipping levée de doute checks.")

    async def capture_images(self, image_queue: asyncio.Queue) -> bool:
        """
        Capture images from all cameras concurrently.

        Args:
            image_queue (asyncio.Queue): Queue to store captured images.

        Returns:
            bool: True if all cameras detect daytime, False otherwise.
        """
        tasks = [
            capture_camera_image(camera, image_queue, server_ip=self.mediamtx_server_ip) for camera in self.cameras
        ]
        day_times = await asyncio.gather(*tasks)
        return all(day_times)

    async def analyze_stream(self, image_queue: asyncio.Queue) -> None:
        """
        Analyze incoming images from the queue.

        Args:
            image_queue (asyncio.Queue): Queue containing (camera_id, frame) tuples.
        """
        while True:
            item = await image_queue.get()
            if item is None:
                break
            cam_id, frame = item
            try:
                self.engine.predict(frame, cam_id)
            except Exception as e:
                logging.error(f"Error running prediction: {e}")
            finally:
                image_queue.task_done()

    async def night_mode(self) -> bool:
        """
        Check whether it is nighttime according to all cameras.

        Returns:
            bool: True if it is daytime for all cameras, False otherwise.
        """
        for camera in self.cameras:
            cam_id = camera.ip_address
            try:
                if camera.cam_type == "ptz":
                    for idx, pose_id in enumerate(camera.cam_poses):
                        cam_id = f"{camera.ip_address}_{pose_id}"
                        frame = camera.capture()
                        next_pos_id = camera.cam_poses[(idx + 1) % len(camera.cam_poses)]
                        camera.move_camera("ToPos", idx=int(next_pos_id), speed=50)
                        if frame is not None:
                            if not is_day_time(None, frame, "ir"):
                                return False
                else:
                    frame = camera.capture()
                    if frame is not None:
                        if not is_day_time(None, frame, "ir"):
                            return False
            except Exception as e:
                logging.exception(f"Error during image capture from camera {cam_id}: {e}")
        return True

    async def run(self, period: int = 30, send_alerts: bool = True) -> bool:
        """
        Capture images, analyze them, and process alerts if needed.

        Args:
            period (int, optional): Time between each capture loop (seconds).
            send_alerts (bool, optional): Whether to process and send alerts.

        Returns:
            bool: True if it is daytime according to all cameras, False otherwise.
        """
        try:
            image_queue: asyncio.Queue[Any] = asyncio.Queue()

            processor_task = asyncio.create_task(self.analyze_stream(image_queue))
            self.is_day = await self.capture_images(image_queue)

            await image_queue.join()
            await image_queue.put(None)
            await processor_task

            if send_alerts:
                try:
                    self.engine._process_alerts()
                except Exception as e:
                    logging.error(f"Error processing alerts: {e}")

            return self.is_day

        except Exception as e:
            logging.warning(f"Analyze stream error: {e}")
            return True

    async def main_loop(self, period: int, send_alerts: bool = True) -> None:
        """
        Run the main loop that regularly captures and analyzes camera feeds.

        Args:
            period (int): Interval between analysis loops (in seconds).
            send_alerts (bool, optional): Whether to trigger alerts after analysis.
        """
        while True:
            start_ts = time.time()
            await self.run(period, send_alerts)
            if not self.is_day:
                while not await self.night_mode():
                    logging.info("Nighttime detected by at least one camera, sleeping for 1 hour.")
                    await asyncio.sleep(3600)
            else:
                loop_time = time.time() - start_ts
                sleep_time = max(period - loop_time, 0)
                logging.info(f"Loop run under {loop_time:.2f} seconds, sleeping for {sleep_time:.2f} seconds")
                await asyncio.sleep(sleep_time)

    def __repr__(self) -> str:
        """
        Represent the SystemController with its list of cameras.

        Returns:
            str: String representation.
        """
        repr_str = f"{self.__class__.__name__}("
        for cam in self.cameras:
            repr_str += f"\n\t{cam!r},"
        return repr_str + "\n)"
