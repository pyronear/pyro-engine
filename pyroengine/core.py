# Copyright (C) 2022-2024, Pyronear.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

import asyncio
import time
from datetime import datetime
from typing import Any, List

import numpy as np
import urllib3

from .engine import Engine
from .logger_config import logger
from .sensors import ReolinkCamera

__all__ = ["SystemController", "is_day_time"]

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


def is_day_time(cache, frame, strategy, delta=0):
    """
    Determines if it is daytime using specified strategies.

    Strategies:
    1. Time-based: Compares the current time with sunrise and sunset times.
    2. IR-based: Analyzes the color of the image; IR cameras produce black and white images at night.

    Args:
        cache (Path): Cache folder where `sunset_sunrise.txt` is located.
        frame (PIL.Image): Frame to analyze with the IR strategy.
        strategy (str): Strategy to define daytime ("time", "ir", or "both").
        delta (int): Time delta in seconds before and after sunrise/sunset.

    Returns:
        bool: True if it is daytime, False otherwise.
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


async def capture_camera_image(camera: ReolinkCamera, image_queue: asyncio.Queue) -> bool:
    """
    Captures an image from the camera and puts it into a queue. Returns whether it is daytime for this camera.

    Args:
        camera (ReolinkCamera): The camera instance.
        image_queue (asyncio.Queue): The queue to put the captured image.

    Returns:
        bool: True if it is daytime according to this camera, False otherwise.
    """
    cam_id = camera.ip_address
    try:
        if camera.cam_type == "ptz":
            for idx, pose_id in enumerate(camera.cam_poses):
                cam_id = f"{camera.ip_address}"
                frame = camera.capture()
                # Move camera to the next pose to avoid waiting
                next_pos_id = camera.cam_poses[(idx + 1) % len(camera.cam_poses)]
                camera.move_camera("ToPos", idx=int(next_pos_id), speed=50)
                if frame is not None:
                    await image_queue.put((cam_id, frame))
                    await asyncio.sleep(0)  # Yield control
                    if not is_day_time(None, frame, "ir"):
                        return False
        else:
            frame = camera.capture()
            if frame is not None:
                await image_queue.put((cam_id, frame))
                await asyncio.sleep(0)  # Yield control
                if not is_day_time(None, frame, "ir"):
                    return False
    except Exception as e:
        logger.exception(f"Error during image capture from camera {cam_id}: {e}")
    return True


class SystemController:
    """
    Controls the system for capturing and analyzing camera streams.

    Attributes:
        engine (Engine): The image analyzer engine.
        cameras (List[ReolinkCamera]): List of cameras to capture streams from.
    """

    def __init__(self, engine: Engine, cameras: List[ReolinkCamera]) -> None:
        """
        Initializes the SystemController.

        Args:
            engine (Engine): The image analyzer engine.
            cameras (List[ReolinkCamera]): List of cameras to capture streams from.
        """
        self.engine = engine
        self.cameras = cameras
        self.is_day = True

    async def capture_images(self, image_queue: asyncio.Queue) -> bool:
        """
        Captures images from all cameras using asyncio.

        Args:
            image_queue (asyncio.Queue): The queue to put the captured images.

        Returns:
            bool: True if it is daytime according to all cameras, False otherwise.
        """
        tasks = [capture_camera_image(camera, image_queue) for camera in self.cameras]
        day_times = await asyncio.gather(*tasks)
        return all(day_times)

    async def analyze_stream(self, image_queue: asyncio.Queue) -> None:
        """
        Analyzes the image stream from the queue.

        Args:
            image_queue (asyncio.Queue): The queue with images to analyze.
        """
        while True:
            item = await image_queue.get()
            if item is None:
                break
            cam_id, frame = item
            try:
                self.engine.predict(frame, cam_id)
            except Exception as e:
                logger.error(f"Error running prediction: {e}")
            finally:
                image_queue.task_done()  # Mark the task as done

    async def night_mode(self) -> bool:
        """
        Checks and updates the day_time attribute based on the current frame.
        """
        try:
            frame = self.cameras[0].capture()
            if frame is not None:
                self.day_time = is_day_time(None, frame, "ir")
        except Exception as e:
            logger.exception(f"Exception during initial day time check: {e}")

        for camera in self.cameras:
            cam_id = camera.ip_address
            try:
                if camera.cam_type == "ptz":
                    for idx, pose_id in enumerate(camera.cam_poses):
                        cam_id = f"{camera.ip_address}_{pose_id}"
                        frame = camera.capture()
                        # Move camera to the next pose to avoid waiting
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
                logger.exception(f"Error during image capture from camera {cam_id}: {e}")
        return True

    async def run(self, period: int = 30, send_alerts: bool = True) -> bool:
        """
        Captures and analyzes all camera streams, then processes alerts.

        Args:
            period (int): The time period between captures in seconds.
            send_alerts (bool): Boolean to activate / deactivate alert sending.

        Returns:
            bool: True if it is daytime according to all cameras, False otherwise.
        """
        try:
            image_queue: asyncio.Queue[Any] = asyncio.Queue()

            # Start the image processor task
            processor_task = asyncio.create_task(self.analyze_stream(image_queue))

            # Capture images concurrently
            self.is_day = await self.capture_images(image_queue)

            # Wait for the image processor to finish processing
            await image_queue.join()  # Ensure all tasks are marked as done

            # Signal the image processor to stop processing
            await image_queue.put(None)
            await processor_task  # Ensure the processor task completes

            # Process alerts
            if send_alerts:
                try:
                    self.engine._process_alerts(self.cameras)
                except Exception as e:
                    logger.exception(f"Error processing alerts: {e}")

            return self.is_day

        except Exception as e:
            logger.warning(f"Analyze stream error: {e}")
            return True

    async def main_loop(self, period: int, send_alerts: bool = True) -> None:
        """
        Main loop to capture and process images at regular intervals.

        Args:
            period (int): The time period between captures in seconds.
            send_alerts (bool): Boolean to activate / deactivate alert sending.
        """
        while True:
            start_ts = time.time()
            await self.run(period, send_alerts)

            if not self.is_day:
                while not await self.night_mode():
                    logger.info("Nighttime detected by at least one camera, sleeping for 1 hour.")
                    await asyncio.sleep(3600)  # Sleep for 1 hour
            else:
                # Sleep only once all images are processed
                loop_time = time.time() - start_ts
                sleep_time = max(period - (loop_time), 0)
                logger.info(f"Loop run under {loop_time:.2f} seconds, sleeping for {sleep_time:.2f}")
                await asyncio.sleep(sleep_time)

    def __repr__(self) -> str:
        """
        Returns a string representation of the SystemController.

        Returns:
            str: A string representation of the SystemController.
        """
        repr_str = f"{self.__class__.__name__}("
        for cam in self.cameras:
            repr_str += f"\n\t{repr(cam)},"
        return repr_str + "\n)"
