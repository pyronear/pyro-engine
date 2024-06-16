# Copyright (C) 2022-2024, Pyronear.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

import logging
import signal
from multiprocessing import Manager, Pool
from multiprocessing import Queue as MPQueue
from types import FrameType
from typing import List, Optional, Tuple

import urllib3
from PIL import Image

from .engine import Engine
from .sensors import ReolinkCamera

__all__ = ["SystemController"]

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Configure logging
logging.basicConfig(format="%(asctime)s | %(levelname)s: %(message)s", level=logging.INFO, force=True)


def handler(signum: int, frame: Optional[FrameType]) -> None:
    """
    Signal handler for timeout.

    Args:
        signum (int): The signal number.
        frame (Optional[FrameType]): The current stack frame (or None).
    """
    raise Exception("Analyze stream timeout")


def capture_camera_image(args: Tuple[ReolinkCamera, MPQueue]) -> None:
    """
    Captures an image from the camera and puts it into a queue.

    Args:
        args (tuple): A tuple containing the camera instance and a queue.
    """
    camera, queue = args
    if camera.cam_type == "ptz":
        for pose_id in camera.cam_poses:
            try:
                cam_id = f"{camera.ip_address}_{pose_id}"
                frame = camera.capture(pose_id)
                queue.put((cam_id, frame))
            except Exception as e:
                logging.exception(f"Error during image capture from camera {cam_id}: {e}")
    else:
        try:
            cam_id = camera.ip_address
            frame = camera.capture()
            queue.put((cam_id, frame))
        except Exception as e:
            logging.exception(f"Error during image capture from camera {cam_id}: {e}")


class SystemController:
    """
    Implements the full system controller for capturing and analyzing camera streams.

    Attributes:
        engine (Engine): The image analyzer engine.
        cameras (List[ReolinkCamera]): The list of cameras to get the visual streams from.
    """

    def __init__(self, engine: Engine, cameras: List[ReolinkCamera]) -> None:
        """
        Initializes the SystemController with an engine and a list of cameras.

        Args:
            engine (Engine): The image analyzer engine.
            cameras (List[ReolinkCamera]): The list of cameras to get the visual streams from.
        """
        self.engine = engine
        self.cameras = cameras

    def capture_images(self) -> MPQueue:
        """
        Captures images from all cameras using multiprocessing.

        Returns:
            MPQueue: A queue containing the captured images and their camera IDs.
        """
        manager = Manager()
        queue = manager.Queue()

        # Create a list of arguments to pass to capture_camera_image
        args_list = [(camera, queue) for camera in self.cameras]

        # Use a pool of processes to capture images concurrently
        with Pool(processes=len(self.cameras)) as pool:
            pool.map(capture_camera_image, args_list)

        return queue

    def analyze_stream(self, img: Image.Image, cam_id: str) -> None:
        """
        Analyzes the image stream from a specific camera.

        Args:
            img (Image.Image): The image to analyze.
            cam_id (str): The ID of the camera.
        """
        # Run the prediction using the engine
        self.engine.predict(img, cam_id)

    def run(self, period: int = 30) -> None:
        """
        Captures and analyzes all camera streams, then processes alerts.

        Args:
            period (int): The time period between captures in seconds.
        """

        try:
            # Set the signal alarm
            signal.signal(signal.SIGALRM, handler)
            signal.alarm(period)
            # Capture images
            queue = None
            try:
                queue = self.capture_images()
            except Exception as e:
                logging.error(f"Error capturing images: {e}")

            # Analyze each captured frame
            if queue:
                while not queue.empty():
                    cam_id, img = queue.get()
                    try:
                        self.analyze_stream(img, cam_id)
                    except Exception as e:
                        logging.error(f"Error running prediction: {e}")

            # Process alerts
            try:
                self.engine._process_alerts()
            except Exception as e:
                logging.error(f"Error processing alerts: {e}")

            # Disable the alarm
            signal.alarm(0)
        except Exception:
            logging.warning("Analyze stream timeout")

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
