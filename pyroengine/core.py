# Copyright (C) 2022-2024, Pyronear.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

import logging
import signal
from datetime import datetime
from multiprocessing import Manager, Pool
from multiprocessing import Queue as MPQueue
from types import FrameType
from typing import List, Optional, Tuple, cast

import numpy as np
import urllib3
from PIL import Image

from .engine import Engine
from .sensors import ReolinkCamera

__all__ = ["SystemController", "is_day_time"]


urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Configure logging
logging.basicConfig(format="%(asctime)s | %(levelname)s: %(message)s", level=logging.INFO, force=True)


def is_day_time(cache, frame, strategy, delta=0):
    """This function allows to know if it is daytime or not. We have two strategies.
    The first one is to take the current time and compare it to the sunset time.
    The second is to see if we have a color image. The ir cameras switch to ir mode at night and
    therefore produce black and white images. This function can use one or more strategies depending on the use case.

    Args:
        cache (Path): cache folder where sunset_sunrise.txt is located
        frame (PIL image): frame to analyze with ir strategy
        strategy (str): Strategy to define day time [None, time, ir or both]
        delta (int): delta before and after sunset / sunrise in sec

    Returns:
        bool: is day time
    """
    is_day = True
    if strategy in ["both", "time"]:
        with open(cache.joinpath("sunset_sunrise.txt")) as f:
            lines = f.readlines()
        sunrise = datetime.strptime(lines[0][:-1], "%H:%M")
        sunset = datetime.strptime(lines[1][:-1], "%H:%M")
        now = datetime.strptime(datetime.now().isoformat().split("T")[1][:5], "%H:%M")
        if (now - sunrise).total_seconds() < -delta or (sunset - now).total_seconds() < -delta:
            is_day = False

    if strategy in ["both", "ir"]:
        frame = np.array(frame)
        if np.max(frame[:, :, 0] - frame[:, :, 1]) == 0:
            is_day = False

    return is_day


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

    cam_id = camera.ip_address
    try:
        if camera.cam_type == "ptz":
            for pose_id in camera.cam_poses:
                cam_id = f"{camera.ip_address}_{pose_id}"
                frame = camera.capture(pose_id)
                queue.put((cam_id, frame))
        else:
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
        self.day_time = True

    def capture_images(self) -> MPQueue:
        """
        Captures images from all cameras using multiprocessing.

        Returns:
            MPQueue: A queue containing the captured images and their camera IDs.
        """

        manager = Manager()
        queue: MPQueue = cast(MPQueue, manager.Queue())  # Cast to MPQueue

        # Create a list of arguments to pass to capture_camera_image
        args_list: List[Tuple[ReolinkCamera, MPQueue]] = [(camera, queue) for camera in self.cameras]

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

            if not self.day_time:
                try:
                    frame = self.cameras[0].capture()
                    self.day_time = is_day_time(None, frame, "ir")
                except Exception as e:
                    logging.exception(f"Exception during initial day time check: {e}")

            else:

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

                    # Use the last frame to check if it's day_time
                    self.day_time = is_day_time(None, img, "ir")

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
