import asyncio
import logging
import time
from datetime import datetime
from typing import List, Optional

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


async def capture_camera_image(camera: ReolinkCamera, image_queue: asyncio.Queue) -> None:
    """
    Captures an image from the camera and puts it into a queue.

    Args:
        camera (ReolinkCamera): The camera instance.
        image_queue (asyncio.Queue): The queue to put the captured image.
    """
    cam_id = camera.ip_address
    try:
        if camera.cam_type == "ptz":
            for idx, pose_id in enumerate(camera.cam_poses):
                cam_id = f"{camera.ip_address}_{pose_id}"
                frame = camera.capture()
                # In order to avoid waiting for the camera to move we move it to the next pose
                next_pos_id = camera.cam_poses[(idx + 1) % len(camera.cam_poses)]
                camera.move_camera("ToPos", idx=int(next_pos_id), speed=50)
                if frame is not None:
                    await image_queue.put((cam_id, frame))
                    await asyncio.sleep(0)  # Yield control
        else:
            frame = camera.capture()
            if frame is not None:
                await image_queue.put((cam_id, frame))
                await asyncio.sleep(0)  # Yield control
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

    async def capture_images(self, image_queue: asyncio.Queue) -> None:
        """
        Captures images from all cameras using asyncio.

        Args:
            image_queue (asyncio.Queue): The queue to put the captured images.
        """
        tasks = [capture_camera_image(camera, image_queue) for camera in self.cameras]
        await asyncio.gather(*tasks)

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
                start_inference_time = time.time()
                self.engine.predict(frame, cam_id)
                logging.info(f"Inference in {time.time() - start_inference_time:.2f} seconds")
            except Exception as e:
                logging.error(f"Error running prediction: {e}")
            finally:
                if frame is not None:
                    self.day_time = is_day_time(None, frame, "ir")
                    if not self.day_time:
                        logging.info("Switch to night mode")
                image_queue.task_done()  # Mark the task as done

    def check_day_time(self) -> None:
        try:
            frame = self.cameras[0].capture()
            if frame is not None:
                self.day_time = is_day_time(None, frame, "ir")
        except Exception as e:
            logging.exception(f"Exception during initial day time check: {e}")

    async def run(self, period: int = 30) -> None:
        """
        Captures and analyzes all camera streams, then processes alerts.

        Args:
            period (int): The time period between captures in seconds.
        """
        try:
            if not self.day_time:
                self.check_day_time()

            if self.day_time:
                start_time = time.time()
                image_queue = asyncio.Queue()

                # Start the image processor task
                processor_task = asyncio.create_task(self.analyze_stream(image_queue))

                # Capture images concurrently
                await self.capture_images(image_queue)

                # Wait for the image processor to finish processing
                await image_queue.join()  # Ensure all tasks are marked as done

                # Signal the image processor to stop processing
                await image_queue.put(None)
                await processor_task  # Ensure the processor task completes

                logging.info(f"Total time for capturing and processing images: {time.time() - start_time:.2f} seconds")

                # # Process alerts
                # try:
                #     self.engine._process_alerts()
                # except Exception as e:
                #     logging.error(f"Error processing alerts: {e}")

        except Exception as e:
            logging.warning(f"Analyze stream error: {e}")

    async def main_loop(self, period: int):
        while True:
            start_ts = time.time()
            await self.run(period)
            # Sleep only once all images are processed
            await asyncio.sleep(max(period - (time.time() - start_ts), 0))

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
