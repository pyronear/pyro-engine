import logging
import signal
import time
from multiprocessing import Process, Queue, current_process
from queue import Empty, Full
from types import FrameType
from typing import Optional, Tuple

import numpy as np
import urllib3
from PIL import Image

from pyroengine.engine import is_day_time

__all__ = ["SystemController"]

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

logging.basicConfig(format="%(asctime)s | %(levelname)s: %(message)s", level=logging.INFO, force=True)
PredictionResult = Tuple[np.ndarray, Image.Image, str]


def handler(signum: int, frame: Optional[FrameType]) -> None:
    raise Exception("Analyze stream timeout")


def start_process(name, target, args):
    """Start a new process with the given name, target function, and arguments."""
    process = Process(name=name, target=target, args=args, daemon=True)
    process.start()
    return process


def terminate_processes(processes):
    """Terminate the given list of processes."""
    logging.info("Terminating processes due to signal interruption...")
    for process in processes:
        process.terminate()
        process.join()
    logging.info("Processes terminated successfully.")
    exit(0)


class SystemController:
    def __init__(self, engine, cameras):
        self.engine = engine
        self.cameras = cameras
        self.day_time = True

    def capture_images(self, capture_queue: Queue, capture_interval: int = 30):
        """Capture images from cameras and put them into the queue."""
        process_name = current_process().name
        logging.debug(f"[{process_name}] Capture process started")
        while True:
            try:
                start_ts = time.time()

                if not self.day_time:
                    try:
                        frame = self.cameras[0].capture()
                        self.day_time = is_day_time(None, frame, "ir")
                    except Exception as e:
                        logging.exception(f"[{process_name}] Exception during initial day time check: {e}")
                        continue

                try:
                    for idx, camera in enumerate(self.cameras):
                        try:
                            if camera.cam_type == "ptz":
                                for pose_id in camera.cam_poses:
                                    frame = camera.capture(pose_id)
                                    if frame is not None:
                                        logging.debug(
                                            f"[{process_name}] Captured frame from camera {camera.ip_address} at pose {pose_id}"
                                        )
                                        self.process_frame(idx, frame, capture_queue, camera, pose_id)
                                    else:
                                        logging.error(
                                            f"[{process_name}] Failed to capture image from camera {camera.ip_address} at pose {pose_id}"
                                        )
                            else:
                                frame = camera.capture()
                                if frame is not None:
                                    logging.debug(f"[{process_name}] Captured frame from camera {camera.ip_address}")
                                    self.process_frame(idx, frame, capture_queue, camera)
                                else:
                                    logging.error(
                                        f"[{process_name}] Failed to capture image from camera {camera.ip_address}"
                                    )
                        except Exception as e:
                            logging.exception(
                                f"[{process_name}] Exception during image capture from camera {camera.ip_address}: {e}"
                            )
                except Exception as e:
                    logging.exception(f"[{process_name}] Exception during image capture loop: {e}")

                sleep_duration = max(capture_interval - (time.time() - start_ts), 0)
                logging.debug(
                    f"[{process_name}] Sleeping for {sleep_duration:.2f} seconds, {capture_interval} {(time.time() - start_ts)}"
                )
                # Ensure capturing an image every capture_interval seconds
                time.sleep(sleep_duration)
            except Exception as e:
                logging.exception(f"[{process_name}] Unexpected error in capture process: {e}")
                time.sleep(1)

    def process_frame(self, idx, frame, capture_queue, camera, pose_id=None):
        """Process a captured frame and put it into the capture queue if conditions are met."""
        process_name = current_process().name
        self.day_time = is_day_time(None, frame, "ir")
        if self.day_time:
            try:
                cam_id = f"{camera.ip_address}_{pose_id}" if pose_id is not None else camera.ip_address
                capture_queue.put_nowait((idx, frame, cam_id, pose_id))
                logging.debug(f"[{process_name}] Putting frame from camera {camera.ip_address} into capture queue")
                logging.debug(f"[{process_name}] Capture queue size: {capture_queue.qsize()}")
            except Full:
                logging.warning(
                    f"[{process_name}] Capture queue is full. Dropping frame from camera {camera.ip_address}"
                )
        else:
            logging.info(f"[{process_name}] Not running prediction at night on camera {camera.ip_address}")

    def run_predictions(self, capture_queue: Queue, prediction_queue: Queue):
        """Run predictions on captured images."""
        process_name = current_process().name
        while True:
            if not capture_queue.empty():
                try:
                    logging.debug(f"[{process_name}] Waiting for frames in capture queue")
                    idx, frame, cam_id, pose_id = capture_queue.get(timeout=5)
                    logging.debug(f"[{process_name}] Received frame from capture queue")
                    preds = self.engine.predict(frame, cam_id)
                    logging.debug(
                        f"[{process_name}] Putting prediction results for camera {cam_id} into prediction queue"
                    )
                    prediction_queue.put((preds, frame, cam_id))
                except Exception as e:
                    logging.exception(f"[{process_name}] Exception during prediction")
            else:
                time.sleep(1)

    def process_alerts(self, prediction_queue: Queue):
        """Process prediction results and send alerts."""
        process_name = current_process().name
        logging.debug(f"[{process_name}] Alert process started")
        while True:
            try:
                if not prediction_queue.empty():
                    try:
                        logging.debug(f"[{process_name}] Waiting for prediction results in prediction queue")
                        preds, frame, cam_id = prediction_queue.get(timeout=5)
                        logging.debug(f"[{process_name}] Processing prediction results for camera {cam_id}")
                        self.engine.process_prediction(preds, frame, cam_id)
                        logging.debug(
                            f"[{process_name}] Prediction queue size after processing: {prediction_queue.qsize()}"
                        )

                        # Process all pending alerts
                        if len(self.engine._alerts) > 0:
                            logging.debug(f"[{process_name}] Processing pending alerts")
                            self.engine._process_alerts()
                    except Empty:
                        logging.debug(f"[{process_name}] Prediction queue is empty, sleeping for 1 second")
                        time.sleep(1)
                    except Exception as e:
                        logging.exception(f"[{process_name}] Exception during alert processing")
                else:
                    logging.debug(f"[{process_name}] Prediction queue is empty, sleeping for 1 second")
                    time.sleep(1)
            except Exception as e:
                logging.exception(f"[{process_name}] Unexpected error in alert process: {e}")
                time.sleep(1)

    def run(
        self,
        capture_interval: int = 30,
        capture_queue_size: int = 10,
        prediction_queue_size: int = 10,
        watchdog_interval: int = 5,
    ):
        """Run the system with separate processes for capturing, predicting, and alerting."""
        capture_queue = Queue(maxsize=capture_queue_size)  # Increased size for the queue for captured frames
        prediction_queue = Queue(maxsize=prediction_queue_size)  # Queue for prediction results

        capture_process = start_process("capture_process", self.capture_images, (capture_queue, capture_interval))
        prediction_process = start_process(
            "prediction_process", self.run_predictions, (capture_queue, prediction_queue)
        )
        alert_process = start_process("alert_process", self.process_alerts, (prediction_queue,))

        processes = [capture_process, prediction_process, alert_process]

        signal.signal(signal.SIGTERM, lambda signum, frame: terminate_processes(processes))
        signal.signal(signal.SIGINT, lambda signum, frame: terminate_processes(processes))

        try:
            # Infinite loop to monitor and restart processes if they stop
            while True:
                if not capture_process.is_alive():
                    logging.warning("Capture process stopped, restarting...")
                    capture_process = start_process(
                        "capture_process", self.capture_images, (capture_queue, capture_interval)
                    )
                if not prediction_process.is_alive():
                    logging.warning("Prediction process stopped, restarting...")
                    prediction_process = start_process(
                        "prediction_process", self.run_predictions, (capture_queue, prediction_queue)
                    )
                if not alert_process.is_alive():
                    logging.warning("Alert process stopped, restarting...")
                    alert_process = start_process("alert_process", self.process_alerts, (prediction_queue,))

                processes = [capture_process, prediction_process, alert_process]

                logging.debug(f"Capture queue size: {capture_queue.qsize()}")
                logging.debug(f"Prediction queue size: {prediction_queue.qsize()}")
                time.sleep(watchdog_interval)  # Interval to check if the process is alive
        except KeyboardInterrupt:
            logging.info("Terminating processes...")
            terminate_processes(processes)

    def __repr__(self) -> str:
        repr_str = f"{self.__class__.__name__}("
        for cam in self.cameras:
            repr_str += f"\n\t{repr(cam)},"
        return repr_str + "\n)"
