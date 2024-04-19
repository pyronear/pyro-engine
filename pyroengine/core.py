import logging
import signal
import threading
import time
from multiprocessing import Process, Queue
from types import FrameType
from typing import Optional

import urllib3

__all__ = ["SystemController"]

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

logging.basicConfig(format="%(asctime)s | %(levelname)s: %(message)s", level=logging.INFO, force=True)


def handler(signum: int, frame: Optional[FrameType]) -> None:
    raise Exception("Analyze stream timeout")


class SystemController:
    def __init__(self, engine, cameras):
        self.engine = engine
        self.cameras = cameras
        self.prediction_results = Queue()  # Queue for handling results

    def capture_and_predict(self, idx):
        """Capture an image from the camera and perform prediction in a single function."""
        try:
            img = self.cameras[idx].capture()
        except Exception:
            logging.warning(f"Unable to fetch stream from camera {self.cameras[idx]}")
        if img is not None:
            try:
                preds = self.engine.predict(img, self.cameras[idx].ip_address)
                # Send the result along with the image and camera ID for further processing
                self.prediction_results.put((preds, img, self.cameras[idx].ip_address))
            except Exception:
                logging.warning(f"Unable to analyze stream from camera {self.cameras[idx]}")
        else:
            logging.error(f"Failed to capture image from camera {self.cameras[idx].ip_address}")

    def process_results(self):
        """Process results sequentially from the results queue."""
        while not self.prediction_results.empty():
            try:
                preds, frame, cam_id = self.prediction_results.get()
                self.engine.process_prediction(preds, frame, cam_id)
            except Exception:
                logging.warning(f"Unable to process prediction from camera {cam_id}")
        try:
            # Uploading pending alerts
            if len(self.engine._alerts) > 0:
                self.engine._process_alerts()
        except Exception:
            logging.warning(f"Unable to process alerts")

    def run(self, period=30):
        """Create a process for each camera to capture and predict simultaneously."""
        try:
            signal.signal(signal.SIGALRM, handler)
            signal.alarm(int(period))
            processes = []
            for idx in range(len(self.cameras)):
                process = Process(target=self.capture_and_predict, args=(idx,))
                processes.append(process)
                process.start()

            # Process all collected results
            self.process_results()

            signal.alarm(0)
        except Exception:
            logging.warning(f"Analyze stream timeout on {self.cameras[idx]}")

    def __repr__(self) -> str:
        repr_str = f"{self.__class__.__name__}("
        for cam in self.cameras:
            repr_str += f"\n\t{repr(cam)},"
        return repr_str + "\n)"
