import logging
import time
from multiprocessing import Process, Queue
from typing import Optional

import numpy as np
from PIL import Image


class SystemController:
    def __init__(self, engine, cameras):
        self.engine = engine
        self.cameras = cameras
        self.prediction_results = Queue()  # Queue for handling results

    def capture_and_predict(self, idx):
        """Capture an image from the camera and perform prediction in a single function."""
        img = self.cameras[idx].capture()
        if img is not None:
            preds = self.engine.predict(img, self.cameras[idx].ip_address)
            # Send the result along with the image and camera ID for further processing
            self.prediction_results.put((preds, img, self.cameras[idx].ip_address))
        else:
            logging.error(f"Failed to capture image from camera {self.cameras[idx].ip_address}")

    def process_results(self, start_time):
        """Process results sequentially from the results queue."""
        while not self.prediction_results.empty():
            preds, frame, cam_id = self.prediction_results.get()
            self.engine.process_prediction(preds, frame, cam_id)

        # Uploading pending alerts
        if len(self.engine._alerts) > 0:
            self.engine._process_alerts()

    def run(self):
        """Create a process for each camera to capture and predict simultaneously."""
        start_time = time.time()
        processes = []
        for idx in range(len(self.cameras)):
            process = Process(target=self.capture_and_predict, args=(idx,))
            processes.append(process)
            process.start()

        # Process all collected results
        self.process_results(start_time)

    def __repr__(self) -> str:
        repr_str = f"{self.__class__.__name__}("
        for cam in self.cameras:
            repr_str += f"\n\t{repr(cam)},"
        return repr_str + "\n)"
