# Copyright (C) 2020-2025, Pyronear.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

import logging
import threading
import time

from anonymizer.anonymizer_registry import set_result
from anonymizer.vision import Anonymizer
from camera.registry import CAMERA_REGISTRY

ANONYMIZER_MODEL = Anonymizer()


def anonymizer_loop(camera_ip: str, stop_flag: threading.Event):
    cam = CAMERA_REGISTRY[camera_ip]
    logging.info(f"[{camera_ip}] Starting anonymizer loop")

    while not stop_flag.is_set():
        t0 = time.time()
        try:
            frame = cam.capture()
            preds = ANONYMIZER_MODEL(frame)  # list of dicts with cls, score, box
            set_result(camera_ip, preds)
        except Exception as e:
            logging.error(f"[{camera_ip}] Anonymizer loop error: {e}")

    logging.info(f"[{camera_ip}] Anonymizer loop stopped")
