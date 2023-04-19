# Copyright (C) 2022-2023, Pyronear.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

import logging
import signal
from typing import List

import urllib3

from .engine import Engine
from .sensors import ReolinkCamera

__all__ = ["SystemController"]

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

logging.basicConfig(format="%(asctime)s | %(levelname)s: %(message)s", level=logging.INFO, force=True)


def handler():
    raise Exception("Analyze stream timeout")


class SystemController:
    """Implements the full system controller

    Args:
        engine: the image analyzer
        cameras: the cameras to get the visual streams from
    """

    def __init__(self, engine: Engine, cameras: List[ReolinkCamera]) -> None:
        self.engine = engine
        self.cameras = cameras

    def analyze_stream(self, idx: int) -> None:
        assert 0 <= idx < len(self.cameras)
        try:
            img = self.cameras[idx].capture()
            try:
                self.engine.predict(img, self.cameras[idx].ip_address)
            except Exception:
                logging.warning(f"Unable to analyze stream from camera {self.cameras[idx]}")
        except Exception:
            logging.warning(f"Unable to fetch stream from camera {self.cameras[idx]}")

    def run(self, period=30):
        """Analyzes all camera streams"""
        for idx in range(len(self.cameras)):
            try:
                signal.signal(signal.SIGALRM, handler)
                signal.alarm(int(period / len(self.cameras)))
                self.analyze_stream(idx)
                signal.alarm(0)
            except Exception:
                logging.warning(f"Analyze stream timeout on {self.cameras[idx]}")

    def __repr__(self) -> str:
        repr_str = f"{self.__class__.__name__}("
        for cam in self.cameras:
            repr_str += f"\n\t{repr(cam)},"
        return repr_str + "\n)"
