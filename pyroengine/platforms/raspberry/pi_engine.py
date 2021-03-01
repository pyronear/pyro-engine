# Copyright (C) 2021, Pyronear contributors.

# This program is licensed under the GNU Affero General Public License version 3.
# See LICENSE or go to <https://www.gnu.org/licenses/agpl-3.0.txt> for full license details.

import picamera
import time
import sys
from PIL import Image
from pyroengine.engine import PyronearEngine
import os
from dotenv import load_dotenv


class PiEngine(PyronearEngine):
    """
    This class is the Pyronear Engine for Raspberry pi. This engine manage the whole Fire Detection
    process by capturing and saving the image and by predicting if there is a fire or
    not based on this image.
    Example
    -------
    # For a prediction every 5s
    python pi_engine.py 5
    """
    def run(self, sleep_time=10):

        with picamera.PiCamera() as camera:
            camera.start_preview()
            try:
                for filename in camera.capture_continuous('image{counter:02d}.jpg'):
                    # Predict
                    frame = Image.open(filename)
                    self.predict(frame)
                    time.sleep(sleep_time)
            finally:
                camera.stop_preview()


if __name__ == "__main__":
    # API
    load_dotenv()
    api_url = os.getenv('api_url')
    api_login = os.getenv('api_login')
    api_password = os.getenv('api_password')

    sleep_time = int(sys.argv[1])
    # Create Engine
    pi_engine = PiEngine(api_url, api_login, api_password)
    # Run
    pi_engine.run(sleep_time)
