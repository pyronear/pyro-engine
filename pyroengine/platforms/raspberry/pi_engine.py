# Copyright (C) 2021, Pyronear contributors.

# This program is licensed under the GNU Affero General Public License version 3.
# See LICENSE or go to <https://www.gnu.org/licenses/agpl-3.0.txt> for full license details.

import picamera
import time
import sys
from PIL import Image
from pyroengine.engine import PyronearEngine


class PiEngine(PyronearEngine):
    """This class is the Pyronear Engine. This engine manage the whole Fire Detection
       process by capturing and saving the image and by predicting if there is a fire or
       not based on this image.
    Example
    -------
    # For a prediction every 5s
    python pi_engine.py api_login, api_password 5
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

    api_login = sys.argv[1]
    api_password = sys.argv[2]
    sleep_time = sys.argv[3]
    pi_engine = PiEngine(api_login, api_password)
    pi_engine.run(5)
