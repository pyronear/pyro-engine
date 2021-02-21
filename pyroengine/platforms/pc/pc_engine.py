# Copyright (C) 2021, Pyronear contributors.

# This program is licensed under the GNU Affero General Public License version 3.
# See LICENSE or go to <https://www.gnu.org/licenses/agpl-3.0.txt> for full license details.

import cv2
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
    # For a prediction every 5 frame
    python pi_engine.py api_login, api_password 5
    """
    def run(self, every_n_frame=10):

        cap = cv2.VideoCapture(0)

        frame_nb = 0
        while(True):
            # Capture frame-by-frame
            _, frame = cap.read()
            if frame_nb % every_n_frame == 0:
                # Predict
                frame = Image.fromarray(frame[:, :, ::-1])  # from BGR numpy to RGB pillow
                self.predict(frame)


if __name__ == "__main__":
    # Get inputs
    api_login = sys.argv[1]
    api_password = sys.argv[2]
    sleep_time = sys.argv[3]
    # Create Engine
    pi_engine = PiEngine(api_login, api_password)
    # Run
    pi_engine.run(sleep_time)
