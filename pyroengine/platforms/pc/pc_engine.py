# Copyright (C) 2021, Pyronear contributors.

# This program is licensed under the GNU Affero General Public License version 3.
# See LICENSE or go to <https://www.gnu.org/licenses/agpl-3.0.txt> for full license details.

import cv2
import sys
from PIL import Image
from pyroengine.engine import PyronearEngine
import os
from dotenv import load_dotenv


class PcEngine(PyronearEngine):
    """
    This class is the Pyronear Engine for pc. This engine manage the whole Fire Detection
    process by capturing and saving the image and by predicting if there is a fire or
    not based on this image.
    Example
    -------
    # For a prediction every 5 frame
    python pi_engine.py 5
    """
    def run(self, every_n_frame=10):

        cap = cv2.VideoCapture(0)

        frame_nb = 0
        while(True):
            # Capture frame-by-frame
            _, frame = cap.read()
            frame_nb += 1
            if frame_nb % every_n_frame == 0:
                # Predict
                frame = Image.fromarray(frame[:, :, ::-1])  # from BGR numpy to RGB pillow

                self.predict(frame)
                frame_nb = 0


if __name__ == "__main__":
    # API
    load_dotenv()
    api_url = os.getenv('api_url')
    api_login = os.getenv('api_login')
    api_password = os.getenv('api_password')

    sleep_time = int(sys.argv[1])
    # Create Engine
    pi_engine = PcEngine(api_url, api_login, api_password)
    # Run
    pi_engine.run(sleep_time)
