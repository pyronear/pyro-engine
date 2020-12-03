# -*- coding: utf-8 -*-

# Copyright (c) Pyronear contributors.
# This file is dual licensed under the terms of the CeCILL-2.1 and AGPLv3 licenses.
# See the LICENSE file in the root of this repository for complete details.

import cv2
from PIL import Image
import numpy as np
from pyroengine.inference.pyronearPredict import PyronearPredictor


class PyronearEngine:
    """The Pyronear Engine manage the whole Fire Detection process by capturing and saving the image,
       but also by predicting if there is a fire or not based on this image.

    Example
    -------
    pyroEngine = PyronearEngine(configFile, checkpointFile)
    pyroEngine.run_video()

    """

    def __init__(self, config, checkpoint):
        """Init Pyronear Engine."""
        # Pyronear Predictor
        self.pyronearPredictor = PyronearPredictor(config, checkpoint)

    def run_video(self, video=None, logo=None):
        """
        Perform inference on video file.

        Args:
        ----
            video (str, optional): Path to a video. Use pc camera by default.
            logo (str, optional): Path to pyronear logo in imgs folder.

        """
        if logo:
            logo = cv2.imread(logo)
            logo = cv2.resize(logo, (150, 30))
        else:
            logo = 255 * np.ones((30, 150, 3))
            logo = cv2.putText(logo, str('PYRONEAR'), (5, 22), 0, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        if video:
            cap = cv2.VideoCapture(video)
        else:
            cap = cv2.VideoCapture(0)
        frameNb = 0
        while(True):
            # Capture frame-by-frame
            _, frame = cap.read()
            frameNb += 1

            if frame.any():
                im = Image.fromarray(frame[:, :, ::-1])
                res = self.pyronearPredictor.predict(im)

                if frameNb % 3 == 0:
                    frame[:logo.shape[0], :logo.shape[1], :] = logo

                    if res < 0.5:
                        cv2.putText(frame, "%.3f" % res, (95, 22), 0, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
                    else:
                        cv2.putText(frame, "%.3f" % res, (95, 22), 0, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                        cv2.imwrite('fire.jpg', frame)
                        # Send alert to api
                        self.call_api(res)
                        break

                    cv2.imshow('frame', frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

        # When everything done, release the capture
        cap.release()
        cv2.destroyAllWindows()

        cv2.imshow('frame', frame)

        while True:
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    def call_api(self, score):
        """Send alert to the api."""
        pass


if __name__ == "__main__":

    checkpoint = "pyro_checkpoint_V0.1.pth"
    config = "inference.cfg"
    pyroEngine = PyronearEngine(config, checkpoint)
    pyroEngine.run_video()
