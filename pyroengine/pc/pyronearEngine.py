# -*- coding: utf-8 -*-

# Copyright (c) Pyronear contributors.
# This file is dual licensed under the terms of the CeCILL-2.1 and AGPLv3 licenses.
# See the LICENSE file in the root of this repository for complete details.

import cv2
from time import sleep
import glob
from PIL import Image
import smtplib
import ssl


class PyronearEngine:
    """This class is the Pyronear Engine. This engine manage the whole Fire Detection
       process by capturing and saving the image and by predicting if there is a fire or
       not based on this image.
    Example
    -------
    pyronearEngine = PyronearEngine("model/pyronearModel.pth")
    pyronearEngine.run(30)  # For a prediction every 30s
    """
    def __init__(self, imgsFolder, checkpointPath):
        # Camera
        self.camera = cv2.VideoCapture(0)

        # Pyronear Predictor
        self.pyronearPredictor = PyronearPredictor(checkpointPath)

    def run_video(self, timeStep):

        cap = cv2.VideoCapture(0)
        frameNb = 0
        while(True):
            # Capture frame-by-frame
            _, frame = cap.read()
            frameNb+=1
            
            if frame.any():
            
                #frame = np.transpose(frame, (1, 0, 2))
                im = Image.fromarray(frame[:,:,::-1])
                
                res = pred(im)

                res = (res*1000)//1/1000
            
                if ii%3 == 0:

                    frame[:logo.shape[0],:logo.shape[1],:]=logo
                    
                    if res < 0.5:
                        cv2.putText(frame, str(res), (95, 22), 0, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
                    else:
                        cv2.putText(frame, str(res), (95, 22), 0, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
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

    def run(self, imagePath):

        self.camera.start_preview()
        sleep(3)  # gives the camera’s sensor time to sense the light levels
        self.camera.capture(imagePath)
        self.camera.stop_preview()

    def call_api(self, score):
        pass


if __name__ == "__main__":

    pyronearEngine = PyronearEngine('DS', "model/pyronearModel.pth")
    pyronearEngine.run(5)  # For a prediction every 5s
