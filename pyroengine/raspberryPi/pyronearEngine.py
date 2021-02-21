# Copyright (C) 2021, Pyronear contributors.

# This program is licensed under the GNU Affero General Public License version 3.
# See LICENSE or go to <https://www.gnu.org/licenses/agpl-3.0.txt> for full license details.

from picamera import PiCamera
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
        self.camera = PiCamera()
        self.camera.rotation = 270

        # Images Folder
        self.imgsFolder = imgsFolder

        # Pyronear Predictor
        self.pyronearPredictor = PyronearPredictor(checkpointPath)

    def run(self, timeStep):

        imgs = glob.glob(self.imgsFolder + "//*.jpg")
        idx = len(imgs)

        while True:
            imagePath = self.imgsFolder + "//" + str(idx).zfill(8) + ".jpg"
            print(imagePath)
            self.capture(imagePath)
            pred = self.predict(imagePath)
            print(pred)
            idx = idx + 1
            sleep(timeStep)

    def capture(self, imagePath):

        self.camera.start_preview()
        sleep(3)  # gives the camera’s sensor time to sense the light levels
        self.camera.capture(imagePath)
        self.camera.stop_preview()

    def predict(self, imagePath):
        im = Image.open(imagePath)
        pred = self.pyronearPredictor(im)

        if pred[0] > 0.5:
            return "no fire"
        else:
            sendAlert()
            return "FIRE !!!"


def sendAlert():
    port = 587  # For starttls
    smtp_server = "smtp.gmail.com"
    sender_email = "test.pyronear@gmail.com"  # Enter your address
    receiver_email = ""  # Enter receiver address
    # password = ""  # uncomment and add your password
    message = """\
    Subject: FIRE

    Pyronear has detected a fire !!!"""

    context = ssl.create_default_context()
    with smtplib.SMTP(smtp_server, port) as server:
        server.ehlo()  # Can be omitted
        server.starttls(context=context)
        server.ehlo()  # Can be omitted
        server.login(sender_email, password)
        server.sendmail(sender_email, receiver_email, message)


if __name__ == "__main__":

    pyronearEngine = PyronearEngine('DS', "model/pyronearModel.pth")
    pyronearEngine.run(5)  # For a prediction every 5s
