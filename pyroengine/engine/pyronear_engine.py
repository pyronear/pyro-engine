# Copyright (C) 2021, Pyronear contributors.

# This program is licensed under the GNU Affero General Public License version 3.
# See LICENSE or go to <https://www.gnu.org/licenses/agpl-3.0.txt> for full license details.

from .pyronear_predictor import PyronearPredictor
from pyroclient import client


api_url = "http://pyronear-api.herokuapp.com"


class PyronearEngine:
    """
    This class is the Pyronear Engine. This engine manage the whole Fire Detection
    process by capturing and saving the image and by predicting if there is a fire or
    not based on this image.
    Example
    -------
    pyroEngine = PyronearEngine(api_login, api_password)
    pyroEngine.run()
    """
    def __init__(self, api_login, api_password):
        # Pyronear Predictor
        self.pyronearPredictor = PyronearPredictor()

        # API Setup
        self.create_device(api_login, api_password)

    def run(self):
        """Should be implemented for each patform, with an adapted image capture method"""
        pass

    def predict(self, frame):

        res = self.pyronearPredictor.predict(frame)
        if res > 0.5:
            print('Fire detected !!!')
            frame.save('fire.jpg')
            # Send alert to api
            self.send_alert()

    def create_device(self, api_login, api_password):

        self.api_client = client.Client(api_url, api_login, api_password)
        self.event_id = self.api_client.create_event(lat=9, lon=9).json()["id"]

    def send_alert(self):
        # Create a media
        media_id = self.api_client.create_media_from_device().json()["id"]
        # Create an alert linked to the media and the event
        self.api_client.send_alert_from_device(lat=9, lon=9, event_id=self.event_id, media_id=media_id)
        with open('fire.jpg', 'rb') as f:
            bytestring = f.read()
        self.api_client.upload_media(media_id=media_id, image_data=bytestring)
