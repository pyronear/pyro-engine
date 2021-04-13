# Copyright (C) 2021, Pyronear contributors.

# This program is licensed under the GNU Affero General Public License version 3.
# See LICENSE or go to <https://www.gnu.org/licenses/agpl-3.0.txt> for full license details.

from .predictor import PyronearPredictor
from pyroclient import client
import io


class PyronearEngine:
    """
    This class is the Pyronear Engine. This engine manage the whole Fire Detection
    process by capturing and saving the image and by predicting if there is a fire or
    not based on this image.
    Examples:
        >>> pyroEngine = PyronearEngine(api_url, api_login, api_password)
        >>> pyroEngine.run()
    """
    def __init__(self, api_url=None, api_login=None, api_password=None):
        """Init engine"""
        # Pyronear Predictor
        self.pyronearPredictor = PyronearPredictor()

        # API Setup
        self.use_api = False
        self.api_url = api_url
        if self.api_url is not None:
            self.use_api = True
            self.init_api(api_login, api_password)
        self.image = io.BytesIO()

    def predict(self, frame):
        """ run prediction oncomming frame"""
        res = self.pyronearPredictor.predict(frame.convert('RGB'))
        if res > 0.5:
            print(f"Wildfire detection ({res:.2%})")
            if self.use_api:
                frame.save(self.image, format='JPEG')
                # Send alert to api
                self.send_alert()

        return res

    def init_api(self, api_login, api_password):
        """Setup api"""
        self.api_client = client.Client(self.api_url, api_login, api_password)
        self.event_id = self.api_client.create_event(lat=9, lon=9).json()["id"]

    def send_alert(self):
        """Send Alert"""
        # Create a media
        media_id = self.api_client.create_media_from_device().json()["id"]
        # Create an alert linked to the media and the event
        self.api_client.send_alert_from_device(lat=9, lon=9, event_id=self.event_id, media_id=media_id)
        self.api_client.upload_media(media_id=media_id, image_data=self.image.getvalue())
