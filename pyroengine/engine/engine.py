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

    Args:
        detection_threshold (float): wildfire detection threshold in [0, 1]
        api_url (str): url of the pyronear API
        pi_zero_credentials (Dict): api credectials for each pizero, the dictionary should as the one in the example
        save_every_n_frame (int): Send one frame over N to the api for our dataset
        latitude (float): device latitude
        longitude (float): device longitude

    Examples:
        >>> pi_zero_credentials ={}
        >>> pi_zero_credentials['pi_zero_id_1']={'login':'log1', 'password':'pwd1'}
        >>> pi_zero_credentials['pi_zero_id_2']={'login':'log2', 'password':'pwd2'}
        >>> pyroEngine = PyronearEngine(0.6, 'https://api.pyronear.org', pi_zero_credentials, 50)
        >>> pyroEngine.run()
    """
    def __init__(
        self,
        detection_threshold=0.5,
        api_url=None,
        pi_zero_credentials=None,
        save_evry_n_frame=None,
        latitude=None,
        longitude=None
    ):
        """Init engine"""
        # Engine Setup
        self.pyronearPredictor = PyronearPredictor()
        self.detection_threshold = detection_threshold
        self.detection_counter = {}
        self.event_appening = {}
        self.frames_counter = {}
        self.save_evry_n_frame = save_evry_n_frame
        if pi_zero_credentials is not None:
            for pi_zero_id in pi_zero_credentials.keys():
                self.detection_counter[pi_zero_id] = 0
                self.event_appening[pi_zero_id] = False
                self.frames_counter[pi_zero_id] = 0
        else:
            self.detection_counter['-1'] = 0
            self.event_appening['-1'] = False

        # API Setup
        self.use_api = False
        self.api_url = api_url
        self.latitude = latitude
        self.longitude = longitude
        if self.api_url is not None:
            self.use_api = True
            self.init_api(pi_zero_credentials)
        self.stream = io.BytesIO()

    def predict(self, frame, pi_zero_id=None):
        """ run prediction on comming frame"""
        res = self.pyronearPredictor.predict(frame.convert('RGB'))  # run prediction
        if pi_zero_id is None:
            print(f"Wildfire detection score ({res:.2%})")
        else:
            print(f"Wildfire detection score ({res:.2%}), on device {pi_zero_id}")

        if res > self.detection_threshold:
            if pi_zero_id is None:
                pi_zero_id = '-1'  # add default key value

            if not self.event_appening[pi_zero_id]:
                self.detection_counter[pi_zero_id] += 1
                # Ensure counter max value is 3
                if self.detection_counter[pi_zero_id] > 3:
                    self.detection_counter[pi_zero_id] = 3

            # If counter reach 3, start sending alerts
            if self.detection_counter[pi_zero_id] == 3:
                self.event_appening[pi_zero_id] = True

            if self.use_api and self.event_appening[pi_zero_id]:
                frame.save(self.stream, format='JPEG')
                # Send alert to the api
                self.send_alert(pi_zero_id)
                self.stream.seek(0)  # "Rewind" the stream to the beginning so we can read its content

        else:
            if self.detection_counter[pi_zero_id] > 0:
                self.detection_counter[pi_zero_id] -= 1

            if self.detection_counter[pi_zero_id] == 0 and self.event_appening[pi_zero_id]:
                # Stop event
                self.event_appening[pi_zero_id] = False

        # save frame
        if self.use_api and self.save_evry_n_frame:
            self.frames_counter[pi_zero_id] += 1
            if self.frames_counter[pi_zero_id] == self.save_evry_n_frame:
                # Reset frame counter
                self.frames_counter[pi_zero_id] = 0
                # Send frame to the api
                frame.save(self.stream, format='JPEG')
                self.save_frame(pi_zero_id)
                self.stream.seek(0)  # "Rewind" the stream to the beginning so we can read its content

        return res

    def init_api(self, pi_zero_credentials):
        """Setup api"""
        self.api_client = {}
        for pi_zero_id in pi_zero_credentials.keys():
            self.api_client[pi_zero_id] = client.Client(self.api_url, pi_zero_credentials[pi_zero_id]['login'],
                                                        pi_zero_credentials[pi_zero_id]['password'])

    def send_alert(self, pi_zero_id):
        """Send alert"""
        print("Send alert !")
        # Create a media
        media_id = self.api_client[pi_zero_id].create_media_from_device().json()["id"]
        # Create an alert linked to the media and the event
        self.api_client[pi_zero_id].send_alert_from_device(lat=self.latitude, lon=self.longitude, media_id=media_id)
        self.api_client[pi_zero_id].upload_media(media_id=media_id, media_data=self.stream.getvalue())

    def save_frame(self, pi_zero_id):
        """Save frame"""
        print("Upload media for dataset")
        # Create a media
        media_id = self.api_client[pi_zero_id].create_media_from_device().json()["id"]
        # Send media
        self.api_client[pi_zero_id].upload_media(media_id=media_id, media_data=self.stream.getvalue())
