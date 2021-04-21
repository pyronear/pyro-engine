# Copyright (C) 2021, Pyronear contributors.

# This program is licensed under the GNU Affero General Public License version 3.
# See LICENSE or go to <https://www.gnu.org/licenses/agpl-3.0.txt> for full license details.


import io
import os
import time
import picamera
import requests
from dotenv import load_dotenv

load_dotenv()

WEBSERVER_IP = os.environ.get("WEBSERVER_IP")
WEBSERVER_PORT = os.environ.get("WEBSERVER_PORT")


url = f"http://{WEBSERVER_IP}:{WEBSERVER_PORT}/inference/file"

stream = io.BytesIO()
with picamera.PiCamera() as camera:
    camera.resolution = (3280, 2464)  # use maximal resolution
    while True:
        print("capture")
        stream = io.BytesIO()
        camera.start_preview()
        time.sleep(3)  # small sleep here improve image quality
        camera.capture(stream, format="jpeg")
        # "Rewind" the stream to the beginning so we can read its content
        stream.seek(0)
        files = {"file": stream}
        requests.post(url, files=files)  # send image to pi_cental

        time.sleep(3)  # Wait between two capture
