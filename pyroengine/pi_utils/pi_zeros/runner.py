# Copyright (C) 2021, Pyronear contributors.

# This program is licensed under the GNU Affero General Public License version 3.
# See LICENSE or go to <https://www.gnu.org/licenses/agpl-3.0.txt> for full license details.

import argparse
import logging
import io
import os
import time
import picamera
import requests
from dotenv import load_dotenv
from requests import RequestException

load_dotenv()

WEBSERVER_IP = os.environ.get("WEBSERVER_IP")
WEBSERVER_PORT = os.environ.get("WEBSERVER_PORT")


class Runner:
    """This class aims at taking picture using PiCamera and sending the stream to local webserver in main raspberry"""

    logger = logging.getLogger(__name__)
    CAPTURE_DELAY = 3

    def __init__(self, webserver_url, loop_interval=30, max_iteration=None):
        """Initialize parameters for Runner."""
        self.camera = picamera.PiCamera()
        self.camera.resolution = (3280, 2464)  # use maximal resolution
        self.webserver_url = webserver_url
        self.loop_interval = loop_interval
        self.max_iteration = max_iteration

    def capture_stream(self):
        stream = io.BytesIO()
        self.camera.start_preview()
        time.sleep(self.CAPTURE_DELAY)  # small sleep here improve image quality
        self.camera.capture(stream, format="jpeg")
        # "Rewind" the stream to the beginning so we can read its content
        stream.seek(0)
        return {"file": stream}

    def send_stream(self, files):
        try:
            response = requests.post(
                self.webserver_url, files=files
            )  # send image to local webserver
            response.raise_for_status()
        except RequestException as e:
            self.logger.error(f"Unexpected error in get_record(): {e!r}")

    def run(self):
        it = 0
        while self.is_running:
            it += 1
            files = self.capture_stream()
            self.send_stream(files)
            if self.max_iteration is not None:
                if it >= self.max_iteration:
                    break

            time.sleep(self.loop_interval)  # Wait between two captures


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Take picture(s) and send to local web server')
    parser.add_argument('--single', action='store_true', help='Single picture instead of eternal loop')
    parser.add_argument('--write', action='store_true', help='Use /write_image route instead of /inference')
    parser.add_argument('--loop_interval', default=30, action='store_true', help='Time between two photos')
    parser.add_argument('--max_iteration', default=None, action='store_true', help='Time between two photos')
    args = parser.parse_args()

    webserver_local_url = f"http://{WEBSERVER_IP}:{WEBSERVER_PORT}" + \
                          ("/inference/file" if not args.write else "/write_image/file")

    runner = Runner(webserver_local_url, loop_interval=args.loop_interval, max_iteration=args.max_iteration)
    if args.single:
        files = runner.capture_stream()
        runner.send_stream(files)
    else:
        runner.run()
