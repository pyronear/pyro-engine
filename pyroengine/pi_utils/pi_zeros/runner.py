# Copyright (C) 2020-2022, Pyronear.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

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
    LOOP_INTERVAL = 30

    def __init__(self, webserver_url):
        """Initialize parameters for Runner."""
        self.camera = picamera.PiCamera()
        self.camera.resolution = (3280, 2464)  # use maximal resolution
        self.webserver_url = webserver_url
        self.is_running = True

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
        while self.is_running:
            files = self.capture_stream()
            self.send_stream(files)
            time.sleep(self.LOOP_INTERVAL)  # Wait between two captures

    def stop_runner(self):
        self.is_running = False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Take picture(s) and send to local web server"
    )
    parser.add_argument(
        "--single", action="store_true", help="Single picture instead of eternal loop"
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="Use /write_image route instead of /inference",
    )
    args = parser.parse_args()

    webserver_local_url = f"http://{WEBSERVER_IP}:{WEBSERVER_PORT}" + (
        "/inference/file" if not args.write else "/write_image/file"
    )
    runner = Runner(webserver_local_url)
    if args.single:
        files = runner.capture_stream()
        runner.send_stream(files)
    else:
        runner.run()
