# Copyright (C) 2020-2022, Pyronear.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.


import json
import logging
import os
import time
from io import BytesIO

import requests
import urllib3
from dotenv import load_dotenv
from PIL import Image

from pyroengine import Engine

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

logging.basicConfig(format="%(asctime)s | %(levelname)s: %(message)s", level=logging.INFO, force=True)


def setup_engine():
    with open("data/config_data.json") as json_file:
        config_data = json.load(json_file)

    # Loading pi zeros datas
    with open("data/cameras_credentials.json") as json_file:
        cameras_credentials = json.load(json_file)

    engine = Engine(
        config_data["hub_repo"],
        config_data["conf_threshold"],
        config_data["api_url"],
        cameras_credentials,
        config_data["latitude"],
        config_data["longitude"],
        frame_saving_period=config_data["save_evry_n_frame"],
    )

    return engine, cameras_credentials, config_data["loop_time"]


def capture(ip, CAM_USER, CAM_PWD):
    url = f"https://{ip}/cgi-bin/api.cgi?cmd=Snap&channel=0&rs=wuuPhkmUCeI9WG7C&user={CAM_USER}&password={CAM_PWD}"

    response = requests.get(url, verify=False, timeout=3)
    return Image.open(BytesIO(response.content))


def main():
    load_dotenv("data/.env")

    CAM_USER = os.environ.get("CAM_USER")
    CAM_PWD = os.environ.get("CAM_PWD")

    engine, cameras_credentials, loop_time = setup_engine()

    while True:
        start_ts = time.time()
        for ip_address in cameras_credentials:
            try:
                img = capture(ip_address, CAM_USER, CAM_PWD)
            except Exception:
                logging.warning(f"Unable to get image from camera {ip_address}")
            try:
                engine.predict(img, ip_address)
            except Exception:
                logging.warning(f"Analysis failed for image from camera {ip_address}")
        # Sleep only once all images are processed
        time.sleep(max(loop_time - time.time() + start_ts, 0))


if __name__ == "__main__":
    main()
