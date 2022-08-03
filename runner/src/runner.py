# Copyright (C) 2020-2022, Pyronear.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.


from PIL import Image
import requests
from io import BytesIO
from pyroengine.engine import PyronearEngine
from dotenv import load_dotenv
import os
import time
import json
import logging
import urllib3


urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

logging.basicConfig(
    format="%(asctime)s | %(levelname)s: %(message)s", level=logging.INFO, force=True
)


def setup_engine():
    with open("data/config_data.json") as json_file:
        config_data = json.load(json_file)

    # Loading config datas
    detection_threshold = config_data["detection_threshold"]
    api_url = config_data["api_url"]
    save_evry_n_frame = config_data["save_evry_n_frame"]
    loop_time = config_data["loop_time"]
    latitude = config_data["latitude"]
    longitude = config_data["longitude"]
    model_weights = config_data["model_weights"]

    # Loading pi zeros datas
    with open("data/cameras_credentials.json") as json_file:
        cameras_credentials = json.load(json_file)

    engine = PyronearEngine(
        detection_threshold,
        api_url,
        cameras_credentials,
        save_evry_n_frame,
        latitude,
        longitude,
        model_weights=model_weights,
    )

    return engine, cameras_credentials, loop_time


def capture(ip, CAM_USER, CAM_PWD):
    url = f"https://{ip}/cgi-bin/api.cgi?cmd=Snap&channel=0&rs=wuuPhkmUCeI9WG7C&user={CAM_USER}&password={CAM_PWD}"

    response = requests.get(url, verify=False, timeout=3)
    return Image.open(BytesIO(response.content))


load_dotenv("data/.env")

CAM_USER = os.environ.get("CAM_USER")
CAM_PWD = os.environ.get("CAM_PWD")

engine, cameras_credentials, loop_time = setup_engine()

while True:
    for ip in cameras_credentials.keys():
        try:
            start_time = time.time()
            img = capture(ip, CAM_USER, CAM_PWD)
            pred = engine.predict(img, ip)

            time.sleep(max(loop_time - time.time() + start_time, 0))
        except Exception:
            logging.warning(f"Unable to get image from camera {ip}")
