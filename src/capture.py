# Copyright (C) 2020-2022, Pyronear.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.


import json
import logging
import os
import time
from io import BytesIO

import requests
import urllib3
from dotenv import load_dotenv
from PIL import Image

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

logging.basicConfig(format="%(asctime)s | %(levelname)s: %(message)s", level=logging.INFO, force=True)


def capture(ip, CAM_USER, CAM_PWD):
    url = f"https://{ip}/cgi-bin/api.cgi?cmd=Snap&channel=0&rs=wuuPhkmUCeI9WG7C&user={CAM_USER}&password={CAM_PWD}"

    response = requests.get(url, verify=False, timeout=3)
    return Image.open(BytesIO(response.content))


def main():
    load_dotenv("/home/pi/pyro-engine/runner/data/.env")

    CAM_USER = os.environ.get("CAM_USER")
    CAM_PWD = os.environ.get("CAM_PWD")

    with open("/home/pi/pyro-engine/runner/data/cameras_credentials.json") as json_file:
        cameras_credentials = json.load(json_file)

    for ip in cameras_credentials.keys():
        try:
            img = capture(ip, CAM_USER, CAM_PWD)
            file = os.path.join("/home/pi/captured_images", ip, f"{time.strftime('%Y%m%d-%H%M%S')}.jpg")
            os.makedirs(os.path.split(file)[0], exist_ok=True)
            img.save(file)

            time.sleep(1)
        except Exception:
            logging.warning(f"Unable to get image from camera {ip}")


if __name__ == "__main__":
    main()
