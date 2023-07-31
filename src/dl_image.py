import json
import logging
import multiprocessing as mp
import os
import signal
import time
from io import BytesIO
from itertools import repeat

import requests
from PIL import Image
from requests.auth import HTTPDigestAuth
from tqdm import tqdm

logging.basicConfig(format="%(asctime)s | %(levelname)s: %(message)s", level=logging.INFO, force=True)


def handler():
    raise Exception("Analyze stream timeout")


def get_img(q):
    url = q.get()
    cam_id = url.split("/cgi")[0].split(":")[-1]
    name = os.path.join("data/last_img", cam_id + ".jpg")
    try:
        signal.signal(signal.SIGALRM, handler)
        signal.alarm(30)

        user, password = url.split("usr=")[1].split("&pwd=")
        response = requests.get(url, auth=HTTPDigestAuth(user, password))

        im = Image.open(BytesIO(response.content))

        assert isinstance(im, Image.Image)

        os.makedirs(os.path.dirname(name), exist_ok=True)
        im.save(name, quality=100)
        logging.info(f"Save cam {cam_id}")

        signal.alarm(0)
    except:
        if os.path.isfile(name):
            os.remove(name)
        logging.warning(f"Error {cam_id}")


if __name__ == "__main__":
    with open("data/credentials.json", "rb") as json_file:
        cameras_credentials = json.load(json_file)

    time_dt = {url: time.time() - 30 for url in cameras_credentials.keys()}

    urls = list(cameras_credentials.keys())

    manager = mp.Manager()
    pool = mp.Pool(2)
    q = manager.Queue()

    while True:
        for url, last in time_dt.items():
            dt = time.time() - last

            if dt > 20:
                q.put(url)
                time_dt[url] = time.time()

        pool.apply_async(get_img, (q,))

        time.sleep(1)
