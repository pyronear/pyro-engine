# Copyright (C) 2022-2024, Pyronear.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.


import argparse
import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv

from pyroengine.sensors import ReolinkCamera


def main(args):
    print(args)

    # .env loading
    load_dotenv(".env")
    CAM_USER = os.environ.get("CAM_USER")
    CAM_PWD = os.environ.get("CAM_PWD")
    assert isinstance(CAM_USER, str) and isinstance(CAM_PWD, str)

    # Loading camera creds
    with open(args.creds, "rb") as json_file:
        cameras_credentials = json.load(json_file)

    # Create cache dir
    cache = Path(args.cache, "snapshots")
    cache.mkdir(parents=True, exist_ok=True)

    cams = [ReolinkCamera(_ip, CAM_USER, CAM_PWD) for _ip in cameras_credentials]

    imgs = [cam.capture() for cam in cams]

    files = [cache.joinpath(f"{cam.login}_{time.strftime('%Y%m%d-%H%M%S')}.jpg") for cam, img in zip(cams, imgs)]

    for path, img in zip(files, imgs):
        img.save(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Camera screenshot script", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # Camera & cache
    parser.add_argument("--creds", type=str, default="data/credentials.json", help="Camera credentials")
    parser.add_argument("--cache", type=str, default="./data", help="Cache folder")
    args = parser.parse_args()

    main(args)
