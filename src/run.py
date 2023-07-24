# Copyright (C) 2022-2023, Pyronear.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.


import argparse
import json
import logging
import os
import time
from pathlib import Path

import urllib3
from dotenv import load_dotenv

from pyroengine import SystemController
from pyroengine.engine import Engine
from pyroengine.sensors import ReolinkCamera

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

logging.basicConfig(format="%(asctime)s | %(levelname)s: %(message)s", level=logging.INFO, force=True)


def main(args):
    print(args)

    # .env loading
    load_dotenv(".env")
    API_URL = os.environ.get("API_URL")
    LAT = float(os.environ.get("LAT"))
    LON = float(os.environ.get("LON"))
    assert isinstance(API_URL, str) and isinstance(LAT, float) and isinstance(LON, float)
    CAM_USER = os.environ.get("CAM_USER")
    CAM_PWD = os.environ.get("CAM_PWD")
    assert isinstance(CAM_USER, str) and isinstance(CAM_PWD, str)

    # Loading camera creds
    with open(args.creds, "rb") as json_file:
        cameras_credentials = json.load(json_file)

    # Check if model is available in cache
    cache = Path(args.cache)

    model_path = cache.joinpath("model.onnx") if args.model_path is None else args.model_path

    engine = Engine(
        model_path,
        args.thresh,
        API_URL,
        cameras_credentials,
        LAT,
        LON,
        frame_saving_period=args.save_period // args.period,
        cache_folder=args.cache,
        backup_size=args.backup_size,
        alert_relaxation=args.alert_relaxation,
        frame_size=args.frame_size,
        cache_backup_period=args.cache_backup_period,
        cache_size=args.cache_size,
        jpeg_quality=args.jpeg_quality,
        day_time_strategy=args.day_time_strategy,
    )

    sys_controller = SystemController(
        engine,
        [ReolinkCamera(_ip, CAM_USER, CAM_PWD) for _ip in cameras_credentials],
    )

    while True:
        start_ts = time.time()
        sys_controller.run(args.period)
        # Sleep only once all images are processed
        time.sleep(max(args.period - time.time() + start_ts, 0))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Raspberry Pi system controller", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # Model
    parser.add_argument("--model_path", type=str, default="data/model.onnx", help="model path")
    parser.add_argument("--thresh", type=float, default=0.25, help="Confidence threshold")
    # Camera & cache
    parser.add_argument("--creds", type=str, default="data/credentials.json", help="Camera credentials")
    parser.add_argument("--cache", type=str, default="./data", help="Cache folder")
    parser.add_argument(
        "--frame-size",
        type=tuple,
        default=(720, 1280),
        help="Resize frame to frame_size before sending it to the api in order to save bandwidth (H, W)",
    )
    parser.add_argument("--jpeg_quality", type=int, default=80, help="Jpeg compression")
    parser.add_argument("--cache-size", type=int, default=20, help="Maximum number of alerts to save in cache")
    parser.add_argument(
        "--alert_relaxation",
        type=int,
        default=3,
        help="Number of consecutive positive detections required to send the first alert",
    )
    parser.add_argument(
        "--cache_backup_period", type=int, default=60, help="Number of minutes between each cache backup to disk"
    )
    parser.add_argument("--day_time_strategy", type=str, default="ir", help="strategy to define if it's daytime")
    # Backup
    parser.add_argument("--backup-size", type=int, default=10000, help="Local backup can't be bigger than 10Go")

    # Time config
    parser.add_argument("--period", type=int, default=30, help="Number of seconds between each camera stream analysis")
    parser.add_argument("--save-period", type=int, default=3600, help="Number of seconds between each media save")
    args = parser.parse_args()

    main(args)
