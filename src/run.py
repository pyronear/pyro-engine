# Copyright (C) 2022-2024, Pyronear.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

import argparse
import asyncio
import json
import logging
import os

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

    splitted_cam_creds = {}
    cameras = []
    for _ip, cam_data in cameras_credentials.items():
        cam_poses = []
        for creds in cam_data["credentials"]:
            if cam_data["type"] == "ptz":
                splitted_cam_creds[_ip + "_" + str(creds["posid"])] = creds
                cam_poses.append(creds["posid"])
            else:
                splitted_cam_creds[_ip] = creds

        cameras.append(ReolinkCamera(_ip, CAM_USER, CAM_PWD, cam_data["type"], cam_poses, args.protocol))

    engine = Engine(
        args.model_path,
        args.thresh,
        args.max_bbox_size,
        API_URL,
        splitted_cam_creds,
        LAT,
        LON,
        cache_folder=args.cache,
        backup_size=args.backup_size,
        nb_consecutive_frames=args.nb_consecutive_frames,
        frame_size=args.frame_size,
        cache_backup_period=args.cache_backup_period,
        cache_size=args.cache_size,
        jpeg_quality=args.jpeg_quality,
        day_time_strategy=args.day_time_strategy,
        save_captured_frames=args.save_captured_frames,
    )

    sys_controller = SystemController(
        engine,
        cameras,
    )

    asyncio.run(sys_controller.main_loop(args.period, args.send_alerts))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Raspberry Pi system controller", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # Model
    parser.add_argument("--model_path", type=str, default=None, help="model path")
    parser.add_argument("--thresh", type=float, default=0.15, help="Confidence threshold")
    parser.add_argument("--max_bbox_size", type=float, default=0.4, help="Maximum bbox size")

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
        "--nb-consecutive_frames",
        type=int,
        default=4,
        help="Number of consecutive frames to combine for prediction",
    )
    parser.add_argument(
        "--cache_backup_period", type=int, default=60, help="Number of minutes between each cache backup to disk"
    )
    parser.add_argument("--day_time_strategy", type=str, default="ir", help="strategy to define if it's daytime")
    parser.add_argument("--protocol", type=str, default="https", help="Camera protocol")
    # Backup
    parser.add_argument("--backup-size", type=int, default=10000, help="Local backup can't be bigger than 10Go")

    # Debug
    parser.add_argument("--save_captured_frames", type=bool, default=False, help="Save all captured frames locally")
    parser.add_argument("--send_alerts", type=bool, default=True, help="Save all captured frames locally")

    # Time config
    parser.add_argument("--period", type=int, default=30, help="Number of seconds between each camera stream analysis")
    args = parser.parse_args()

    main(args)
