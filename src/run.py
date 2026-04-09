# Copyright (C) 2022-2026, Pyronear.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

import argparse
import json
import logging
import os
import pathlib

import urllib3
from dotenv import load_dotenv

from pyroclient import client
from pyroengine import SystemController
from pyroengine.engine import Engine

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

logging.basicConfig(format="%(asctime)s | %(levelname)s: %(message)s", level=logging.INFO, force=True)


def main(args):
    print(args)

    # .env loading
    load_dotenv(".env")
    api_url = os.environ.get("API_URL")
    assert isinstance(api_url, str)
    cam_user = os.environ.get("CAM_USER")
    cam_pwd = os.environ.get("CAM_PWD")
    assert isinstance(cam_user, str)
    assert isinstance(cam_pwd, str)

    # Loading camera creds
    with pathlib.Path(args.creds).open("rb") as json_file:
        camera_data = json.load(json_file)

    # Validate that pose_ids from credentials exist in the API
    for ip, cam_data in camera_data.items():
        api_client = client.Client(cam_data["token"], api_url)
        response = api_client.get_current_poses()
        response.raise_for_status()
        api_pose_ids = {pose["id"] for pose in response.json()}
        local_pose_ids = set(cam_data.get("pose_ids", []))
        missing = local_pose_ids - api_pose_ids
        if missing:
            raise ValueError(f"Camera {ip} ({cam_data.get('name', '')}): pose_ids {missing} not found in API")
        logging.info(f"Camera {ip}: all pose_ids {local_pose_ids} verified in API")

    splitted_cam_creds = {}
    for ip, cam_data in camera_data.items():
        bbox_mask_url = None
        if "bbox_mask_url" in cam_data:
            bbox_mask_url = cam_data["bbox_mask_url"]

        if cam_data["type"] == "ptz":
            cam_poses = cam_data["poses"]
            cam_pose_ids = cam_data["pose_ids"]
            for patrol_id, pose_id in zip(cam_poses, cam_pose_ids, strict=False):
                splitted_cam_creds[ip + "_" + str(patrol_id)] = (cam_data["token"], pose_id, bbox_mask_url)
        else:
            splitted_cam_creds[ip] = cam_data["token"], cam_data["pose_ids"][0], bbox_mask_url

    engine = Engine(
        model_path=args.model_path,
        conf_thresh=args.thresh,
        max_bbox_size=args.max_bbox_size,
        api_url=api_url,
        cam_creds=splitted_cam_creds,
        cache_folder=args.cache,
        backup_size=args.backup_size,
        nb_consecutive_frames=args.nb_consecutive_frames,
        frame_size=args.frame_size,
        cache_backup_period=args.cache_backup_period,
        cache_size=args.cache_size,
        jpeg_quality=args.jpeg_quality,
        day_time_strategy=args.day_time_strategy,
        save_captured_frames=args.save_captured_frames,
        save_detections_frames=args.save_detections_frames,
    )

    sys_controller = SystemController(engine, camera_data, args.pyro_camera_api_url)

    sys_controller.main_loop(args.period, args.send_alerts)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Raspberry Pi system controller",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Model
    parser.add_argument("--model_path", type=str, default=None, help="model path")
    parser.add_argument("--thresh", type=float, default=0.35, help="Confidence threshold")
    parser.add_argument("--max_bbox_size", type=float, default=0.4, help="Maximum bbox size")

    # Camera & cache

    parser.add_argument("--pyro_camera_api_url", type=str, default="http://127.0.0.1:8081", help="Camera api url")
    parser.add_argument("--creds", type=str, default="data/credentials.json", help="Camera credentials")
    parser.add_argument("--cache", type=str, default="./data", help="Cache folder")
    parser.add_argument(
        "--frame-size",
        type=tuple,
        default=(720, 1280),
        help="Resize frame to frame_size before sending it to the api in order to save bandwidth (H, W)",
    )
    parser.add_argument("--jpeg_quality", type=int, default=80, help="Jpeg compression")
    parser.add_argument(
        "--cache-size",
        type=int,
        default=20,
        help="Maximum number of alerts to save in cache",
    )
    parser.add_argument(
        "--nb-consecutive_frames",
        type=int,
        default=7,
        help="Number of consecutive frames to combine for prediction",
    )
    parser.add_argument(
        "--cache_backup_period",
        type=int,
        default=60,
        help="Number of minutes between each cache backup to disk",
    )
    parser.add_argument(
        "--day_time_strategy",
        type=str,
        default="ir",
        help="strategy to define if it's daytime",
    )
    parser.add_argument("--protocol", type=str, default="https", help="Camera protocol")
    # Backup
    parser.add_argument(
        "--backup-size",
        type=int,
        default=10000,
        help="Local backup can't be bigger than 10Go",
    )

    # Debug
    parser.add_argument(
        "--save_captured_frames",
        type=bool,
        default=False,
        help="Save all captured frames locally",
    )
    parser.add_argument(
        "--save_detections_frames",
        type=bool,
        default=False,
        help="Save all locally detection frames locally",
    )
    parser.add_argument(
        "--send_alerts",
        type=bool,
        default=True,
        help="Save all captured frames locally",
    )

    # Time config
    parser.add_argument(
        "--period",
        type=int,
        default=30,
        help="Number of seconds between each camera stream analysis",
    )
    args = parser.parse_args()

    main(args)
