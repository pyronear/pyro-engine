# Copyright (C) 2022-2025, Pyronear.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

import argparse
import json
import os
import time

import cv2
import numpy as np
from dotenv import load_dotenv

from pyroengine.sensors import ReolinkCamera


def main():
    """
    Script to set up camera positions by moving to specific poses, capturing images,
    and saving them with new pose IDs.

    Mapping:
    - Move to pose [start_pose] and save as pose_id 0
    - Move to pose [start_pose + 1] and save as pose_id 1
    - Move to pose [start_pose + 2] and save as pose_id 2
    - Move to pose [start_pose + 3] and save as pose_id 3

    After saving, if demo mode is enabled, moves to each pose in sequence at speed 64 with a 2-second pause,
    repeating 3 times.
    """
    load_dotenv()
    cam_user = os.getenv("CAM_USER")
    cam_pwd = os.getenv("CAM_PWD")

    parser = argparse.ArgumentParser(
        description="Set up camera positions by saving specific poses to new pose IDs and capturing images."
    )
    parser.add_argument("--creds", default="data/credentials.json", help="Path to camera credentials JSON file")
    parser.add_argument("--username", default=cam_user)
    parser.add_argument("--password", default=cam_pwd)
    parser.add_argument("--protocol", default="http")
    parser.add_argument("--output_folder", default="final_positions")
    parser.add_argument("--demo", action="store_true", help="If set, demo mode is activated")
    parser.add_argument("--ip", type=str, default=None, help="Optional single camera IP to run on")
    parser.add_argument("--start_pose", type=int, default=22, help="Starting pose index (default is 22)")

    args = parser.parse_args()

    with open(args.creds, "r") as f:
        creds = json.load(f)

    selected_creds = {args.ip: creds[args.ip]} if args.ip else creds

    for ip, cam_data in selected_creds.items():
        print(f"\n🔧 Processing camera {ip}")
        focus_position = cam_data.get("focus_position")
        cam_type = cam_data.get("type", "ptz")
        cam_poses = cam_data.get("poses", [])
        cam_azimuths = cam_data.get("azimuths", [cam_data.get("azimuth", 0)])

        output_folder = f"{args.output_folder}/{ip.replace('.', '_')}"
        os.makedirs(output_folder, exist_ok=True)

        camera = ReolinkCamera(
            ip_address=ip,
            username=args.username,
            password=args.password,
            cam_type=cam_type,
            cam_poses=cam_poses,
            cam_azimuths=cam_azimuths,
            protocol=args.protocol,
            focus_position=focus_position,
        )

        # Dynamic pose mapping
        pose_mapping = {args.start_pose + i: i for i in range(4)}

        try:
            for original_pose, new_pose_id in pose_mapping.items():
                print(f"Moving to original pose {original_pose} at speed 64...")
                camera.move_camera(operation="ToPos", speed=64, idx=original_pose)
                time.sleep(2)

                print(f"Capturing image for new pose ID {new_pose_id}...")
                image = camera.capture()
                if image:
                    image_np = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                    image_np = cv2.resize(image_np, (2560, 1440))
                    filename = f"pose_{new_pose_id}.jpg"
                    image_path = os.path.join(output_folder, filename)
                    cv2.imwrite(image_path, image_np)
                    print(f"Image saved at {image_path}")
                else:
                    print(f"Failed to capture image for pose ID {new_pose_id}.")

                print(f"Saving current position as new pose ID {new_pose_id}...")
                camera.set_ptz_preset(idx=new_pose_id)
                time.sleep(1)

            if args.demo:
                print("Running in demo mode...")
                for _ in range(3):
                    for original_pose in pose_mapping:
                        print(f"Moving to pose {original_pose} at speed 64 for demo...")
                        camera.move_camera(operation="ToPos", speed=64, idx=original_pose)
                        time.sleep(2)

        except Exception as e:
            print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
