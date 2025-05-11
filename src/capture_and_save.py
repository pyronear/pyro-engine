# Copyright (C) 2022-2025, Pyronear.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

import argparse
import os
import time

import cv2
import numpy as np
from dotenv import load_dotenv

from pyroengine.sensors import ReolinkCamera


def main():
    """
    Script to visit camera positions 20 to 27, capture an image at each position,
    resize to 1280x720, and save locally.
    """
    # Load environment variables
    load_dotenv()
    cam_user = os.getenv("CAM_USER")
    cam_pwd = os.getenv("CAM_PWD")

    # Argument parsing
    parser = argparse.ArgumentParser(
        description="Capture images from positions 20 to 27, resize to 1280x720, and save locally."
    )
    parser.add_argument("--ip", required=True, help="IP address of the Reolink camera")
    parser.add_argument("--username", help="Username for camera access", default=cam_user)
    parser.add_argument("--password", help="Password for camera access", default=cam_pwd)
    parser.add_argument("--protocol", help="Protocol (http or https)", default="http")
    parser.add_argument(
        "--output_folder",
        help="Folder to save captured images",
        default="captured_positions",
    )

    args = parser.parse_args()

    output_folder = f"{args.output_folder}/{args.ip.split('.')[-1]}"
    os.makedirs(output_folder, exist_ok=True)

    # Initialize camera
    camera = ReolinkCamera(
        ip_address=args.ip,
        username=args.username,
        password=args.password,
        protocol=args.protocol,
    )

    try:
        for pose_id in range(20, 28):  # Positions 20 through 27
            print(f"Moving to pose {pose_id} at speed 64...")
            camera.move_camera(operation="ToPos", speed=64, idx=pose_id)
            time.sleep(2)  # Give camera time to move

            print(f"Capturing image at pose {pose_id}...")
            image = camera.capture()
            if image:
                # Convert to OpenCV format
                image_np = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                resized = cv2.resize(image_np, (1280, 720))
                filename = f"pose_{pose_id}.jpg"
                image_path = os.path.join(output_folder, filename)
                cv2.imwrite(image_path, resized)
                print(f"Saved resized image to {image_path}")
            else:
                print(f"Failed to capture image at pose {pose_id}.")

    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
