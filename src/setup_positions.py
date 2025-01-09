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
    Script to set up camera positions by moving to specific poses, capturing images,
    and saving them with new pose IDs.

    Mapping:
    - Move to pose 22 and save as pose_id 0
    - Move to pose 23 and save as pose_id 1
    - Move to pose 24 and save as pose_id 2
    - Move to pose 25 and save as pose_id 3

    After saving, if demo mode is enabled, moves to each pose in sequence at speed 64 with a 2-second pause,
    repeating 3 times.
    """
    # Load environment variables
    load_dotenv()
    cam_user = os.getenv("CAM_USER")
    cam_pwd = os.getenv("CAM_PWD")

    # Argument parsing
    parser = argparse.ArgumentParser(
        description="Set up camera positions by saving specific poses to new pose IDs and capturing images."
    )
    parser.add_argument("--ip", required=True, help="IP address of the Reolink camera")
    parser.add_argument("--username", help="Username for camera access", default=cam_user)
    parser.add_argument("--password", help="Password for camera access", default=cam_pwd)
    parser.add_argument("--protocol", help="Protocol (http or https)", default="http")
    parser.add_argument("--output_folder", help="Folder to save captured images", default="captured_positions")
    parser.add_argument("--demo", action="store_true", help="If set, demo mode is activated")

    args = parser.parse_args()

    output_folder = f"{args.output_folder}/{args.ip.split('.')[-1]}"

    # Create output directory if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Initialize camera
    camera = ReolinkCamera(ip_address=args.ip, username=args.username, password=args.password, protocol=args.protocol)

    # Mapping of original poses to new pose IDs
    pose_mapping = {22: 0, 23: 1, 24: 2, 25: 3}

    try:
        # First, move to each pose, capture an image, and save it as a new pose ID
        for original_pose, new_pose_id in pose_mapping.items():
            print(f"Moving to original pose {original_pose} at speed 64...")
            camera.move_camera(operation="ToPos", speed=64, idx=original_pose)
            time.sleep(2)  # Allow time for the camera to move

            # Capture an image at the position
            print(f"Capturing image for new pose ID {new_pose_id}...")
            image = camera.capture()
            if image:
                # Convert image to OpenCV-compatible format and save it
                image_np = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                filename = f"pose_{new_pose_id}.jpg"
                image_path = os.path.join(output_folder, filename)
                cv2.imwrite(image_path, image_np)
                print(f"Image saved at {image_path}")
            else:
                print(f"Failed to capture image for pose ID {new_pose_id}.")

            # Save the position as a new pose ID
            print(f"Saving current position as new pose ID {new_pose_id}...")
            camera.set_ptz_preset(idx=new_pose_id)
            print(f"Position saved with new pose ID {new_pose_id}.")
            time.sleep(1)

        # If demo mode is enabled, go through each pose three times with a 2-second pause
        if args.demo:
            print("Running in demo mode...")
            for _ in range(3):  # Loop three times
                for original_pose in pose_mapping.keys():
                    print(f"Moving to pose {original_pose} at speed 64 for demo...")
                    camera.move_camera(operation="ToPos", speed=64, idx=original_pose)
                    time.sleep(2)  # Pause for 2 seconds at each pose

    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
