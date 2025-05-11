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


def calculate_shift_time(fov, overlap, cam_speed_1, cam_stop_time, shift_angle=0):
    """
    Calculates the shift time based on FOV, overlap, camera speed, stop time, and shift angle.
    """
    shift_time = (fov / 2 - (4 * fov - 3 * overlap - 180 + shift_angle) + overlap / 2) / cam_speed_1 - cam_stop_time
    return shift_time


def draw_axes_on_image(image, fov):
    """Draws central axes and graduation marks for left and right rotation on the image."""
    height, width, _ = image.shape

    # Define line positions centrally
    line_y_top = height // 3  # Upper third of the image
    line_y_bottom = 2 * height // 3  # Lower third of the image

    # Draw main lines
    cv2.line(image, (0, line_y_top), (width, line_y_top), (255, 255, 255), 3)
    cv2.line(image, (0, line_y_bottom), (width, line_y_bottom), (255, 255, 255), 3)

    # Font settings for text
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.5
    font_thickness = 2
    text_color = (255, 0, 255)

    # Add legends
    cv2.putText(
        image,
        "Right Rotation",
        (10, line_y_top - 40),
        font,
        font_scale,
        text_color,
        font_thickness,
        lineType=cv2.LINE_AA,
    )
    cv2.putText(
        image,
        "Left Rotation",
        (10, line_y_bottom + 60),
        font,
        font_scale,
        text_color,
        font_thickness,
        lineType=cv2.LINE_AA,
    )

    # Draw graduation marks for the top line (0 to fov for right rotation)
    num_graduations = 10
    for i in range(num_graduations + 1):
        x_pos = int(i * width / num_graduations)
        angle_right = i * (fov / num_graduations)

        # Graduation mark
        cv2.line(
            image,
            (x_pos, line_y_top - 20),
            (x_pos, line_y_top + 20),
            (255, 255, 255),
            2,
        )

        # Label
        text = f"{angle_right:.1f}"
        (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, font_thickness)
        cv2.putText(
            image,
            text,
            (x_pos - text_width // 2, line_y_top + 40 + text_height),
            font,
            font_scale,
            text_color,
            font_thickness,
            lineType=cv2.LINE_AA,
        )

    # Draw graduation marks for the bottom line (from -fov to 0 for left rotation)
    for i in range(num_graduations + 1):
        x_pos = int(i * width / num_graduations)
        angle_left = -fov + i * (fov / num_graduations)

        # Graduation mark
        cv2.line(
            image,
            (x_pos, line_y_bottom - 20),
            (x_pos, line_y_bottom + 20),
            (255, 255, 255),
            2,
        )

        # Label
        text = f"{angle_left:.1f}"
        (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, font_thickness)
        cv2.putText(
            image,
            text,
            (x_pos - text_width // 2, line_y_bottom - 30),
            font,
            font_scale,
            text_color,
            font_thickness,
            lineType=cv2.LINE_AA,
        )

    return image


def main():
    """
    Script to control a Reolink camera for specific movements and capture images.
    """
    # Load environment variables
    load_dotenv()
    cam_user = os.getenv("CAM_USER")
    cam_pwd = os.getenv("CAM_PWD")

    # Argument parsing
    parser = argparse.ArgumentParser(
        description="Script to control Reolink camera for specific movements and capture images."
    )
    parser.add_argument("--ip", required=True, help="IP address of the Reolink camera")
    parser.add_argument("--username", help="Username for camera access", default=cam_user)
    parser.add_argument("--password", help="Password for camera access", default=cam_pwd)
    parser.add_argument("--protocol", help="Protocol (http or https)", default="http")
    parser.add_argument(
        "--output_folder",
        help="Folder to save captured images",
        default="captured_images",
    )
    parser.add_argument("--fov", type=float, default=54.2, help="Field of View of the camera")
    parser.add_argument("--overlap", type=float, default=6, help="Overlap angle between positions")
    parser.add_argument("--cam_speed_1", type=float, default=1.4, help="Camera speed for PTZ operation")
    parser.add_argument(
        "--cam_stop_time",
        type=float,
        default=0.5,
        help="Camera stop time after movement",
    )
    parser.add_argument(
        "--move_duration",
        type=float,
        default=34,
        help="Duration in seconds for the rightward move in each loop",
    )
    parser.add_argument(
        "--shift_angle",
        type=float,
        default=0,
        help="Shift angle for time calculation adjustment",
    )

    args = parser.parse_args()

    # Calculate shift time using the provided shift_angle parameter
    shift_time = calculate_shift_time(args.fov, args.overlap, args.cam_speed_1, args.cam_stop_time, args.shift_angle)

    # Create output directory if it doesn't exist
    os.makedirs(args.output_folder, exist_ok=True)

    # Initialize camera
    camera = ReolinkCamera(
        ip_address=args.ip,
        username=args.username,
        password=args.password,
        protocol=args.protocol,
    )

    try:
        # Move to position 10 at speed 64
        print("Moving to position 10 at speed 64.")
        camera.move_camera(operation="ToPos", speed=64, idx=10)
        time.sleep(1)

        # Move down for 10 seconds at speed 64
        print("Moving down for 10 seconds at speed 64.")
        camera.move_in_seconds(s=10, operation="Down", speed=64)

        # Move down for 2 seconds at speed 2
        print("Moving down for 2 seconds at speed 2.")
        camera.move_in_seconds(s=2, operation="Down", speed=2)

        # Move right for calculated shift time at speed 1
        print(f"Moving right for {shift_time:.2f} seconds at speed 1.")
        camera.move_in_seconds(s=shift_time, operation="Right", speed=1)

        # Loop to capture, register PTZ positions, and move right
        for i in range(8):
            # Capture image
            print(f"Loop {i + 1}/8: Capturing image.")
            image = camera.capture()
            if image:
                # Convert the image to an OpenCV-compatible format
                image_np = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

                # Draw axes on the image
                image_np = draw_axes_on_image(image_np, args.fov)

                # Save the modified image
                filename = f"im_{args.ip.split('.')[-1]}_{i}.jpg"
                image_path = os.path.join(args.output_folder, filename)
                image_np = cv2.resize(image_np, (640, 360))
                cv2.imwrite(image_path, image_np)
                print(f"Image saved at {image_path}")
            else:
                print("Failed to capture image.")

            # Register PTZ position from 20 to 27
            ptz_position = 20 + i
            print(f"Registering PTZ position {ptz_position}.")
            camera.set_ptz_preset(idx=ptz_position)

            # Move right for specified duration at speed 1
            print(f"Moving right for {args.move_duration} seconds at speed 1.")
            camera.move_in_seconds(s=args.move_duration, operation="Right", speed=1)

            time.sleep(1)

    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
