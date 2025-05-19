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

# Calibrated pan speed level 5 (from measurements)
PAN_SPEED_LEVEL = 5
PAN_DEG_PER_SEC = 7.1131  # average from your table
CAM_STOP_TIME = 0.5  # seconds


def calculate_center_shift_time(fov, overlap, cam_speed_deg_per_sec, cam_stop_time, shift_angle=0, latency=0.3):
    """
    Calculates the time needed to center the camera based on FOV layout.
    """
    effective_angle = fov / 2 - (4 * fov - 3 * overlap - 180 + shift_angle) + overlap / 2
    shift_time = effective_angle / cam_speed_deg_per_sec - cam_stop_time
    return shift_time - latency


def calculate_overlap_shift_time(fov, overlap, cam_speed_deg_per_sec, cam_stop_time, latency=0.3):
    """
    Calculates the time needed to shift from one view to the next.
    """
    effective_angle = fov - overlap
    shift_time = effective_angle / cam_speed_deg_per_sec - cam_stop_time
    return shift_time - latency


def draw_axes_on_image(image, fov):
    height, width, _ = image.shape
    line_y_top = height // 3
    line_y_bottom = 2 * height // 3

    cv2.line(image, (0, line_y_top), (width, line_y_top), (255, 255, 255), 3)
    cv2.line(image, (0, line_y_bottom), (width, line_y_bottom), (255, 255, 255), 3)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.5
    font_thickness = 2
    text_color = (255, 0, 255)

    cv2.putText(image, "Right Rotation", (10, line_y_top - 40), font, font_scale, text_color, font_thickness)
    cv2.putText(image, "Left Rotation", (10, line_y_bottom + 60), font, font_scale, text_color, font_thickness)

    num_graduations = 10
    for i in range(num_graduations + 1):
        x_pos = int(i * width / num_graduations)
        angle_right = i * (fov / num_graduations)
        cv2.line(image, (x_pos, line_y_top - 20), (x_pos, line_y_top + 20), (255, 255, 255), 2)
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
        )

    for i in range(num_graduations + 1):
        x_pos = int(i * width / num_graduations)
        angle_left = -fov + i * (fov / num_graduations)
        cv2.line(image, (x_pos, line_y_bottom - 20), (x_pos, line_y_bottom + 20), (255, 255, 255), 2)
        text = f"{angle_left:.1f}"
        (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, font_thickness)
        cv2.putText(
            image, text, (x_pos - text_width // 2, line_y_bottom - 30), font, font_scale, text_color, font_thickness
        )

    return image


def main():
    load_dotenv()
    cam_user = os.getenv("CAM_USER")
    cam_pwd = os.getenv("CAM_PWD")

    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", required=True, help="Reolink camera IP address")
    parser.add_argument("--username", default=cam_user)
    parser.add_argument("--password", default=cam_pwd)
    parser.add_argument("--protocol", default="http")
    parser.add_argument("--output_folder", default="captured_images")
    parser.add_argument("--fov", type=float, default=54.2)
    parser.add_argument("--overlap", type=float, default=6)
    parser.add_argument("--shift_angle", type=float, default=0)
    parser.add_argument("--draw", type=bool, default=False)
    args = parser.parse_args()

    center_shift_time = calculate_center_shift_time(
        args.fov, args.overlap, PAN_DEG_PER_SEC, CAM_STOP_TIME, args.shift_angle
    )
    overlap_shift_time = calculate_overlap_shift_time(args.fov, args.overlap, PAN_DEG_PER_SEC, CAM_STOP_TIME)

    os.makedirs(args.output_folder, exist_ok=True)
    camera = ReolinkCamera(ip_address=args.ip, username=args.username, password=args.password, protocol=args.protocol)

    try:
        print("Moving to position 10 at speed 64.")
        camera.move_camera(operation="ToPos", speed=64, idx=10)
        time.sleep(1)

        print("Moving down for 10 seconds at speed 64.")
        camera.move_in_seconds(s=10, operation="Down", speed=64)

        print("Moving down for 3 seconds at speed 2.")
        camera.move_in_seconds(s=3, operation="Down", speed=2)

        print(f"Shifting to center: {center_shift_time:.2f}s at speed {PAN_SPEED_LEVEL}.")
        camera.move_in_seconds(s=center_shift_time, operation="Right", speed=PAN_SPEED_LEVEL)

        for i in range(8):
            print(f"Loop {i + 1}/8: Capturing image.")
            image = camera.capture()
            if image:
                image_np = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                if args.draw:
                    image_np = draw_axes_on_image(image_np, args.fov)
                filename = f"im_{args.ip.split('.')[-1]}_{i}.jpg"
                image_path = os.path.join(args.output_folder, filename)
                image_np = cv2.resize(image_np, (1280, 720))
                cv2.imwrite(image_path, image_np)
                print(f"Image saved at {image_path}")
            else:
                print("Failed to capture image.")

            ptz_position = 20 + i
            print(f"Registering PTZ position {ptz_position}.")
            camera.set_ptz_preset(idx=ptz_position)

            print(f"Shifting to next field: {overlap_shift_time:.2f}s at speed {PAN_SPEED_LEVEL}.")
            camera.move_in_seconds(s=overlap_shift_time, operation="Right", speed=PAN_SPEED_LEVEL)
            time.sleep(1)

    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
