# Copyright (C) 2022-2025, Pyronear.
# Licensed under the Apache License 2.0.

import argparse
import json
import os
import time

import cv2
import numpy as np
from dotenv import load_dotenv

from pyroengine.sensors import ReolinkCamera

# Camera movement parameters
PAN_SPEED_LEVEL = 5
PAN_DEG_PER_SEC = 7.1131
CAM_STOP_TIME = 0.3


def calculate_center_shift_time(fov, overlap, cam_speed_deg_per_sec, cam_stop_time, shift_angle=0):
    effective_angle = fov / 2 - (4 * fov - 3 * overlap - 180) + overlap / 2 + shift_angle

    shift_time = effective_angle / cam_speed_deg_per_sec - cam_stop_time * 2  # higher speed, longer stop
    return shift_time


def calculate_overlap_shift_time(fov, overlap, cam_speed_deg_per_sec, cam_stop_time):
    effective_angle = fov - overlap
    shift_time = effective_angle / cam_speed_deg_per_sec - cam_stop_time
    return shift_time


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


def process_camera(ip, cam_data, args):
    print(f"\n🔧 Processing camera {ip}")
    focus_position = cam_data.get("focus_position")
    cam_type = cam_data.get("type", "ptz")
    cam_poses = cam_data.get("poses", [])
    cam_azimuths = cam_data.get("azimuths", [cam_data.get("azimuth", 0)])

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

    if focus_position is not None:
        print(f"📌 Setting manual focus to {focus_position}")
        camera.set_auto_focus(disable=True)
        time.sleep(1)
        camera.set_manual_focus(position=focus_position)
        time.sleep(2)

    center_shift_time = calculate_center_shift_time(
        args.fov, args.overlap, PAN_DEG_PER_SEC, CAM_STOP_TIME, args.shift_angle
    )
    overlap_shift_time = calculate_overlap_shift_time(args.fov, args.overlap, PAN_DEG_PER_SEC, CAM_STOP_TIME)

    try:
        print("🧭 Moving to position 10 at speed 64.")
        camera.move_camera(operation="ToPos", speed=64, idx=10)
        time.sleep(1)

        print("⬇️ Moving down for 10 seconds at speed 64.")
        camera.move_in_seconds(s=10, operation="Down", speed=64)

        print("⬇️ Moving down for 3 seconds at speed 2.")
        camera.move_in_seconds(s=3, operation="Down", speed=2)

        print(f"➡️ Shifting to center: {center_shift_time:.2f}s at speed {PAN_SPEED_LEVEL}.")
        if center_shift_time > 0:
            camera.move_in_seconds(s=center_shift_time, operation="Right", speed=PAN_SPEED_LEVEL)
        else:
            camera.move_in_seconds(s=-center_shift_time, operation="Left", speed=PAN_SPEED_LEVEL)

        for i in range(8):
            print(f"📸 Capturing image {i + 1}/8")
            image = camera.capture()
            if image:
                image_np = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                if args.draw:
                    image_np = draw_axes_on_image(image_np, args.fov)
                image_np = cv2.resize(image_np, (2560, 1440))
                filename = f"{ip.replace('.', '_')}_im_{i}.jpg"
                actual_folder = os.path.join(args.output_folder, ip.replace(".", "_"))
                os.makedirs(actual_folder, exist_ok=True)
                image_path = os.path.join(actual_folder, filename)
                cv2.imwrite(image_path, image_np)
                print(f"✅ Saved image at {image_path}")
            else:
                print("⚠️ Failed to capture image.")

            ptz_position = 20 + i
            print(f"💾 Registering PTZ position {ptz_position}.")
            camera.set_ptz_preset(idx=ptz_position)

            print(f"➡️ Shifting to next field: {overlap_shift_time:.2f}s at speed {PAN_SPEED_LEVEL}.")
            camera.move_in_seconds(s=overlap_shift_time, operation="Right", speed=PAN_SPEED_LEVEL)
            time.sleep(1)

    except Exception as e:
        print(f"❌ Error with camera {ip}: {e}")


def main():
    load_dotenv()
    default_user = os.getenv("CAM_USER", "admin")
    default_pwd = os.getenv("CAM_PWD", "@Pyronear")

    parser = argparse.ArgumentParser()
    parser.add_argument("--creds", default="data/credentials.json", help="Path to camera credentials JSON file")
    parser.add_argument("--username", default=default_user)
    parser.add_argument("--password", default=default_pwd)
    parser.add_argument("--protocol", default="http")
    parser.add_argument("--output_folder", default="captured_poses")
    parser.add_argument("--fov", type=float, default=54.2)
    parser.add_argument("--overlap", type=float, default=8)
    parser.add_argument("--shift_angle", type=float, default=0)
    parser.add_argument("--draw", type=bool, default=False)
    args = parser.parse_args()

    with open(args.creds, "r") as f:
        creds = json.load(f)

    for ip, cam_data in creds.items():
        process_camera(ip, cam_data, args)


if __name__ == "__main__":
    main()
