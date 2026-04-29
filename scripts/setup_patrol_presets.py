# Copyright (C) 2022-2026, Pyronear.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

"""
Setup patrol presets on a PTZ camera via the pyro_camera_api client.

Sequence:
  1. Go to pose 10
  2. Move left 70°  -> save as preset 0
  3. Move right 45° -> save as preset 1
  4. Move right 45° -> save as preset 2
  5. Move right 45° -> save as preset 3
"""

from __future__ import annotations

import argparse
import time

from pyro_camera_api_client import PyroCameraAPIClient

START_POSE = 10
LEFT_DEG = 70.0
STEP_DEG = 45.0


def wait(seconds: float) -> None:
    time.sleep(seconds)


def main() -> None:
    parser = argparse.ArgumentParser(description="Setup patrol presets on a PTZ camera.")
    parser.add_argument("--cam-ip", required=True, help="Camera IP, e.g. 192.168.1.11")
    parser.add_argument("--login", default="admin", help="Camera login (handled server-side, kept for reference)")
    parser.add_argument("--pwd", default="@Pyronear", help="Camera password (handled server-side, kept for reference)")
    parser.add_argument("--base-url", default="http://192.168.1.99:8080", help="Camera API base URL")
    parser.add_argument("--settle", type=float, default=4.0, help="Seconds to wait after each move before saving")
    args = parser.parse_args()

    # Note: --login and --pwd are accepted for parity with operator notes,
    # but the camera API service uses its own server-side credentials.
    _ = (args.login, args.pwd)

    client = PyroCameraAPIClient(base_url=args.base_url, timeout=30.0)

    print(f"[1/5] Moving to pose {START_POSE}")
    client.move_camera(camera_ip=args.cam_ip, pose_id=START_POSE)
    wait(args.settle + 4.0)

    print(f"[2/5] Moving left {LEFT_DEG}°, saving as preset 0")
    client.move_camera(camera_ip=args.cam_ip, direction="Left", degrees=LEFT_DEG)
    wait(args.settle)
    client.set_preset(camera_ip=args.cam_ip, idx=0)
    print("       preset 0 saved")

    for preset_id in (1, 2, 3):
        print(f"[{preset_id + 2}/5] Moving right {STEP_DEG}°, saving as preset {preset_id}")
        client.move_camera(camera_ip=args.cam_ip, direction="Right", degrees=STEP_DEG)
        wait(args.settle)
        client.set_preset(camera_ip=args.cam_ip, idx=preset_id)
        print(f"       preset {preset_id} saved")

    print("Done. Listing presets:")
    print(client.list_presets(camera_ip=args.cam_ip))


if __name__ == "__main__":
    main()
