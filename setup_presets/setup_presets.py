# Copyright (C) 2022-2026, Pyronear.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.


import argparse
import concurrent.futures
import os
import sys

from dotenv import load_dotenv

from reolink import ReolinkClient

PAN_STEP_DEG = 45.0


def process_camera(ip: str, user: str, password: str, protocol: str) -> None:
    def log(msg: str) -> None:
        print(f"[{ip}] {msg}", flush=True)

    log("🔧 starting from current position")
    cam = ReolinkClient(ip, user, password, protocol=protocol)

    log(f"⬅️  Panning {2 * PAN_STEP_DEG:.0f}° left")
    if not cam.pan_degrees("Left", 2 * PAN_STEP_DEG):
        log("❌ left pan failed — aborting")
        return

    for preset_idx in (0, 1, 2, 3):
        if preset_idx > 0:
            log(f"➡️  Panning {PAN_STEP_DEG:.0f}° right")
            if not cam.pan_degrees("Right", PAN_STEP_DEG):
                log(f"❌ right pan failed — aborting (preset {preset_idx} not saved)")
                return
        log(f"💾 Registering PTZ preset {preset_idx}")
        if not cam.set_preset(preset_idx):
            log(f"❌ failed to save preset {preset_idx} — aborting")
            return
    log("✅ done")


def main() -> None:
    load_dotenv()
    user = os.getenv("CAM_USER")
    password = os.getenv("CAM_PWD")
    if not user or not password:
        print("❌ Missing CAM_USER or CAM_PWD (set them in your .env)")
        sys.exit(1)

    parser = argparse.ArgumentParser(
        description="Register PTZ presets 0-3 starting from each camera's current position."
    )
    parser.add_argument("ips", nargs="+", help="One or more camera IPs, e.g. 192.168.1.11 192.168.1.12")
    parser.add_argument("--protocol", default="https", choices=["http", "https"])
    args = parser.parse_args()

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(args.ips)) as executor:
        futures = [executor.submit(process_camera, ip, user, password, args.protocol) for ip in args.ips]
        for future in concurrent.futures.as_completed(futures):
            future.result()


if __name__ == "__main__":
    main()
