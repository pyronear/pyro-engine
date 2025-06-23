#!/usr/bin/env python3
# Copyright (C) 2022-2025, Pyronear.
# Apache 2.0 licence – see LICENCE file for details.

import argparse
import json
import sys
from pathlib import Path

from pyroengine.sensors import ReolinkCamera


# ----------------------------------------------------------------------
# Main loop
# ----------------------------------------------------------------------
def process_all_cameras(
    credentials_path: str = "/home/engine/data/credentials.json",
    save_images: bool = False,
) -> None:
    if not Path(credentials_path).is_file():
        sys.exit(f"Credentials file not found: {credentials_path}")

    with open(credentials_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for ip, cfg in data.items():
        print(f"\n===> Processing camera {ip} ({cfg.get('name', 'Unnamed')})")

        camera = ReolinkCamera(
            ip_address=ip,
            username="admin",  # change if needed
            password="@Pyronear",  # change if needed
            protocol="http",
            cam_type=cfg.get("type", "ptz"),
            cam_poses=cfg.get("poses", []),
            cam_azimuths=cfg.get("azimuths", []),
            focus_position=cfg.get("focus_position"),
        )

        best_focus = camera.focus_finder(save_images=save_images)
        print(f"Final best focus for {ip}: {best_focus}")

        # Optional: write the new value back to the in-memory dict
        cfg["focus_position"] = best_focus

    # Optional: persist the updated focus values
    # with open(credentials_path, "w", encoding="utf-8") as f:
    #     json.dump(data, f, indent=4)
    #     print(f"\nUpdated credentials saved to {credentials_path}")


# ----------------------------------------------------------------------
# CLI entry point
# ----------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run greedy autofocus on every camera listed in credentials.json")
    parser.add_argument(
        "--credentials",
        default="/home/engine/data/credentials.json",
        help="Path to credentials JSON file (default: %(default)s)",
    )
    parser.add_argument(
        "--save_images",
        action="store_true",
        help="Save each captured frame under focus_debug/<ip_address>",
    )
    args = parser.parse_args()

    process_all_cameras(args.credentials, save_images=args.save_images)
