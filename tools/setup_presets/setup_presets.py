# Copyright (C) 2022-2026, Pyronear.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.


import argparse
import time
from typing import Any, Dict, List

from pyro_camera_api_client import PyroCameraAPIClient

PAN_STEP_DEG = 45.0
REFERENCE_PRESET = 10
GOTO_SETTLE_SECONDS = 6.0


def _camera_entries(api: PyroCameraAPIClient) -> List[Dict[str, Any]]:
    payload = api.get_camera_infos()
    if isinstance(payload, dict):
        return list(payload.get("cameras", []))
    return list(payload)


def _has_reference_preset(api: PyroCameraAPIClient, ip: str) -> bool:
    payload = api.list_presets(ip)
    presets = payload.get("presets") or []
    for preset in presets:
        if preset.get("id") == REFERENCE_PRESET and preset.get("enable", 0) == 1:
            return True
    return False


def process_camera(ip: str, api: PyroCameraAPIClient) -> None:
    print(f"\n🔧 Processing camera {ip}")

    if not _has_reference_preset(api, ip):
        print(f"⚠️  Camera {ip}: reference preset {REFERENCE_PRESET} not configured — skipping")
        return

    print(f"🧭 Going to reference preset {REFERENCE_PRESET}")
    api.goto_preset(camera_ip=ip, pose_id=REFERENCE_PRESET)
    time.sleep(GOTO_SETTLE_SECONDS)

    print(f"⬅️  Panning {2 * PAN_STEP_DEG:.0f}° left to the leftmost position")
    api.move_by_degrees(camera_ip=ip, direction="Left", degrees=2 * PAN_STEP_DEG)

    for preset_idx in (0, 1, 2, 3):
        if preset_idx > 0:
            print(f"➡️  Panning {PAN_STEP_DEG:.0f}° right")
            api.move_by_degrees(camera_ip=ip, direction="Right", degrees=PAN_STEP_DEG)

        print(f"💾 Registering PTZ preset {preset_idx}")
        api.set_preset(camera_ip=ip, idx=preset_idx)


def main() -> None:
    parser = argparse.ArgumentParser(description="Register PTZ presets 0-3 around reference preset 10.")
    parser.add_argument("--api_url", default="http://127.0.0.1:8081", help="Pyro camera API base URL")
    parser.add_argument(
        "--ip",
        type=str,
        default=None,
        help="Optional IP to run on a single camera (e.g. 192.168.1.11). If omitted, runs on every PTZ camera.",
    )
    args = parser.parse_args()

    api = PyroCameraAPIClient(args.api_url, timeout=30.0)

    if args.ip:
        process_camera(args.ip, api)
        return

    entries = _camera_entries(api)
    if not entries:
        print("⚠️  No cameras returned by the API")
        return

    print(f"🔎 Found {len(entries)} camera(s)")
    for entry in entries:
        ip = entry.get("camera_id") or entry.get("ip")
        if not ip:
            print(f"⏭️  Skipping entry without camera_id/ip: {entry!r}")
            continue
        cam_type = str(entry.get("type", "")).lower()
        if cam_type != "ptz":
            print(f"⏭️  Skipping {ip}: not a PTZ camera (type={entry.get('type')!r})")
            continue
        process_camera(ip, api)


if __name__ == "__main__":
    main()
