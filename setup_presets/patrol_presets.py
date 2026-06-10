# Copyright (C) 2022-2026, Pyronear.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.


import argparse
import concurrent.futures
import os
import sys
import time

from dotenv import load_dotenv
from reolink import ReolinkClient

PRESETS = (0, 1, 2, 3)
DWELL_SECONDS = 3.0


def patrol_camera(ip: str, user: str, password: str, protocol: str, n_patrols: int) -> None:
    def log(msg: str) -> None:
        print(f"[{ip}] {msg}", flush=True)

    log(f"🚓 starting patrol — {n_patrols} cycle(s)")
    cam = ReolinkClient(ip, user, password, protocol=protocol)

    for cycle in range(1, n_patrols + 1):
        for preset in PRESETS:
            log(f"➡️  cycle {cycle}/{n_patrols} → preset {preset}")
            if not cam.ptz("ToPos", speed=50, idx=preset):
                log(f"❌ failed to go to preset {preset} — aborting")
                return
            time.sleep(DWELL_SECONDS)
    log("✅ done")


def main() -> None:
    load_dotenv()
    user = os.getenv("CAM_USER")
    password = os.getenv("CAM_PWD")
    if not user or not password:
        print("❌ Missing CAM_USER or CAM_PWD (set them in your .env)")
        sys.exit(1)

    parser = argparse.ArgumentParser(
        description=f"Patrol PTZ presets {list(PRESETS)} with {DWELL_SECONDS:.0f}s dwell between poses."
    )
    parser.add_argument("ips", nargs="+", help="One or more camera IPs, e.g. 192.168.1.11 192.168.1.12")
    parser.add_argument("-n", "--n_patrols", type=int, default=3, help="Number of patrol cycles (default 3)")
    parser.add_argument("--protocol", default="https", choices=["http", "https"])
    args = parser.parse_args()

    if args.n_patrols < 1:
        print("❌ -n/--n_patrols must be >= 1")
        sys.exit(1)

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(args.ips)) as executor:
        futures = [executor.submit(patrol_camera, ip, user, password, args.protocol, args.n_patrols) for ip in args.ips]
        for future in concurrent.futures.as_completed(futures):
            future.result()


if __name__ == "__main__":
    main()
