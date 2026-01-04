# Copyright (C) 2022-2026, Pyronear.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.


import argparse
import os
import time

from dotenv import load_dotenv

from pyroengine.sensors import ReolinkCamera

# Load credentials from .env file
load_dotenv()
CAM_USER = os.getenv("CAM_USER", "admin")
CAM_PWD = os.getenv("CAM_PWD", "@Pyronear")
PROTOCOL = "http"

# Parse the camera IP address from CLI arguments
parser = argparse.ArgumentParser(description="Sweep through focus values and capture images.")
parser.add_argument("--ip", required=True, help="IP address of the Reolink camera")
args = parser.parse_args()
CAM_IP = args.ip

# Range of focus values to test
focus_values = list(range(680, 750, 1))

# Output directory for saved images
output_dir = "focus_tests"
os.makedirs(output_dir, exist_ok=True)

# Create the camera instance
cam = ReolinkCamera(
    ip_address=CAM_IP,
    username=CAM_USER,
    password=CAM_PWD,
    protocol=PROTOCOL,
)

# Disable autofocus
cam.set_auto_focus(disable=True)
time.sleep(1)

# Sweep focus values and capture images
for focus in focus_values:
    print(f"🔧 Setting focus to {focus}")
    cam.set_manual_focus(position=focus)
    time.sleep(2)  # Give the camera time to adjust focus
    img = cam.capture()
    if img:
        path = os.path.join(output_dir, f"focus_{focus}.jpg")
        img.resize((1280, 720)).save(path)
        print(f"📸 Saved image at {path}")
    else:
        print(f"⚠️ Failed to capture image at focus {focus}")
