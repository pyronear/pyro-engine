import json
import time
import numpy as np
from PIL import Image
import cv2
from pyroengine.sensors import ReolinkCamera

# --------------------------
# Sharpness measurement
# --------------------------
def measure_sharpness(pil_image):
    t0 = time.time()
    img = pil_image.convert('L')
    arr = np.array(img)
    laplacian = cv2.Laplacian(arr, cv2.CV_64F)
    sharpness = laplacian.var()
    elapsed = time.time() - t0
    return sharpness, elapsed

# --------------------------
# Greedy focus finder
# --------------------------
def find_best_focus(camera_controller, default=720, min_focus=600, max_focus=800):
    def capture_and_measure(pos):
        camera_controller.set_manual_focus(position=pos)
        t0 = time.time()
        img = camera_controller.capture()
        sharpness, _ = measure_sharpness(img)
        total_time = time.time() - t0
        print(f"Focus {pos}: Sharpness = {sharpness:.2f}, TotalStep = {total_time:.2f}s")
        return sharpness

    current_pos = default
    current_sharp = capture_and_measure(current_pos)

    prev_sharp = capture_and_measure(current_pos - 1)
    next_sharp = capture_and_measure(current_pos + 1)

    if prev_sharp > current_sharp and prev_sharp >= next_sharp:
        direction = -1
    elif next_sharp > current_sharp:
        direction = 1
    else:
        print(f"\nBest focus position: {current_pos} with sharpness: {current_sharp:.2f}")
        return current_pos

    best_pos = current_pos + direction
    best_sharp = max(prev_sharp, next_sharp)

    while True:
        next_pos = best_pos + direction
        if next_pos < min_focus or next_pos > max_focus:
            break

        sharp = capture_and_measure(next_pos)
        if sharp > best_sharp:
            best_pos = next_pos
            best_sharp = sharp
        else:
            break

    print(f"\nBest focus position for {camera_controller.ip}: {best_pos} with sharpness: {best_sharp:.2f}")
    return best_pos

# --------------------------
# Main loop
# --------------------------
def process_all_cameras(credentials_path='credentials.json'):
    with open(credentials_path, 'r') as f:
        data = json.load(f)

    for ip, config in data.items():
        print(f"\n===> Processing camera {ip} ({config.get('name', 'Unnamed')})")

        focus_start = config.get("focus_position", 720)

        # Initialize ReolinkCamera
        camera = ReolinkCamera(
            ip_address=ip,
            username="admin",
            password="@Pyronear",
            protocol="http"
        )

        best_focus = find_best_focus(camera, default=focus_start)
        print(f"Final best focus for {ip}: {best_focus}")

# --------------------------
# Run
# --------------------------
if __name__ == '__main__':
    process_all_cameras()
