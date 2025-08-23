import time
from statistics import mean, stdev

from anonymizer.vision import Anonymizer
from reolink import ReolinkCamera

# Initialize model and camera
ncnn_model = Anonymizer()
cam = ReolinkCamera(
    ip_address="192.168.1.12",
    username="admin",
    password="@Pyronear",
    cam_type="ptz",
    cam_poses=[0, 1, 2, 3],
    cam_azimuths=[0, 1, 2, 3],
    focus_position=720,
)

NUM_ITERS = 30
WARMUP = 3

capture_times = []
infer_times = []
total_times = []
num_dets = []  # number of detections per frame


def now():
    return time.perf_counter()


# Warmup runs
for _ in range(WARMUP):
    im = cam.capture()
    if im is not None:
        _ = ncnn_model(im)

print(f"Starting timed run of {NUM_ITERS} iterations.")

for i in range(1, NUM_ITERS + 1):
    t0 = now()
    # Capture
    t_cap0 = now()
    im = cam.capture()
    t_cap1 = now()

    if im is None:
        print(f"[{i:02d}] capture failed, skipping")
        continue

    # Inference
    t_inf0 = now()
    pred = ncnn_model(im)
    t_inf1 = now()

    # Stats
    cap_t = t_cap1 - t_cap0
    inf_t = t_inf1 - t_inf0
    tot_t = t_inf1 - t0

    capture_times.append(cap_t)
    infer_times.append(inf_t)
    total_times.append(tot_t)
    num_dets.append(0 if pred is None else int(getattr(pred, "shape", [0])[0]))

    fps = 1.0 / tot_t if tot_t > 0 else float("inf")
    print(
        f"[{i:02d}] capture {cap_t * 1000:.1f} ms, infer {inf_t * 1000:.1f} ms, total {tot_t * 1000:.1f} ms, fps {fps:.2f}, dets {num_dets[-1]}"
    )


# Summary
def fmt_stats(vals, label):
    if not vals:
        print(f"{label}: no data")
        return
    avg = mean(vals)
    sd = stdev(vals) if len(vals) > 1 else 0.0
    print(
        f"{label}: avg {avg * 1000:.1f} ms, std {sd * 1000:.1f} ms, min {min(vals) * 1000:.1f} ms, max {max(vals) * 1000:.1f} ms"
    )


print("\nSummary over successful iterations:")
fmt_stats(capture_times, "Capture")
fmt_stats(infer_times, "Inference")
fmt_stats(total_times, "Total frame")
if total_times:
    avg_fps = 1.0 / mean(total_times)
    print(f"Average FPS: {avg_fps:.2f}")
if num_dets:
    print(f"Average detections per frame: {mean(num_dets):.2f}")
