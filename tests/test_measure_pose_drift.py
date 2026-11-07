import csv
import importlib.util
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

_spec = importlib.util.spec_from_file_location(
    "measure_pose_drift", Path(__file__).parents[1] / "src" / "measure_pose_drift.py"
)
mpd = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mpd)


def _scene(shift=(0.0, 0.0)) -> Image.Image:
    rng = np.random.default_rng(0)
    base = cv2.GaussianBlur((rng.random((720, 1280)) * 255).astype(np.uint8), (0, 0), 2)
    m = np.float32([[1, 0, shift[0]], [0, 1, shift[1]]])
    gray = cv2.warpAffine(base, m, (1280, 720), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
    rgb = np.stack([gray, gray // 2, gray // 3], axis=-1)  # R != G: not an IR frame
    return Image.fromarray(rgb)


def test_register_recovers_known_shift():
    for tx, ty in [(0.4, -0.3), (7.6, 2.2)]:
        dx, dy, _ = mpd.register(mpd.gradient(_scene()), mpd.gradient(_scene((tx, ty))))
        assert abs(dx - tx) < 0.1
        assert abs(dy - ty) < 0.1


def test_drift_logger_writes_rows_and_keeps_large_offsets(tmp_path):
    drift = mpd.DriftLogger(tmp_path, max_px=5.0, min_cc=0.5, max_saved=10)
    drift.process("cam_1", _scene())  # becomes the reference
    drift.process("cam_1", _scene())  # identical frame: skipped
    drift.process("cam_1", _scene((1.0, 0.0)))
    drift.process("cam_1", _scene((8.0, 0.0)))

    assert (tmp_path / "ref" / "cam_1.jpg").exists()
    rows = list(csv.DictReader((tmp_path / "drift.csv").open()))
    assert len(rows) == 3
    assert abs(float(rows[-1]["dx"]) - 8.0) < 0.5
    assert len(list((tmp_path / "anomalies").glob("*.jpg"))) == 1


def test_register_keeps_inputs_intact():
    # 432x768 crops have FFT-optimal sizes, where phaseCorrelate would window its inputs in place.
    ref = np.random.default_rng(0).random((432, 768)).astype(np.float32)
    img, ref0 = ref.copy(), ref.copy()
    mpd.register(ref, img)
    assert np.array_equal(ref, ref0)


def test_large_frames_are_downscaled_before_saving(tmp_path):
    drift = mpd.DriftLogger(tmp_path, max_px=5.0, min_cc=0.5, max_saved=10)
    drift.process("cam_4k", _scene().resize((3840, 2160)))
    assert Image.open(tmp_path / "ref" / "cam_4k.jpg").size == (1280, 720)
