"""Re-run Classifier on images that produced raw detections, but resize via OpenCV
(INTER_AREA + INTER_LINEAR) instead of PIL BILINEAR. Used to test whether the
~46% detection drop from the (3840x2160 -> 1280x720) downscale is interpolation-driven.

Output is a new JSONL with: raw / pil_bilinear / cv2_area / cv2_linear predictions.
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

from pyro_predictor.vision import Classifier

IMAGES_ROOT = Path("/Users/mateo/pyronear/test/sun_test/01_raw/01_raw/images")
SRC_JSONL = Path("/Users/mateo/pyronear/deploy/pyro-engine/compare_compression_bulk.jsonl")
OUT_PATH = Path("/Users/mateo/pyronear/deploy/pyro-engine/compare_resize_cv2.jsonl")
FRAME_SIZE: Tuple[int, int] = (720, 1280)  # (H, W)
EXTS = {".jpg", ".jpeg", ".png"}


def preds_to_list(preds) -> list:
    return [
        {
            "x1": float(p[0]),
            "y1": float(p[1]),
            "x2": float(p[2]),
            "y2": float(p[3]),
            "conf": float(p[4]),
        }
        for p in preds
    ]


def cv2_resize_to_pil(rgb_pil: Image.Image, interpolation: int) -> Image.Image:
    arr = np.asarray(rgb_pil)  # H, W, 3 in RGB
    resized = cv2.resize(arr, (FRAME_SIZE[1], FRAME_SIZE[0]), interpolation=interpolation)
    return Image.fromarray(resized)


def process_one(classifier: Classifier, path: Path, prior: dict) -> dict:
    rel = str(path.relative_to(IMAGES_ROOT))
    try:
        with Image.open(path) as img:
            rgb = img.convert("RGB")

        cv2_area = cv2_resize_to_pil(rgb, cv2.INTER_AREA)
        cv2_linear = cv2_resize_to_pil(rgb, cv2.INTER_LINEAR)

        cv2_area_preds = classifier(cv2_area, {})
        cv2_linear_preds = classifier(cv2_linear, {})

        return {
            "path": rel,
            "raw": prior.get("raw", []),
            "pil_bilinear": prior.get("resized", []),
            "cv2_area": preds_to_list(cv2_area_preds),
            "cv2_linear": preds_to_list(cv2_linear_preds),
        }
    except Exception as e:
        return {"path": rel, "error": repr(e)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    with SRC_JSONL.open() as f:
        all_records = [json.loads(line) for line in f if line.strip()]
    targets = [r for r in all_records if "error" not in r and r.get("raw")]
    print(f"Source: {len(all_records)} total records, {len(targets)} with raw detections")

    done: set[str] = set()
    if OUT_PATH.exists():
        with OUT_PATH.open() as f:
            done = {json.loads(line)["path"] for line in f if line.strip()}
        print(f"Resuming: {len(done)} already processed")

    todo = [r for r in targets if r["path"] not in done]
    if args.limit:
        todo = todo[: args.limit]
    print(f"Processing {len(todo)} images")

    if not todo:
        return 0

    classifier = Classifier(verbose=False)

    t0 = time.time()
    with OUT_PATH.open("a") as f:
        for r in tqdm(todo, unit="img", smoothing=0.05):
            record = process_one(classifier, IMAGES_ROOT / r["path"], r)
            f.write(json.dumps(record) + "\n")
            f.flush()

    print(f"Done. Appended {len(todo)} records to {OUT_PATH} in {(time.time() - t0) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
