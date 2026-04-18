"""Run Classifier on all images at three stages: raw, resized (720,1280), and JPEG q=80.

Saves per-image, per-stage predictions to a JSONL file for comparison.
"""

import argparse
import io
import json
import sys
import time
from pathlib import Path
from typing import Tuple

from PIL import Image
from tqdm import tqdm

from pyro_predictor.vision import Classifier

IMAGES_ROOT = Path("/Users/mateo/pyronear/test/sun_test/01_raw/01_raw/images")
OUT_PATH = Path("/Users/mateo/pyronear/deploy/pyro-engine/compare_compression_bulk.jsonl")
FRAME_SIZE: Tuple[int, int] = (720, 1280)  # (H, W) - engine default
JPEG_QUALITY = 80
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


def process_one(classifier: Classifier, path: Path) -> dict:
    rel = str(path.relative_to(IMAGES_ROOT))
    try:
        with Image.open(path) as img:
            rgb = img.convert("RGB")

        raw_preds = classifier(rgb, {})

        resized = rgb.resize((FRAME_SIZE[1], FRAME_SIZE[0]), Image.Resampling.BILINEAR)
        resized_preds = classifier(resized, {})

        buf = io.BytesIO()
        resized.save(buf, format="JPEG", quality=JPEG_QUALITY)
        buf.seek(0)
        jpeg_img = Image.open(buf).convert("RGB")
        jpeg_preds = classifier(jpeg_img, {})

        return {
            "path": rel,
            "raw_size": list(rgb.size),
            "resized_size": list(resized.size),
            "raw": preds_to_list(raw_preds),
            "resized": preds_to_list(resized_preds),
            "jpeg80": preds_to_list(jpeg_preds),
        }
    except Exception as e:
        return {"path": rel, "error": repr(e)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0, help="Process only first N (0 = all)")
    args = parser.parse_args()

    files = sorted(p for p in IMAGES_ROOT.rglob("*") if p.suffix.lower() in EXTS)

    done: set[str] = set()
    if OUT_PATH.exists():
        with OUT_PATH.open() as f:
            done = {json.loads(line)["path"] for line in f if line.strip()}
        print(f"Found {len(done)} already-processed images in {OUT_PATH.name}")

    todo = [p for p in files if str(p.relative_to(IMAGES_ROOT)) not in done]
    if args.limit:
        todo = todo[: args.limit]
    total = len(todo)
    print(f"Processing {total} new images (skipping {len(files) - total})")

    if total == 0:
        return 0

    classifier = Classifier(verbose=False)

    t0 = time.time()
    with OUT_PATH.open("a") as f:
        for path in tqdm(todo, unit="img", smoothing=0.05):
            record = process_one(classifier, path)
            f.write(json.dumps(record) + "\n")
            f.flush()

    print(f"Done. Appended {total} records to {OUT_PATH} in {(time.time() - t0) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
