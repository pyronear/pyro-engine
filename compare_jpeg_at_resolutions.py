"""On the 265 images with raw detections, compute the missing stage:
raw image round-tripped through JPEG q=80 (full resolution, no resize).

The other three stages already exist in compare_compression_bulk.jsonl:
  - raw                = no transform
  - resized            = PIL BILINEAR -> 1280x720
  - jpeg80 (=resized_jpeg80) = resized then JPEG q=80  (== what API receives)

This script adds:
  - raw_jpeg80         = raw then JPEG q=80 (full 4K)

Output: compare_jpeg_at_resolutions.jsonl with all 4 stages per image.
"""

import argparse
import io
import json
import sys
import time
from pathlib import Path

from PIL import Image
from tqdm import tqdm

from pyro_predictor.vision import Classifier

IMAGES_ROOT = Path("/Users/mateo/pyronear/test/sun_test/01_raw/01_raw/images")
SRC_JSONL = Path("/Users/mateo/pyronear/deploy/pyro-engine/compare_compression_bulk.jsonl")
OUT_PATH = Path("/Users/mateo/pyronear/deploy/pyro-engine/compare_jpeg_at_resolutions.jsonl")
JPEG_QUALITY = 80


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


def jpeg_roundtrip(img: Image.Image, quality: int) -> Image.Image:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def process_one(classifier: Classifier, path: Path, prior: dict) -> dict:
    rel = str(path.relative_to(IMAGES_ROOT))
    try:
        with Image.open(path) as img:
            rgb = img.convert("RGB")

        raw_jpeg = jpeg_roundtrip(rgb, JPEG_QUALITY)
        raw_jpeg_preds = classifier(raw_jpeg, {})

        return {
            "path": rel,
            "raw": prior.get("raw", []),
            "raw_jpeg80": preds_to_list(raw_jpeg_preds),
            "resized": prior.get("resized", []),
            "resized_jpeg80": prior.get("jpeg80", []),
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
    print(f"Source: {len(all_records)} total, {len(targets)} with raw detections")

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
