"""Run the exact same PIL -> JPEG q=80 -> decode -> Classifier path N times on each
image and report whether predictions are bitwise identical across runs.

If outputs vary, NCNN inference itself is non-deterministic (likely threading).
If outputs match exactly, reproducibility issues come from upstream (camera frame
variation, resize being skipped/re-done, etc.), not from the model.
"""

import argparse
import io
import json
from pathlib import Path

from PIL import Image

from pyro_predictor.vision import Classifier

IMAGES_ROOT = Path("/Users/mateo/pyronear/test/sun_test/01_raw/01_raw/images")
SRC_JSONL = Path("/Users/mateo/pyronear/deploy/pyro-engine/compare_compression_bulk.jsonl")
FRAME_SIZE = (720, 1280)  # (H, W)
JPEG_QUALITY = 80


def preprocess(path: Path) -> Image.Image:
    """Engine pipeline: load -> RGB -> PIL bilinear resize -> JPEG q=80 -> decode."""
    with Image.open(path) as img:
        rgb = img.convert("RGB")
    resized = rgb.resize((FRAME_SIZE[1], FRAME_SIZE[0]), Image.Resampling.BILINEAR)
    buf = io.BytesIO()
    resized.save(buf, format="JPEG", quality=JPEG_QUALITY)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def preds_to_tuple(preds) -> tuple:
    """Hashable, order-independent representation of a prediction set."""
    return tuple(sorted((float(p[0]), float(p[1]), float(p[2]), float(p[3]), float(p[4])) for p in preds))


def diff_summary(a, b) -> str:
    if len(a) != len(b):
        return f"COUNT differs: {len(a)} vs {len(b)}"
    diffs = []
    for i, (pa, pb) in enumerate(zip(a, b)):
        deltas = [abs(pa[k] - pb[k]) for k in range(5)]
        if any(d > 0 for d in deltas):
            diffs.append(f"  det{i}: max coord-delta={max(deltas[:4]):.6f}, conf-delta={pa[4] - pb[4]:+.6f}")
    return "\n".join(diffs) if diffs else "identical"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=3, help="Number of repeats per image")
    parser.add_argument("--n-images", type=int, default=5, help="How many images to test")
    parser.add_argument("--seed-source", default="raw_detections", choices=["raw_detections", "first"])
    args = parser.parse_args()

    if args.seed_source == "raw_detections":
        with SRC_JSONL.open() as f:
            recs = [json.loads(line) for line in f if line.strip()]
        with_det = [r for r in recs if "error" not in r and r.get("raw")]
        sample_paths = [IMAGES_ROOT / r["path"] for r in with_det[: args.n_images]]
    else:
        sample_paths = sorted(IMAGES_ROOT.rglob("*.jpg"))[: args.n_images]

    print(f"Testing {len(sample_paths)} images, {args.runs} runs each\n")

    classifier = Classifier(verbose=False)

    n_identical = 0
    n_differ = 0
    for path in sample_paths:
        rel = path.relative_to(IMAGES_ROOT)
        runs = []
        for _ in range(args.runs):
            img = preprocess(path)  # fresh decode each time
            preds = classifier(img, {})
            runs.append(preds_to_tuple(preds))

        unique = set(runs)
        if len(unique) == 1:
            n_identical += 1
            print(f"[OK]   {rel}  -> {len(runs[0])} dets, all {args.runs} runs identical")
        else:
            n_differ += 1
            print(f"[DIFF] {rel}  -> {len(unique)} distinct outputs across {args.runs} runs")
            for i in range(1, args.runs):
                d = diff_summary(runs[0], runs[i])
                if d != "identical":
                    print(f"       run0 vs run{i}:")
                    print(f"       {d}")

    print(f"\nSummary: {n_identical} identical / {n_differ} differ (out of {len(sample_paths)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
