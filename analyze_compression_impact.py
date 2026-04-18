"""Analyze the impact of resize (720x1280) and JPEG-80 compression on Classifier predictions.

Reads compare_compression_bulk.jsonl and reports:
  - per-stage detection counts
  - image-level agreement / disagreement across stages
  - matched-detection confidence deltas and IoU distributions
"""

import argparse
import json
import statistics
from pathlib import Path
from typing import List, Tuple

JSONL_PATH = Path("/Users/mateo/pyronear/deploy/pyro-engine/compare_compression_bulk.jsonl")
STAGES = ("raw", "resized", "jpeg80")
IOU_MATCH = 0.3  # IoU threshold to consider two boxes "the same detection"


def iou(a: dict, b: dict) -> float:
    ix1 = max(a["x1"], b["x1"])
    iy1 = max(a["y1"], b["y1"])
    ix2 = min(a["x2"], b["x2"])
    iy2 = min(a["y2"], b["y2"])
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, a["x2"] - a["x1"]) * max(0.0, a["y2"] - a["y1"])
    area_b = max(0.0, b["x2"] - b["x1"]) * max(0.0, b["y2"] - b["y1"])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def match_boxes(boxes_a: List[dict], boxes_b: List[dict]) -> Tuple[List[Tuple[int, int, float]], List[int], List[int]]:
    """Greedy IoU matching. Returns matched pairs, unmatched_a indices, unmatched_b indices."""
    pairs = []
    for i, a in enumerate(boxes_a):
        for j, b in enumerate(boxes_b):
            score = iou(a, b)
            if score >= IOU_MATCH:
                pairs.append((score, i, j))
    pairs.sort(reverse=True)
    used_a, used_b, matched = set(), set(), []
    for score, i, j in pairs:
        if i in used_a or j in used_b:
            continue
        used_a.add(i)
        used_b.add(j)
        matched.append((i, j, score))
    unmatched_a = [i for i in range(len(boxes_a)) if i not in used_a]
    unmatched_b = [j for j in range(len(boxes_b)) if j not in used_b]
    return matched, unmatched_a, unmatched_b


def fmt_dist(values: List[float], unit: str = "") -> str:
    if not values:
        return "n=0"
    q = statistics.quantiles(values, n=4) if len(values) > 1 else [values[0]] * 3
    return (
        f"n={len(values)} "
        f"mean={statistics.mean(values):.3f}{unit} "
        f"median={statistics.median(values):.3f}{unit} "
        f"p25={q[0]:.3f} p75={q[2]:.3f} "
        f"min={min(values):.3f} max={max(values):.3f}"
    )


def analyze_pair(records: List[dict], a: str, b: str) -> None:
    """Compare detections in stage `a` vs stage `b` across all records."""
    print(f"\n=== {a}  →  {b} ===")
    matched_iou: List[float] = []
    matched_conf_delta: List[float] = []  # b.conf - a.conf (positive = stage b higher)
    lost: List[float] = []  # confs of detections present in a but not b
    gained: List[float] = []  # confs of detections present in b but not a
    images_lost_all = 0  # had detections in a, none survived to b
    images_gained_all = 0  # had no detections in a, gained some in b

    for r in records:
        boxes_a = r.get(a, [])
        boxes_b = r.get(b, [])
        matched, unmatched_a, unmatched_b = match_boxes(boxes_a, boxes_b)
        for i, j, score in matched:
            matched_iou.append(score)
            matched_conf_delta.append(boxes_b[j]["conf"] - boxes_a[i]["conf"])
        for i in unmatched_a:
            lost.append(boxes_a[i]["conf"])
        for j in unmatched_b:
            gained.append(boxes_b[j]["conf"])
        if boxes_a and not boxes_b:
            images_lost_all += 1
        if not boxes_a and boxes_b:
            images_gained_all += 1

    total_a = sum(len(r.get(a, [])) for r in records)
    total_b = sum(len(r.get(b, [])) for r in records)
    print(f"detections: {a}={total_a}  {b}={total_b}  matched={len(matched_iou)}")
    print(f"images that lost ALL detections going {a}→{b}: {images_lost_all}")
    print(f"images that gained NEW detections going {a}→{b}: {images_gained_all}")
    print(f"matched IoU       : {fmt_dist(matched_iou)}")
    print(f"conf delta (b-a)  : {fmt_dist(matched_conf_delta)}")
    print(f"lost det confs    : {fmt_dist(lost)}")
    print(f"gained det confs  : {fmt_dist(gained)}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", type=Path, default=JSONL_PATH)
    parser.add_argument("--conf", type=float, default=0.0, help="Filter detections with conf >= this")
    args = parser.parse_args()

    with args.jsonl.open() as f:
        records = [json.loads(line) for line in f if line.strip()]

    errors = [r for r in records if "error" in r]
    records = [r for r in records if "error" not in r]
    print(f"Loaded {len(records)} records ({len(errors)} errors skipped)")

    if args.conf > 0:
        for r in records:
            for s in STAGES:
                r[s] = [d for d in r.get(s, []) if d["conf"] >= args.conf]
        print(f"Applied conf >= {args.conf} filter")

    print("\n--- per-stage totals ---")
    for s in STAGES:
        n_det = sum(len(r.get(s, [])) for r in records)
        n_img = sum(1 for r in records if r.get(s))
        confs = [d["conf"] for r in records for d in r.get(s, [])]
        print(f"{s:8s}  detections={n_det:5d}  images_with_det={n_img:4d}  conf {fmt_dist(confs)}")

    print("\n--- image-level agreement ---")
    counts: dict[tuple[int, int, int], int] = {}
    for r in records:
        key = (
            1 if r.get(STAGES[0]) else 0,
            1 if r.get(STAGES[1]) else 0,
            1 if r.get(STAGES[2]) else 0,
        )
        counts[key] = counts.get(key, 0) + 1
    print(f"  raw resized jpeg80 -> count")
    for key, n in sorted(counts.items(), key=lambda x: -x[1]):
        marks = "".join("Y" if v else "." for v in key)
        print(f"  {marks[0]}    {marks[1]}      {marks[2]}      -> {n}")

    analyze_pair(records, "raw", "resized")
    analyze_pair(records, "resized", "jpeg80")
    analyze_pair(records, "raw", "jpeg80")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
