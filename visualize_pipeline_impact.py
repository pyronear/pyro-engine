"""Bucket images into 'big_change' vs 'stable' based on raw -> full-pipeline prediction
diff, draw raw predictions on each, and write to two folders for visual inspection.

Bucketing rule (raw vs resized_jpeg80):
  - big_change : >=1 raw detection was lost entirely, OR
                 >=1 matched detection lost >= MIN_DROP confidence, OR
                 >=1 matched detection crossed from >= ALERT_THRESH to < ALERT_THRESH
  - stable     : all raw detections matched with |Δconf| < STABLE_TOL and stayed on the
                 same side of ALERT_THRESH
  - other      : low-conf borderline cases (skipped from output, kept for stats)
"""

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

IMAGES_ROOT = Path("/Users/mateo/pyronear/test/sun_test/01_raw/01_raw/images")
SRC_JSONL = Path("/Users/mateo/pyronear/deploy/pyro-engine/compare_jpeg_at_resolutions.jsonl")
CV2_JSONL = Path("/Users/mateo/pyronear/deploy/pyro-engine/compare_resize_cv2.jsonl")
OUT_ROOT = Path("/Users/mateo/pyronear/deploy/pyro-engine/pipeline_impact_viz")
BIG_DIR = OUT_ROOT / "big_change"
STABLE_DIR = OUT_ROOT / "stable"

# Color mapping for the 4 prediction stages (BGR-friendly RGB tuples)
STAGE_COLORS = {
    "raw":            ((255,  60,  60), True),   # red,   label above
    "pil_resized":    ((255, 200,  40), True),   # yellow,label above
    "pil_jpeg80":     ((  0, 200, 255), False),  # cyan,  label below
    "cv2_linear":     ((  0, 220,  90), False),  # green, label below
}

ALERT_THRESH = 0.35
MIN_DROP = 0.15
STABLE_TOL = 0.05
IOU_MATCH = 0.3
DRAW_MIN_CONF = 0.15


def iou(a, b):
    ix1 = max(a["x1"], b["x1"])
    iy1 = max(a["y1"], b["y1"])
    ix2 = min(a["x2"], b["x2"])
    iy2 = min(a["y2"], b["y2"])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    aa = max(0, a["x2"] - a["x1"]) * max(0, a["y2"] - a["y1"])
    bb = max(0, b["x2"] - b["x1"]) * max(0, b["y2"] - b["y1"])
    u = aa + bb - inter
    return inter / u if u > 0 else 0


def match(A, B, thr=IOU_MATCH):
    pairs = [(iou(a, b), i, j) for i, a in enumerate(A) for j, b in enumerate(B)]
    pairs = [p for p in pairs if p[0] >= thr]
    pairs.sort(reverse=True)
    used_a, used_b, m = set(), set(), []
    for s, i, j in pairs:
        if i in used_a or j in used_b:
            continue
        used_a.add(i)
        used_b.add(j)
        m.append((i, j, s))
    return m, [i for i in range(len(A)) if i not in used_a]


def classify(raw, after) -> str:
    if not raw:
        return "other"
    matched, lost_idx = match(raw, after)

    has_big_drop = False
    has_demotion = False
    max_abs_delta = 0.0
    crossed_threshold = False

    for i, j, _ in matched:
        ca, cb = raw[i]["conf"], after[j]["conf"]
        d = cb - ca
        max_abs_delta = max(max_abs_delta, abs(d))
        if d <= -MIN_DROP:
            has_big_drop = True
        if ca >= ALERT_THRESH and cb < ALERT_THRESH:
            has_demotion = True
        if (ca >= ALERT_THRESH) != (cb >= ALERT_THRESH):
            crossed_threshold = True

    has_lost = any(raw[i]["conf"] >= 0.2 for i in lost_idx)

    if has_lost or has_big_drop or has_demotion:
        return "big_change"
    if not lost_idx and max_abs_delta < STABLE_TOL and not crossed_threshold:
        return "stable"
    return "other"


def _draw_one_set(draw: ImageDraw.ImageDraw, preds, W: int, H: int, color, font, label_above: bool, tag: str):
    line_w = max(3, W // 600)
    for p in preds:
        if p["conf"] < DRAW_MIN_CONF:
            continue
        x1, y1, x2, y2 = p["x1"] * W, p["y1"] * H, p["x2"] * W, p["y2"] * H
        draw.rectangle([x1, y1, x2, y2], outline=color, width=line_w)
        label = f"{tag} {p['conf']:.2f}"
        if label_above:
            tx, ty = x1 + 4, max(0, y1 - 28)
        else:
            tx, ty = x1 + 4, min(H - 28, y2 + 2)
        bbox = draw.textbbox((tx, ty), label, font=font)
        draw.rectangle(bbox, fill=color)
        draw.text((tx, ty), label, fill="white", font=font)


def _draw_legend(draw: ImageDraw.ImageDraw, font, stages: list[str]) -> None:
    pad = 10
    rows = [(s, STAGE_COLORS[s][0]) for s in stages]
    text_w = max(draw.textbbox((0, 0), name, font=font)[2] for name, _ in rows)
    box_h = (font.size + 8) * len(rows) + 2 * pad
    box_w = text_w + 60
    draw.rectangle([0, 0, box_w, box_h], fill=(0, 0, 0))
    for i, (name, color) in enumerate(rows):
        y = pad + i * (font.size + 8)
        draw.rectangle([pad, y + 4, pad + 24, y + font.size + 4], fill=color)
        draw.text((pad + 32, y), name, fill="white", font=font)


def draw_predictions(img: Image.Image, stage_preds: dict) -> Image.Image:
    """stage_preds is {stage_name: list[pred]}. Stages drawn in STAGE_COLORS order."""
    out = img.copy()
    draw = ImageDraw.Draw(out)
    W, H = out.size
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", max(20, W // 80))
    except OSError:
        font = ImageFont.load_default()
    stages_drawn = []
    for stage, (color, above) in STAGE_COLORS.items():
        preds = stage_preds.get(stage)
        if preds is None:
            continue
        _draw_one_set(draw, preds, W, H, color=color, font=font, label_above=above, tag=stage)
        stages_drawn.append(stage)
    if stages_drawn:
        _draw_legend(draw, font, stages_drawn)
    return out


def annotation_suffix(raw, after) -> str:
    """Short tag describing the worst change, used in the output filename."""
    if not raw:
        return "no_raw"
    matched, lost_idx = match(raw, after)
    worst_drop = 0.0
    for i, j, _ in matched:
        worst_drop = max(worst_drop, raw[i]["conf"] - after[j]["conf"])
    n_lost = len(lost_idx)
    return f"drop{worst_drop:.2f}_lost{n_lost}".replace(".", "")


def main() -> int:
    BIG_DIR.mkdir(parents=True, exist_ok=True)
    STABLE_DIR.mkdir(parents=True, exist_ok=True)

    with SRC_JSONL.open() as f:
        records = [json.loads(line) for line in f if line.strip()]
    records = [r for r in records if "error" not in r]

    cv2_by_path: dict[str, list] = {}
    if CV2_JSONL.exists():
        with CV2_JSONL.open() as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                if "error" in rec:
                    continue
                cv2_by_path[rec["path"]] = rec.get("cv2_linear", [])
        print(f"Loaded cv2_linear preds for {len(cv2_by_path)} paths")

    counts = {"big_change": 0, "stable": 0, "other": 0}
    for r in records:
        raw = r.get("raw", [])
        pil_resized = r.get("resized", [])
        pil_jpeg80 = r.get("resized_jpeg80", [])
        cv2_linear = cv2_by_path.get(r["path"])

        bucket = classify(raw, pil_jpeg80)
        counts[bucket] += 1
        if bucket == "other":
            continue

        src = IMAGES_ROOT / r["path"]
        if not src.exists():
            continue

        if bucket == "big_change":
            stage_preds = {
                "raw": raw,
                "pil_resized": pil_resized,
                "pil_jpeg80": pil_jpeg80,
            }
            if cv2_linear is not None:
                stage_preds["cv2_linear"] = cv2_linear
        else:  # stable
            stage_preds = {"raw": raw}

        try:
            with Image.open(src) as img:
                rgb = img.convert("RGB")
            annotated = draw_predictions(rgb, stage_preds)
        except Exception as e:
            print(f"skip {r['path']}: {e}")
            continue

        suffix = annotation_suffix(raw, pil_jpeg80)
        flat_name = r["path"].replace("/", "__")
        stem = Path(flat_name).stem
        ext = Path(flat_name).suffix or ".jpg"
        out_name = f"{stem}__{suffix}{ext}"
        target_dir = BIG_DIR if bucket == "big_change" else STABLE_DIR
        annotated.save(target_dir / out_name, format="JPEG", quality=85)

    print(f"Total records considered: {len(records)}")
    for k, v in counts.items():
        print(f"  {k:11s} : {v}")
    print(f"Wrote big_change images to: {BIG_DIR}")
    print(f"Wrote stable     images to: {STABLE_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
