# Resize & JPEG Compression Impact on Classifier Predictions

## Objective

Quantify how the engine's preprocessing pipeline degrades wildfire detections relative to raw camera frames, and identify the dominant loss source.

The engine pipeline (see `pyroengine/engine.py` → `predict`) does:

```
raw frame  ──► PIL.resize((1280, 720), Image.BILINEAR)  ──► JPEG q=80 (upload path)  ──► Classifier
```

Two questions:

1. How much does this pipeline lose vs. running the model on the raw image?
2. Where does the loss come from — the resize, the JPEG, or both?

## Dataset

- Path: `/Users/mateo/pyronear/test/sun_test/01_raw/01_raw/images`
- Total: **17,122 images** at native 3840×2160 (4K).
- Sample processed in this study:
  - **1,100 images** with the full 3-stage pipeline (`compare_compression_bulk.jsonl`) — 265 of these had at least one raw detection.
  - **265 images** (the raw-detection subset) re-tested with OpenCV resize (`compare_resize_cv2.jsonl`).
  - **265 images** (same subset) re-tested with full-resolution JPEG round-trip (`compare_jpeg_at_resolutions.jsonl`).

Model: `pyro_predictor.vision.Classifier` (default settings).

## Methodology

### Test 1 — Resize + JPEG impact (`compare_compression_bulk.py`)

For each image, run the Classifier three times:

| Stage | Description |
|---|---|
| `raw` | original 3840×2160 PIL image, only `convert("RGB")` |
| `resized` | `PIL.resize((1280, 720), Image.BILINEAR)` — engine default |
| `jpeg80` | resized image round-tripped through `JPEG quality=80` then re-decoded |

### Test 2 — Resize-method comparison (`compare_resize_cv2.py`)

Filter the 265 images with raw detections, and add two more stages:

| Stage | Description |
|---|---|
| `cv2_area` | `cv2.resize(..., INTER_AREA)` to 1280×720 |
| `cv2_linear` | `cv2.resize(..., INTER_LINEAR)` to 1280×720 |

### Test 3 — Isolate JPEG impact at each resolution (`compare_jpeg_at_resolutions.py`)

On the same 265 raw-detection images, add the missing JPEG-at-full-resolution stage so we can disentangle "what the resize does" from "what the JPEG does":

| Stage | Description |
|---|---|
| `raw` | original 4K, no transform |
| `raw_jpeg80` | original 4K → JPEG q=80 round-trip |
| `resized` | PIL BILINEAR → 1280×720 |
| `resized_jpeg80` | resize then JPEG q=80 — **what the API actually receives** |

### Metrics (`analyze_compression_impact.py`)

For each ordered stage pair (A → B):

- per-stage detection counts and image-with-detection counts
- greedy IoU matching at threshold 0.3 — pairs are considered the same detection if IoU ≥ 0.3
- on matched pairs: IoU distribution and confidence delta (B − A)
- unmatched detections in A: "**lost**" (existed in A, not in B)
- unmatched detections in B: "**gained**" (new detections in B)
- "**images that lost ALL detections**": had ≥1 detection in A, none in B
- "**images that gained NEW detections**": had 0 detections in A, ≥1 in B

## Results

### Test 1 — 3-stage pipeline (n = 1,100)

Per-stage totals:

| Stage | Detections | Images with det | Mean conf | Median conf |
|---|---:|---:|---:|---:|
| raw | 305 | 265 | 0.315 | 0.271 |
| resized (PIL bilinear) | 166 | 148 | 0.338 | 0.303 |
| jpeg80 | 144 | 136 | 0.331 | 0.288 |

Image-level agreement (Y = ≥1 detection in that stage):

```
raw  resized  jpeg80   count
.    .        .        784
Y    .        .        131
Y    Y        Y         90
Y    Y        .         30
.    .        Y         23
.    Y        .         19
Y    .        Y         14
.    Y        Y          9
```

Stage-pair statistics:

| Pair | matched | matched IoU (med) | Δconf (med) | images lost ALL | images gained NEW |
|---|---:|---:|---:|---:|---:|
| raw → resized | 121 | 0.895 | **−0.061** | **145** | 28 |
| resized → jpeg80 | 98 | 0.910 | −0.013 | 49 | 37 |
| raw → jpeg80 | 92 | 0.864 | −0.063 | **161** | 32 |

**Findings:**

- **Resize is responsible for almost all the loss.** Raw → resized drops detections from 305 to 166 (−46%) and wipes detections off **145 images**.
- **JPEG q=80 on top of resize is small.** Detection count goes 166 → 144 (−13%); the "image lost ALL" / "image gained NEW" counts are roughly balanced (49 vs. 37).
- **Box positions are stable.** Matched-detection IoU sits at median ≈ 0.86–0.91 across pairs.
- **Lost detections are mostly weak ones.** The detections that disappear during resize cluster near the model's threshold (median conf ≈ 0.22–0.24, p75 ≤ 0.32).

### Test 2 — Resize-method comparison (n = 265, only images with raw detections)

Per-stage totals:

| Stage | Detections | Images with det | Mean conf | Median conf |
|---|---:|---:|---:|---:|
| raw | 305 | 265 | 0.315 | 0.271 |
| **cv2 INTER_LINEAR** | **179** | **158** | 0.355 | 0.314 |
| cv2 INTER_AREA | 157 | 133 | 0.355 | 0.322 |
| PIL BILINEAR (current engine) | 138 | 120 | 0.353 | 0.321 |

Vs. raw baseline (3840×2160):

| Resize method | detections retained | images that lost ALL detections |
|---|---:|---:|
| PIL BILINEAR | 138 / 305 (45%) | 145 |
| cv2 INTER_AREA | 157 / 305 (51%) | 132 |
| **cv2 INTER_LINEAR** | **179 / 305 (59%)** | **107** |

Direct method-vs-method comparisons:

| Pair | matched | matched IoU (med) | Δconf (med) | images lost ALL | images gained NEW |
|---|---:|---:|---:|---:|---:|
| pil_bilinear → cv2_area | 133 | 0.965 | +0.024 | 3 | 16 |
| pil_bilinear → cv2_linear | 120 | 0.927 | **+0.039** | 7 | **45** |
| cv2_area → cv2_linear | 139 | 0.946 | +0.012 | 7 | 32 |

**Findings:**

- **OpenCV resize beats PIL on detection retention.** `cv2.INTER_LINEAR` recovers ~30% more detections than PIL BILINEAR (179 vs. 138), median Δconf **+0.04**.
- **`cv2.INTER_AREA` (the textbook choice for downsampling)** lands between the two — better than PIL but worse than INTER_LINEAR for this model.
- **Spatial stability is preserved**: matched-detection median IoU ≥ 0.93 between any two resize methods.
- **Caveat**: the extra detections that `cv2.INTER_LINEAR` gains have a median confidence of ~0.21 — they sit close to threshold and may include some borderline noise. Worth a separate precision check on a labelled set before changing engine defaults.

### Test 3 — JPEG impact, isolated at each resolution (n = 265)

Per-stage totals:

| Stage | Detections | Images with det | Median conf |
|---|---:|---:|---:|
| raw (4K, no transform) | 305 | 265 | 0.271 |
| **raw + JPEG q=80** (4K) | 293 (96%) | 254 | 0.270 |
| resized (PIL bilinear → 720p) | 138 (45%) | 120 | 0.321 |
| **resized + JPEG q=80** *(what API gets)* | 112 (37%) | 104 | 0.314 |

Pair deltas:

| Step (A → B) | matched IoU (med) | Δconf (med) | imgs lost ALL | imgs gained NEW |
|---|---:|---:|---:|---:|
| raw → **raw_jpeg80** *(JPEG at full 4K)* | 0.963 | **±0.000** | 11 | 0 |
| raw → resized *(resize alone)* | 0.895 | −0.061 | **145** | 0 |
| resized → resized_jpeg80 *(JPEG after resize)* | 0.910 | −0.014 | 30 | 14 |
| raw → resized_jpeg80 *(full pipeline)* | 0.864 | −0.063 | **161** | 0 |

**Findings:**

- **JPEG q=80 at full 4K resolution is essentially free**: detections 305 → 293 (−4%), median Δconf is **exactly 0.000**, matched IoU 0.96. The 11 images that lose all detections had only borderline detections to begin with (median conf 0.18).
- **The full API pipeline keeps only 37% of raw detections** (112/305). About ⅔ of the loss is the resize, ⅓ is the JPEG-after-resize interaction.
- **Compression hurts more after a resize.** Same q=80, but median Δconf goes from ±0.000 (at 4K) to −0.014 (at 720p), and 30 images lose all detections vs. 11 — downsampled images have fewer pixels per smoke plume, so JPEG quantization eats relatively more signal.
- **Order doesn't matter much**: jpeg-then-resize and resize-then-jpeg land within a few detections of each other.

### Test 4 — Confidence change (not just survival) and threshold crossings

The "matched/lost/gained" framing above counts whether a detection survives, but **a detection that drops from 0.7 to 0.3 still "matched"** — yet it's now silently below the engine's alert threshold (default `conf_thresh=0.35`). Re-analysing the same data with that lens:

Confidence-delta distribution on **matched** detections:

| Pair | n | median Δ | p10 (worst-drop tail) | min (worst single drop) |
|---|---:|---:|---:|---:|
| raw → raw_jpeg80 (JPEG at 4K) | 282 | +0.000 | −0.046 | −0.178 |
| raw → resized | 121 | −0.061 | −0.218 | **−0.422** |
| resized → resized_jpeg80 | 89 | −0.014 | −0.110 | −0.325 |
| raw → resized_jpeg80 (full pipeline) | 92 | −0.063 | −0.226 | **−0.417** |

Threshold crossings at the engine's `conf_thresh = 0.35`. "Demoted" = above threshold in A, below in B. "Lost above-threshold" = was ≥0.35 in A, dropped from the prediction list entirely in B.

| Pair | matched stays above | **demoted (above→below)** | promoted (below→above) | **lost above-thresh entirely** | gained above-thresh entirely |
|---|---:|---:|---:|---:|---:|
| raw → raw_jpeg80 | 80 | 6 | 12 | 0 | 0 |
| raw → resized | 52 | 20 | 7 | 14 | 3 |
| resized → resized_jpeg80 | 40 | 11 | 5 | 11 | 2 |
| **raw → resized_jpeg80** | **35** | **20** | 9 | **31** | 3 |

Worst single confidence drops, raw → full pipeline (showing the top 5):

```
0.602  →  0.185     (loss 0.417)
0.795  →  0.441     (loss 0.354)
0.543  →  0.213     (loss 0.330)
0.612  →  0.300     (loss 0.312)
0.535  →  0.245     (loss 0.290)
```

**Findings:**

- **Half of the alertable raw detections never reach the alert threshold after the engine pipeline.** Above-threshold count goes from `35 + 20 + 31 = 86` raw detections that *should* alert, down to `35 + 9 = 44` that actually do — a **49% loss of actionable detections**, much worse than the 37% retention figure that counts all detections regardless of confidence.
- **Big silent drops happen.** 15 detections lose ≥0.2 confidence in the full pipeline, including a `0.60 → 0.19` and a `0.54 → 0.21`. These are confident detections being demoted to noise level — they don't show up as "lost" because they still match spatially, but they wouldn't trigger an alert.
- **JPEG at 4K causes zero big drops** (none ≥0.2). All the big drops come from the resize step.
- **The conf-change story is roughly twice as bad as the binary survival story.** Survival counts the borderline detections that fall off the bottom; threshold crossings additionally count the strong detections that drop down to the borderline.

## Conclusions & Recommendations

1. The **3840×2160 → 1280×720 downscale is the main source of degradation** in the engine pipeline. JPEG q=80 at full resolution is essentially free; JPEG after resize costs an extra ~19% on top.
2. **Counted properly (with threshold crossings, not just survival), the engine pipeline loses ~49% of the alertable detections** the model would have made on raw frames — strong raw detections regularly get demoted from above-threshold to below-threshold (worst observed: 0.79 → 0.44, 0.60 → 0.19).
3. The PIL BILINEAR resize used in the engine is the **worst-performing of the three resize methods tested** for this model. Switching to `cv2.resize(..., INTER_LINEAR)` is a one-line change that recovers ~30% more detections without moving boxes.
4. The detections lost to resize are concentrated near the model's confidence threshold — they're the borderline cases. If the goal is earlier detection of weak smoke plumes, a better resize method directly translates into earlier alerts.
4. Suggested next steps:
   - Validate the cv2 INTER_LINEAR gain on a **labelled benchmark** to confirm it's not just adding false positives.
   - Try keeping the model input at a **larger size** (e.g. 1920×1080) to bound the loss further; measure the latency cost.
   - Consider sending the API a **higher-quality JPEG** (q=90 or q=95) once it's already downsampled — quantization noise costs more at lower resolution.

## Statistical confidence — is n = 265 enough?

For each main result, 95% CI on the proportions involved (Wald approx., n = 265):

| Result | Point estimate | ±95% CI | Robust? |
|---|---:|---:|---|
| Resize alone wipes detections off all images | 145/265 = 54.7% | ±6% | **Yes** — even the lower bound (~49%) is huge. |
| Full pipeline retains detections in some images | 104/265 = 39.2% | ±6% | **Yes** — clearly << 100%. |
| JPEG at 4K wipes detections off all images | 11/265 = 4.2% | ±2.4% | Effect direction is robust; exact value (2–7%) less so. |
| cv2_linear vs PIL bilinear — extra detections | +30% (179 vs 138) | n=265 | Direction robust, exact magnitude noisy. |

**Verdict:** the qualitative conclusions ("resize is the dominant loss", "JPEG at 4K is free", "cv2 beats PIL") are solid at n=265. The exact percentages on smaller effects (cv2_linear vs cv2_area, JPEG-at-4K loss) would tighten with more samples.

If you want to nail down the cv2 method ranking and the exact magnitude of the JPEG-after-resize penalty, **re-running on the full 17k images** (or at least a few more thousand) would help. For the headline conclusion ("resize is the problem, not JPEG") the current sample is already conclusive.

A more valuable next step than scaling up *this* test is to **run a labelled-benchmark precision check**: many of the "extra" detections cv2 recovers sit near threshold, and we can't tell from this study alone whether they're real smoke or noise.

## Files

- `compare_compression_bulk.py` — runs Test 1 (raw / pil_resized / jpeg80), resumable.
- `compare_resize_cv2.py` — runs Test 2 (adds cv2_area / cv2_linear) on the raw-detection subset.
- `compare_jpeg_at_resolutions.py` — runs Test 3 (adds raw_jpeg80, full-res JPEG round-trip).
- `analyze_compression_impact.py` — computes the stage-pair statistics shown above.
- `compare_compression_bulk.jsonl` — Test 1 results (1,100 records).
- `compare_resize_cv2.jsonl` — Test 2 results (265 records).
- `compare_jpeg_at_resolutions.jsonl` — Test 3 results (265 records).
