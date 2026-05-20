# PTZ Calibration Report — Reolink 823A16 & 823S2

**Date:** 2026-04-02
**Method:** Automated ORB keypoint matching + manual annotation for micro-pulse
**Timing:** Server-side (move+sleep+stop on Pi, no VPN latency)

## Motor Model

```
displacement = omega * T + bias
```

- **omega** = angular velocity (deg/s)
- **T** = commanded duration (s)
- **bias** = coast distance from mechanical inertia at start/stop (deg)
- At T=0 (micro-pulse), displacement = bias

## Calibration Values

### Reolink 823A16

#### Pan (zoom 0, speeds 1-5)

| Speed | omega (deg/s) | bias (deg) | R2     |
|-------|---------------|------------|--------|
| 1     | 1.4782        | 1.5604     | 0.9909 |
| 2     | 2.9035        | 3.1656     | 0.9980 |
| 3     | 4.5721        | 4.8206     | 0.9945 |
| 4     | 6.1209        | 6.5182     | 0.9990 |
| 5     | 7.8310        | 8.4503     | 0.9986 |

#### Tilt (zoom 0, speeds 1-3)

| Speed | omega (deg/s) | bias (deg) | R2     |
|-------|---------------|------------|--------|
| 1     | 1.9432        | 2.1793     | 0.9989 |
| 2     | 3.7885        | 4.2829     | 0.9964 |
| 3     | 5.7655        | 6.3717     | 0.9955 |

### Reolink 823S2

#### Pan (zoom 0, speeds 1-5)

| Speed | omega (deg/s) | bias (deg) | R2     |
|-------|---------------|------------|--------|
| 1     | 1.4034        | 0.6098     | 0.9900 |
| 2     | 2.5692        | 1.3402     | 0.9846 |
| 3     | 4.1081        | 1.6064     | 0.9926 |
| 4     | 5.7028        | 1.8333     | 0.9972 |
| 5     | 7.1806        | 2.4465     | 0.9973 |

#### Tilt (zoom 0, speeds 1-3)

| Speed | omega (deg/s) | bias (deg) | R2     |
|-------|---------------|------------|--------|
| 1     | 2.0094        | 0.8354     | 0.9764 |
| 2     | 3.7474        | 1.7726     | 0.9911 |
| 3     | 5.2022        | 2.9208     | 0.9727 |

Note: 823S2 has significantly lower bias than 823A16 (0.6-2.9 deg vs 1.6-8.5 deg).

## Micro-pulse

A micro-pulse is a T=0 move (move+stop back-to-back). Its displacement equals the
bias at speed 1. To maximize precision, the camera zooms to 41 before the pulse
(Reolink limits motor speed at high zoom), then restores the original zoom.

| Camera | Axis | Micro-pulse (deg) | Speed 1 bias at z0 (deg) |
|--------|------|-------------------|--------------------------|
| 823A16 | Pan  | 1.60 +/- 0.01     | 1.56                     |
| 823A16 | Tilt | 2.07 +/- 0.12     | 2.18                     |
| 823S2  | Pan  | 0.56 +/- 0.12     | 0.61                     |
| 823S2  | Tilt | 1.08 +/- 0.15     | 0.84                     |

Micro-pulse and bias values are consistent, confirming micro-pulse = bias.

## Zoom-Speed Limitation

Reolink cameras internally limit PTZ speed when zoomed in. A full sweep across
13 zoom levels on the 823A16 (pan axis, speeds 1-5) revealed:

- **zoom 0:** All speeds work at their calibrated values
- **zoom >= 1:** Higher speeds are progressively capped to ~1.5 deg/s

**Decision:** Use multi-speed tables at zoom 0 only. For zoom > 0, use speed 1.

### Effective omega (deg/s) by zoom — 823A16 pan

| Zoom | Speed 1 | Speed 2 | Speed 3 | Speed 4 | Speed 5 |
|------|---------|---------|---------|---------|---------|
| 0    | 1.48    | **2.90**| **4.57**| **6.12**| **7.83**|
| 1    | 1.52    | 1.27    | **2.88**|    —    |    —    |
| 5    | 1.55    | 1.46    | **2.87**| **4.42**|    —    |
| 10   |    —    | 1.51    | 1.45    | **2.94**|    —    |
| 15   |    —    | 1.56    | 1.42    | **2.97**|    —    |
| 20   |    —    | 1.46    | 1.56    | 1.42    |    —    |
| 25+  |    —    | ~1.5    | ~1.5    | ~1.5    |    —    |

**Bold** = faster than the ~1.5 deg/s cap.

The camera has internal speed tiers (~1.45, ~2.95, ~4.50, ~6.15, ~7.85 deg/s).
As zoom increases, each requested speed is shifted down by one or more tiers.

Speed 1 at zoom 41 is only 10% slower than at zoom 0 (omega 1.326 vs 1.478),
so using the zoom 0 value for all zooms is acceptable.

## Production Rules (routes_control.py)

1. **zoom == 0** (patrol): use full speed table, pick highest speed where
   `T = (target - bias) / omega` falls within [0.3s, 4.0s]
2. **zoom > 0** (streaming click-to-move): use speed 1 only
3. **micro-pulse** (angle < bias at speed 1): zoom to 41, fire T=0 move, restore zoom

## Calibration Method

- **Displacement measurement:** ORB keypoints (2000 features), BFMatcher with
  Lowe's ratio test (0.75), median pixel displacement. Manual landmark annotation
  for micro-pulse at zoom 41.
- **FOV conversion:** Measured lookup table per zoom level (chained QR-code method)
- **Server-side timing:** Single API call with `duration` parameter executes
  move+sleep+stop on the Pi's local network, eliminating VPN latency
- **Durations:** 0.25, 0.4, 0.7, 1.2, 2.0s (speeds 2-5); 0.15-1.2s (speed 1 at z41)
- **Settle time:** 3.0s after stop, 3.5s after zoom change for autofocus
