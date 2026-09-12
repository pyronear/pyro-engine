# Copyright (C) 2022-2026, Pyronear.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

"""Client for the temporal validation service (pyro_temporal_api)."""

import json
from typing import Any, Dict, List, Sequence, Tuple

import requests

__all__ = ["TemporalClient"]

# (frame_id, jpeg bytes, boxes as [[x1, y1, x2, y2, conf], ...] normalized xyxy)
WindowFrame = Tuple[str, bytes, List[List[float]]]


class TemporalClient:
    """Submit a window of frames for one camera pose, then poll the verdict on a later round."""

    def __init__(self, url: str, timeout: float = 10.0) -> None:
        self.url = url.rstrip("/")
        self.timeout = timeout

    def submit(self, cam_id: str, window: Sequence[WindowFrame]) -> str:
        files = [("frames", (f"{frame_id}.jpg", data, "image/jpeg")) for frame_id, data, _ in window]
        boxes = [frame_boxes for _, _, frame_boxes in window]
        response = requests.post(
            f"{self.url}/jobs",
            data={"cam_id": cam_id, "boxes": json.dumps(boxes)},
            files=files,
            timeout=self.timeout,
        )
        response.raise_for_status()
        return str(response.json()["job_id"])

    def result(self, job_id: str) -> Dict[str, Any]:
        """Return the job dict: ``status`` is ``pending``, ``done`` (with ``verdict``) or ``error``."""
        response = requests.get(f"{self.url}/jobs/{job_id}", timeout=self.timeout)
        response.raise_for_status()
        return dict(response.json())
