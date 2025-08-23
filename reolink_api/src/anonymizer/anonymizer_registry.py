# Copyright (C) 2020-2025, Pyronear.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

import threading
import time
from typing import Dict, List, Optional, TypedDict


class BBox(TypedDict):
    cls: str
    score: float
    box: List[float]  # [x1, y1, x2, y2] in pixels


class AnonResult(TypedDict, total=False):
    timestamp: float
    bboxes: List[BBox]


# Latest prediction per camera
ANON_RESULTS: Dict[str, AnonResult] = {}
# Per camera read write lock
ANON_LOCKS: Dict[str, threading.Lock] = {}

# Thread and flag per camera while stream is active
ANON_THREADS: Dict[str, threading.Thread] = {}
ANON_FLAGS: Dict[str, threading.Event] = {}


def set_result(camera_ip: str, bboxes: List[BBox]) -> None:
    if camera_ip not in ANON_LOCKS:
        ANON_LOCKS[camera_ip] = threading.Lock()
    with ANON_LOCKS[camera_ip]:
        ANON_RESULTS[camera_ip] = {"timestamp": time.time(), "bboxes": bboxes}


def get_result(camera_ip: str) -> Optional[AnonResult]:
    lock = ANON_LOCKS.get(camera_ip)
    if lock is None:
        return None
    with lock:
        return ANON_RESULTS.get(camera_ip)
