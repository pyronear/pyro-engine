# Copyright (C) 2020-2025, Pyronear.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.
# Copyright (C) 2020-2025, Pyronear.
# Apache 2.0

from typing import List, Tuple

from anonymizer.anonymizer_registry import get_result
from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()

# Each bbox is a fixed length list: [x1, y1, x2, y2, score]
BBoxTuple = Tuple[float, float, float, float, float]


class LastPrediction(BaseModel):
    camera_ip: str
    timestamp: float
    bboxes: List[BBoxTuple]


@router.get("/last_prediction/{camera_ip}", response_model=LastPrediction)
def get_last_prediction(camera_ip: str):
    res = get_result(camera_ip)
    if not res:
        return {
            "camera_ip": camera_ip,
            "timestamp": 0.0,
            "bboxes": [],
        }
    return {
        "camera_ip": camera_ip,
        "timestamp": res["timestamp"],
        "bboxes": [tuple(bb) for bb in res["bboxes"]],
    }
