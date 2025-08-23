# Copyright (C) 2020-2025, Pyronear.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

from typing import List

from anonymizer.anonymizer_registry import get_result
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

router = APIRouter()


class BBox(BaseModel):
    cls: str = Field(..., description="class name for example face or plate")
    score: float = Field(..., ge=0.0, le=1.0)
    box: List[float] = Field(..., min_items=4, max_items=4, description="[x1, y1, x2, y2] in pixels")


class LastPrediction(BaseModel):
    camera_ip: str
    timestamp: float
    bboxes: List[BBox]


@router.get("/last_prediction/{camera_ip}", response_model=LastPrediction)
def get_last_prediction(camera_ip: str):
    res = get_result(camera_ip)
    if not res:
        raise HTTPException(status_code=404, detail="No predictions yet for this camera")
    return {
        "camera_ip": camera_ip,
        "timestamp": res["timestamp"],
        "bboxes": res["bboxes"],
    }
