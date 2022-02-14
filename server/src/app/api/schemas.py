# Copyright (C) 2020-2022, Pyronear.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.


from datetime import datetime
from pydantic import BaseModel, Field, validator


# Template classes
class _CreatedAt(BaseModel):
    created_at: datetime = None

    @staticmethod
    @validator('created_at', pre=True, always=True)
    def default_ts_created(v):
        return v or datetime.utcnow()


class _Id(BaseModel):
    id: int = Field(..., gt=0)


# Metrics
class MetricIn(_CreatedAt, _Id):
    cpu_temperature_C: float = Field(None, example=17.3)
    mem_available_GB: float = Field(None, example=1.23)
    cpu_usage_percent: float = Field(None, example=51.8)


class MetricOut(MetricIn, _CreatedAt):
    pass
