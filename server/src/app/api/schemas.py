# Copyright (C) 2021, Pyronear contributors.

# This program is licensed under the GNU Affero General Public License version 3.
# See LICENSE or go to <https://www.gnu.org/licenses/agpl-3.0.txt> for full license details.


from datetime import datetime
from pydantic import BaseModel, Field, validator

from typing import Optional


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
class MetricIn(_CreatedAt):
    cpu_temperature_C: float = Field(None, example=17.3)
    mem_available_GB: float = Field(None, example=1.23)
    cpu_usage_percent: float = Field(None, example=51.8)


class MetricOut(MetricIn, _CreatedAt):
    pass


class Token(BaseModel):
    access_token: str = Field(..., example="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.423fgFGTfttrvU6D1k7vF92hH5vaJHCGFYd8E")
    token_type: str = Field(..., example="bearer")


class TokenData(BaseModel):
    username: Optional[str] = None


class Device(BaseModel):
    username: str
    disabled: Optional[bool] = None


class DeviceInDB(Device):
    hashed_password: str
