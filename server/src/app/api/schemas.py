from datetime import datetime
# from fastapi import File, UploadFile
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


class MetricOut(MetricIn):
	pass


# Inference
class InferenceIn(_CreatedAt, _Id):
	#not sure about the format yet
	image: bytes = Field(...)


class InferenceOut(_CreatedAt, _Id):
	pass
