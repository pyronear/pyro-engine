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
	# multipart/form-data ? 
	# https://pydantic-docs.helpmanual.io/usage/types/#pydantic-types. ---> PyObject
	# create custom data type ? https://pydantic-docs.helpmanual.io/usage/types/#custom-data-types
	# image: UploadFile = File(...)
	# image_files: UploadFile = File(..., media_type='image/jpeg')
	# or   https://github.com/tiangolo/fastapi/issues/2257#issuecomment-717924054
	# --->   image_files: Optional[List[UploadFile]] = File(None, media_type='image/jpeg'),
	# https://github.com/tiangolo/fastapi/issues/2257

	# ----> Gros boloss, on va mettre un nom d'image, et encore, mais en fait, jsute appeler dans la route InferenceIn. & un file !!  File(None, media_type='image/jpeg'),
	image: bytes = Field(...)


class InferenceOut(_CreatedAt, _Id):
	pass
