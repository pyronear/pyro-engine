from fastapi import APIRouter

from app.api.schemas import MetricIn, MetricOut

router = APIRouter()


@router.post("/", response_model=MetricOut, status_code=201, summary="Send metrics from a device")
async def send_metrics(payload: MetricIn):
    """Send metrics from a device based on the given information

    Below, click on "Schema" for more detailed information about arguments
    or "Example Value" to get a concrete idea of arguments
    """
    # some async operation could happen here
    # the idea it to send metrics to our "log data plateform"
    return {**payload.dict()}
