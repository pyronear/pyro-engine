from fastapi import APIRouter, BackgroundTasks

from app.api.schemas import InferenceIn, InferenceOut

router = APIRouter()


def predict_and_alert(InferenceIn):
	""" predict smoke from an image and if yes raise an alert """

	# TODO : use inference class and pyro-api client to raise an alert if needed
	return {"comment": "set fire to the rain please"}


@router.post("/", response_model=InferenceOut, status_code=201, summary="Send img from a device to predict smoke")
async def inference(payload: InferenceIn, background_tasks: BackgroundTasks):
    """Send img from a device based on the given information in order to predict smoke

    Below, click on "Schema" for more detailed information about arguments
    or "Example Value" to get a concrete idea of arguments
    """
    # some async operation could happen here
    # the idea it to send metrics to our "log data plateform"

    # Do we need a backround task here for inference ?
    background_tasks.add_task(predict_and_alert, payload)

    return {"info" : "take some more pic please"}