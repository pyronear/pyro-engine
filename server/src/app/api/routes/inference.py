from fastapi import APIRouter, BackgroundTasks
from fastapi import File, UploadFile

from app.api.schemas import InferenceOut

router = APIRouter()


def predict_and_alert(file):
    """ predict smoke from an image and if yes raise an alert """

    # TODO : use inference class and pyro-api client to raise an alert if needed
    return {"comment": "set fire to the rain please"}


@router.post("/", response_model=InferenceOut, status_code=201, summary="Send img from a device to predict smoke")
async def inference(background_tasks: BackgroundTasks,
                    file: UploadFile = File(...)
                    ):
    """Send img from a device based on the given information in order to predict smoke

    Below, click on "Schema" for more detailed information about arguments
    or "Example Value" to get a concrete idea of arguments
    """
    # some async operation could happen here

    # We might need a backround task here for inference ?
    background_tasks.add_task(predict_and_alert, file)

    return {"id": 12}
