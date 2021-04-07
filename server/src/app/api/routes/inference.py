# Copyright (C) 2021, Pyronear contributors.

# This program is licensed under the GNU Affero General Public License version 3.
# See LICENSE or go to <https://www.gnu.org/licenses/agpl-3.0.txt> for full license details.


from fastapi import APIRouter, BackgroundTasks
from fastapi import File, UploadFile

router = APIRouter()


def predict_and_alert(file):
    """ predict smoke from an image and if yes raise an alert """
    # TODO : use inference class and pyro-api client to raise an alert if needed
    return {"comment": "set fire to the rain please"}


@router.post("/", status_code=201, summary="Send img from a device to predict smoke")
async def inference(background_tasks: BackgroundTasks,
                    file: UploadFile = File(...)
                    ):
    """
    Send img from a device based on the given information in order to predict smoke

    Below, click on "Schema" for more detailed information about arguments
    or "Example Value" to get a concrete idea of arguments
    """
    # some async operation could happen here

    # We might need a backround task here for inference
    background_tasks.add_task(predict_and_alert, file)
