# Copyright (C) 2021, Pyronear contributors.

# This program is licensed under the GNU Affero General Public License version 3.
# See LICENSE or go to <https://www.gnu.org/licenses/agpl-3.0.txt> for full license details.


from fastapi import APIRouter

from app.api.schemas import MetricIn, MetricOut

router = APIRouter()


@router.post("/", response_model=MetricOut, status_code=201, summary="Send metrics from a device")
async def log_metrics(payload: MetricIn):
    """
    Send metrics from a device based on the given information

    Below, click on "Schema" for more detailed information about arguments
    or "Example Value" to get a concrete idea of arguments
    """
    # some async operation could happen here
    # the idea it to send metrics to our "log data plateform"
    return {**payload.dict()}
