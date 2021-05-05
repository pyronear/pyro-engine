# Copyright (C) 2021, Pyronear contributors.

# This program is licensed under the GNU Affero General Public License version 3.
# See LICENSE or go to <https://www.gnu.org/licenses/agpl-3.0.txt> for full license details.


import os
import io
from pathlib import Path
from datetime import datetime

from fastapi import APIRouter, BackgroundTasks
from fastapi import File, UploadFile
from pyroengine.engine import PyronearEngine
from PIL import Image


router = APIRouter()
engine = PyronearEngine()  # need to add api setup parameters here


def write_image(file):
    """Write image to $IMAGE_FOLDER (or /tmp) as image_YYYY-MM-DD_hh_mm_ss.jpg"""
    image = Image.open(io.BytesIO(file))  # Load Image
    now = datetime.now().strftime('%Y-%m-%d_%H_%M_%S')
    # FIXME: add device to filename when it becomes available
    fname = Path(os.getenv('IMAGE_FOLDER', '/tmp/')) / f'{now}.jpg'
    image.save(fname)


@router.post("/file/", status_code=201, summary="Write image from a device")
async def inference(background_tasks: BackgroundTasks,
                    file: UploadFile = File(...)
                    ):
    """
    Get image from pizero and write it to $IMAGE_FOLDER
    """
    # Call write_image as background task
    background_tasks.add_task(write_image, await file.read())
