# Copyright (C) 2022-2026, Pyronear.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.


from io import BytesIO
from unittest.mock import MagicMock, patch

from PIL import Image
from pyro_camera_api_client.client import PyroCameraAPIClient


def test_client_exposes_capture_time():
    buffer = BytesIO()
    Image.new("RGB", (8, 8)).save(buffer, format="JPEG")
    response = MagicMock(
        status_code=200, content=buffer.getvalue(), headers={"X-Captured-At": "2026-10-30T08:40:51+00:00"}
    )
    client = PyroCameraAPIClient("http://fake.url")

    with patch.object(client, "_request", return_value=response):
        image = client.get_latest_image("cam", 8)

    assert image.info["captured_at"] == "2026-10-30T08:40:51+00:00"
