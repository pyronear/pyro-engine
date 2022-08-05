from io import BytesIO

import pytest
import requests
from PIL import Image


@pytest.fixture(scope="session")
def mock_image_stream(tmpdir_factory):
    url = "https://github.com/pyronear/pyro-vision/releases/download/v0.1.2/fire_sample_image.jpg"
    return requests.get(url).content


@pytest.fixture(scope="session")
def mock_image_content(tmpdir_factory, mock_image_stream):
    return Image.open(BytesIO(mock_image_stream))
