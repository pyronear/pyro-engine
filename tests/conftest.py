from io import BytesIO

import pytest
import requests
from PIL import Image


@pytest.fixture(scope="session")
def mock_classification_image(tmpdir_factory):
    url = "https://github.com/pyronear/pyro-vision/releases/download/v0.1.2/fire_sample_image.jpg"
    return Image.open(BytesIO(requests.get(url).content))
