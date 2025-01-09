from io import BytesIO

import pytest
import requests
from PIL import Image


@pytest.fixture(scope="session")
def mock_wildfire_stream(tmpdir_factory):
    url = "https://github.com/pyronear/pyro-engine/releases/download/v0.1.1/fire_sample_image.jpg"
    return requests.get(url).content


@pytest.fixture(scope="session")
def mock_wildfire_image(tmpdir_factory, mock_wildfire_stream):
    return Image.open(BytesIO(mock_wildfire_stream))


@pytest.fixture(scope="session")
def mock_forest_stream(tmpdir_factory):
    url = "https://github.com/pyronear/pyro-engine/releases/download/v0.1.1/forest_sample.jpg"
    return requests.get(url).content


@pytest.fixture(scope="session")
def mock_forest_image(tmpdir_factory, mock_forest_stream):
    return Image.open(BytesIO(mock_forest_stream))
