import pytest
from requests.exceptions import ConnectTimeout

from pyroengine.sensors import ReolinkCamera


def test_reolinkcamera(mock_wildfire_image):

    with pytest.raises(ConnectTimeout):
        ReolinkCamera("192.168.1.1", "login", "pwd")
