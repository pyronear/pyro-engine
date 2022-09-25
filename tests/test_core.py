import pytest

from pyroengine.core import SystemController
from pyroengine.engine import Engine


def test_systemcontroller(tmpdir_factory, mock_wildfire_image):

    # Cache
    folder = str(tmpdir_factory.mktemp("engine_cache"))

    engine = Engine("pyronear/rexnet1_3x", cache_folder=folder)
    cams = []
    controller = SystemController(engine, cams)

    with pytest.raises(AssertionError):
        controller.analyze_stream(0)

    assert len(repr(controller).split("\n")) == 2
