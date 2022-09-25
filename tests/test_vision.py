import numpy as np
from huggingface_hub import hf_hub_download

from pyroengine.vision import Classifier


def test_classifier(mock_wildfire_image):

    # Instantiate the ONNX model
    model = Classifier("pyronear/rexnet1_0x", revision="983c33f0b1aacde134e934e91550e6bd8651b31c")
    # Check preprocessing
    out = model.preprocess_image(mock_wildfire_image)
    assert isinstance(out, np.ndarray) and out.dtype == np.float32
    assert out.shape == (1, 3, 256, 384)
    # Check inference
    out = model(mock_wildfire_image)
    assert isinstance(out, np.ndarray) and out.dtype == np.float32
    assert out.shape == (1,)
    assert out >= 0 and out <= 1

    # Load static file
    cfg_path = hf_hub_download("pyronear/rexnet1_0x", filename="config.json")
    model_path = hf_hub_download("pyronear/rexnet1_0x", filename="model.onnx")
    Classifier("pyronear/rexnet1_0x", cfg_path=cfg_path, model_path=model_path)
