import numpy as np
from huggingface_hub import hf_hub_download

from pyroengine.engine.vision import Classifier


def test_classifier(mock_classification_image):

    # Instantiae the ONNX model
    model = Classifier("pyronear/rexnet1_3x")
    # Check preprocessing
    out = model.preprocess_image(mock_classification_image)
    assert isinstance(out, np.ndarray) and out.dtype == np.float32
    assert out.shape == (1, 3, 224, 224)
    # Check inference
    out = model(mock_classification_image)
    assert isinstance(out, np.ndarray) and out.dtype == np.float32
    assert out.shape == (1,)
    assert out >= 0 and out <= 1

    # Load static file
    cfg_path = hf_hub_download("pyronear/rexnet1_3x", filename="config.json")
    model_path = hf_hub_download("pyronear/rexnet1_3x", filename="model.onnx")
    Classifier("pyronear/rexnet1_3x", cfg_path=cfg_path, model_path=model_path)
