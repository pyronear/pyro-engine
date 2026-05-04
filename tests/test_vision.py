import hashlib
import pathlib
import shutil

import numpy as np

# Canonical import — Classifier lives in pyro_predictor
from pyro_predictor import Classifier
from pyro_predictor.vision import MODEL_CACHE_SUBDIR, MODEL_SLUG

# pyroengine.vision shim must re-export the same class
from pyroengine.vision import Classifier as ClassifierShim


def test_shim_is_same_class():
    """pyroengine.vision.Classifier must be the same object as pyro_predictor.Classifier."""
    assert ClassifierShim is Classifier


def test_classifier(tmpdir_factory, mock_wildfire_image):
    folder = str(tmpdir_factory.mktemp("engine_cache"))

    # Instantiate the ONNX model
    model = Classifier(model_folder=folder)
    # Check inference
    out = model(mock_wildfire_image)
    assert out.shape[1] == 5
    conf = np.max(out[:, 4])
    assert 0 <= conf <= 1

    # Test onnx model
    model = Classifier(model_folder=folder, format="onnx")
    model_path = str(pathlib.Path(folder) / MODEL_CACHE_SUBDIR / MODEL_SLUG / "onnx_cpu" / "best.onnx")
    assert pathlib.Path(model_path).is_file()

    # Test occlusion mask
    out = model(mock_wildfire_image, {})
    assert out.shape == (1, 5)

    occlusion_bboxes = {"2025-05-28 15:49:17": [0.2, 0.3, 0.4, 0.5]}
    out = model(mock_wildfire_image, occlusion_bboxes)
    assert out.shape == (1, 5)

    occlusion_bboxes = {"2025-05-28 15:49:17": [0.00621796, 0.5494792, 0.02899933, 0.6085069]}
    out = model(mock_wildfire_image, occlusion_bboxes)
    assert out.shape == (0, 5)


def sha256sum(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def test_stale_cache_is_purged(tmpdir_factory):
    folder = pathlib.Path(tmpdir_factory.mktemp("engine_cache"))
    models_dir = folder / MODEL_CACHE_SUBDIR

    # Seed a previous-version slug alongside the current one.
    stale_slug = models_dir / "yolo11s_stale-slug_v0.0.0"
    stale_slug.mkdir(parents=True)
    (stale_slug / "marker.txt").write_text("stale")

    # Seed the pre-slug flat layout (onnx variant, matching the format used below).
    legacy_archive = folder / "onnx_cpu.tar.gz"
    legacy_archive.write_bytes(b"legacy")
    legacy_extract = folder / "onnx_cpu"
    legacy_extract.mkdir()
    (legacy_extract / "marker.txt").write_text("legacy")

    _ = Classifier(model_folder=str(folder), format="onnx")

    assert not stale_slug.exists(), "stale slug should have been purged"
    assert not legacy_archive.exists(), "legacy archive should have been purged"
    assert not legacy_extract.exists(), "legacy extract dir should have been purged"
    assert (models_dir / MODEL_SLUG / "onnx_cpu" / "best.onnx").is_file()


def test_download(tmpdir_factory):
    folder = str(tmpdir_factory.mktemp("engine_cache"))

    # First download
    _ = Classifier(model_folder=folder, format="onnx")
    model_path = str(pathlib.Path(folder) / MODEL_CACHE_SUBDIR / MODEL_SLUG / "onnx_cpu" / "best.onnx")
    assert pathlib.Path(model_path).is_file()

    hash1 = sha256sum(model_path)

    # Delete and download again
    pathlib.Path(model_path).unlink()
    shutil.rmtree(pathlib.Path(model_path).parent, ignore_errors=True)
    _ = Classifier(model_folder=folder, format="onnx")

    hash2 = sha256sum(model_path)

    # Test that the model was re-downloaded (at least once more)
    assert hash1 == hash2  # optional if content is static
    assert pathlib.Path(model_path).exists()
