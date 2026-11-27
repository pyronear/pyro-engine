import numpy as np
from PIL import Image

from pyroengine.engine import Engine


class FakeModel:
    """Returns fixed crop-normalized preds and records the crop sizes it is called with."""

    def __init__(self, preds) -> None:
        self.preds = np.array(preds, dtype=np.float32).reshape(-1, 5)
        self.sizes = []

    def __call__(self, img: Image.Image, _occlusion_bboxes: dict | None = None) -> np.ndarray:
        self.sizes.append(img.size)
        return self.preds


def refine(preds, crop_preds, size=(3840, 2160), masks=None):
    engine = Engine.__new__(Engine)  # only the model is needed
    engine.model = FakeModel(crop_preds)
    out = engine._refine_on_crops(Image.new("RGB", size), np.array(preds, dtype=np.float32), masks)
    return out, engine.model.sizes


def test_box_takes_the_best_overlapping_crop_detection():
    # 77x43 px box centred at (1920, 1080): window (1408, 568, 2432, 1592)
    box = [0.49, 0.49, 0.51, 0.51, 0.3]
    crop = [
        [0.45, 0.45, 0.55, 0.55, 0.6],  # overlaps
        [0.48, 0.48, 0.52, 0.52, 0.8],  # overlaps, best
        [0.0, 0.0, 0.05, 0.05, 0.9],  # elsewhere in the crop
    ]
    out, sizes = refine([box], crop)
    assert sizes == [(1024, 1024)]
    np.testing.assert_allclose(
        out,
        [
            [
                (1408 + 0.48 * 1024) / 3840,
                (568 + 0.48 * 1024) / 2160,
                (1408 + 0.52 * 1024) / 3840,
                (568 + 0.52 * 1024) / 2160,
                0.8,
            ]
        ],
        rtol=1e-5,
    )


def test_boxes_resolving_to_the_same_crop_detection_are_merged():
    out, _ = refine([[0.49, 0.49, 0.5, 0.5, 0.3], [0.5, 0.5, 0.51, 0.51, 0.2]], [[0.4, 0.4, 0.6, 0.6, 0.9]])
    assert out.shape == (1, 5)


def test_refined_boxes_on_an_occlusion_mask_are_dropped():
    box, crop = [[0.49, 0.49, 0.51, 0.51, 0.3]], [[0.4, 0.4, 0.6, 0.6, 0.9]]
    assert len(refine(box, crop)[0]) == 1
    out, _ = refine(box, crop, masks={"1": (0.45, 0.45, 0.55, 0.55)})
    assert out.shape == (0, 5)


def test_output_is_sorted_by_confidence():
    large = [0.1, 0.1, 0.2, 0.15, 0.5]  # kept unchanged
    out, _ = refine([large, [0.49, 0.49, 0.51, 0.51, 0.3]], [[0.45, 0.45, 0.55, 0.55, 0.9]])
    assert out[:, 4].tolist() == sorted(out[:, 4].tolist(), reverse=True)


def test_box_without_crop_detection_is_dropped():
    out, _ = refine([[0.49, 0.49, 0.51, 0.51, 0.3]], [[0.0, 0.0, 0.05, 0.05, 0.9]])
    assert out.shape == (0, 5)


def test_window_is_shifted_inside_the_frame():
    # Box in the bottom-right corner: window (2816, 1136, 3840, 2160)
    out, _ = refine([[0.98, 0.98, 1.0, 1.0, 0.3]], [[0.9, 0.9, 1.0, 1.0, 0.5]])
    np.testing.assert_allclose(out[0, :4], [(2816 + 0.9 * 1024) / 3840, (1136 + 0.9 * 1024) / 2160, 1, 1], rtol=1e-5)


def test_window_is_clipped_on_a_small_frame():
    out, sizes = refine([[0.45, 0.45, 0.55, 0.55, 0.3]], [[0.4, 0.4, 0.6, 0.6, 0.5]], size=(1920, 900))
    assert sizes == [(1024, 900)]
    assert out.shape == (1, 5)


def test_large_boxes_and_extra_boxes_are_kept_unchanged():
    large = [0.1, 0.1, 0.2, 0.15, 0.5]  # 384 px wide on 4K
    small = [[0.5, 0.5, 0.51, 0.51, 0.3]] * 6
    out, sizes = refine([large, *small], [])
    # Boxes 2-5 are refined (and dropped: no crop detection), the large one and boxes 6-7 are kept
    assert len(sizes) == 4
    np.testing.assert_allclose(out, np.array([large, small[0]], dtype=np.float32))  # duplicates merged


def test_predict_refines_only_when_enabled_and_the_frame_is_larger(tmp_path):
    calls = []
    for enabled in (True, False):
        engine = Engine(cache_folder=str(tmp_path), frame_size=(720, 1280), crop_refine=enabled)
        engine._refine_on_crops = lambda frame, preds, _masks, enabled=enabled: (
            calls.append((enabled, frame.size)) or preds
        )
        engine.predict(Image.new("RGB", (3840, 2160)))
        engine.predict(Image.new("RGB", (1280, 720)))  # no extra detail on this camera
    assert calls == [(True, (3840, 2160))]
