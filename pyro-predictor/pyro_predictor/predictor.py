# Copyright (C) 2022-2026, Pyronear.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

import logging
from collections import deque
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import numpy.typing as npt
from PIL import Image

from .utils import box_iou, nms
from .vision import Classifier

__all__ = ["Predictor"]

logger = logging.getLogger(__name__)


class Predictor:
    """Wildfire detection predictor: runs model inference and maintains per-camera sliding-window state.

    This class is self-contained and has no dependency on external services (no pyroclient, no HTTP calls).
    It can be used standalone for offline inference or embedded in a larger system like Engine.

    Args:
        model_path: path to an ONNX model file; if None, the default NCNN model is downloaded
        conf_thresh: confidence threshold above which an alert is considered active
        model_conf_thresh: per-frame confidence threshold passed to the YOLO model
        max_bbox_size: discard detections wider than this fraction of the image
        nb_consecutive_frames: sliding-window size for temporal smoothing
        frame_size: if set, resize each frame to (H, W) before inference
        cam_ids: list of camera IDs to pre-initialise state for
        verbose: if False, suppress all informational log output (default True)
        kwargs: forwarded to Classifier

    Examples:
        >>> from pyro_predictor import Predictor
        >>> predictor = Predictor()
        >>> conf = predictor.predict(pil_image, cam_id="192.168.1.10_0")
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        conf_thresh: float = 0.15,
        model_conf_thresh: float = 0.05,
        max_bbox_size: float = 0.4,
        nb_consecutive_frames: int = 8,
        frame_size: Optional[Tuple[int, int]] = None,
        cam_ids: Optional[List[str]] = None,
        verbose: bool = True,
        **kwargs: Any,
    ) -> None:
        self.verbose = verbose
        self.model = Classifier(
            model_path=model_path, conf=model_conf_thresh, max_bbox_size=max_bbox_size, verbose=verbose, **kwargs
        )
        self.conf_thresh = conf_thresh
        self.model_conf_thresh = model_conf_thresh
        self.max_bbox_size = max_bbox_size
        self.nb_consecutive_frames = nb_consecutive_frames
        self.frame_size = frame_size

        self._states: Dict[str, Dict[str, Any]] = {"-1": self._new_state()}
        if cam_ids:
            for cam_id in cam_ids:
                self._states[cam_id] = self._new_state()

    def _new_state(self) -> Dict[str, Any]:
        # Window holds nb_consecutive_frames - 1 past frames; pool = current + window = nb total.
        return {
            "last_predictions": deque(maxlen=self.nb_consecutive_frames - 1),
            "ongoing": False,
        }

    def _update_states(
        self,
        frame: Image.Image,
        preds: np.ndarray,
        cam_key: str,
        encoded_bytes: Optional[bytes] = None,
    ) -> float:
        nb = self.nb_consecutive_frames

        # Pool = current preds + every past frame's raw preds in the sliding window.
        pool = np.zeros((0, 5), dtype=np.float64)
        pool = np.concatenate([pool, preds])
        for _, box, _, _, _, _ in self._states[cam_key]["last_predictions"]:
            if box.shape[0] > 0:
                pool = np.concatenate([pool, box])

        conf = 0.0
        output_predictions: npt.NDArray[np.float64] = np.zeros((0, 5), dtype=np.float64)

        if pool.shape[0]:
            candidates = nms(pool)
            # box_iou(A, B) returns shape (len(B), len(A)); call with (pool, candidates) -> (n_cand, n_pool)
            ious_pool = box_iou(pool[:, :4], candidates[:, :4])
            overlap = ious_pool > 0  # (n_cand, n_pool)
            counts = overlap.sum(axis=1)
            sums = (overlap * pool[:, 4]).sum(axis=1)
            combine_conf = sums / nb

            valid_mask = (counts >= (nb // 2)) & (combine_conf > self.conf_thresh)
            valid_candidates = candidates[valid_mask]
            valid_conf = combine_conf[valid_mask]

            if valid_conf.size > 0:
                conf = float(valid_conf.max())

            if valid_candidates.shape[0] and preds.shape[0]:
                # ious_preds shape: (n_valid, n_preds)
                ious_preds = box_iou(preds[:, :4], valid_candidates[:, :4])
                overlap_preds = ious_preds > 0
                has_match = overlap_preds.any(axis=0)
                if has_match.any():
                    matched_cand = overlap_preds.argmax(axis=0)
                    rows = []
                    for p_idx in np.where(has_match)[0]:
                        c_idx = int(matched_cand[p_idx])
                        x1, y1, x2, y2 = preds[p_idx, :4]
                        rows.append([x1, y1, x2, y2, float(valid_conf[c_idx])])
                    output_predictions = np.round(np.array(rows, dtype=np.float64), 3)

        self._states[cam_key]["last_predictions"].append((
            frame,
            preds,
            output_predictions.tolist(),
            datetime.now(timezone.utc).isoformat(),
            False,
            encoded_bytes,
        ))
        self._states[cam_key]["ongoing"] = conf > self.conf_thresh
        return conf

    def predict(
        self,
        frame: Image.Image,
        cam_id: Optional[str] = None,
        occlusion_bboxes: Optional[Dict[Any, Any]] = None,
        fake_pred: Optional[np.ndarray] = None,
    ) -> float:
        """Run inference on a frame and return the aggregated wildfire confidence score.

        Args:
            frame: input PIL image
            cam_id: camera identifier; uses a default slot when None
            occlusion_bboxes: dict of occlusion bounding boxes to suppress detections
            fake_pred: bypass model inference with a pre-computed raw prediction array (for evaluation)

        Returns:
            confidence score in [0, 1]
        """
        cam_key = cam_id or "-1"
        if cam_key not in self._states:
            self._states[cam_key] = self._new_state()

        if isinstance(self.frame_size, tuple):
            target = (self.frame_size[1], self.frame_size[0])  # PIL expects (W, H)
            if frame.size != target:
                frame = frame.resize(target, Image.BILINEAR)  # type: ignore[attr-defined]

        if fake_pred is None:
            preds = self.model(frame.convert("RGB"), occlusion_bboxes or {})
        else:
            if fake_pred.size == 0:
                preds = np.empty((0, 5))
            else:
                preds = self.model.post_process(fake_pred, pad=(0, 0))
                preds = preds[(preds[:, 2] - preds[:, 0]) < self.max_bbox_size, :]
                preds = np.reshape(preds, (-1, 5))

        if self.verbose:
            logger.info(f"pred for {cam_key} : {preds}")
        conf = self._update_states(frame, preds, cam_key)

        if self.verbose:
            device_str = f"Camera '{cam_id}' - " if isinstance(cam_id, str) else ""
            pred_str = "Wildfire detected" if conf > self.conf_thresh else "No wildfire"
            logger.info(f"{device_str}{pred_str} (confidence: {conf:.2%})")

        return float(conf)
