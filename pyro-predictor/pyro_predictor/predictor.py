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
        return {
            "last_predictions": deque(maxlen=self.nb_consecutive_frames),
            "ongoing": False,
            "anchor_bbox": None,
            "anchor_ts": None,
            "miss_count": 0,
        }

    def _update_states(
        self,
        frame: Image.Image,
        preds: np.ndarray,
        cam_key: str,
        encoded_bytes: Optional[bytes] = None,
    ) -> float:
        prev_ongoing = self._states[cam_key]["ongoing"]

        conf_th = self.conf_thresh * self.nb_consecutive_frames
        if prev_ongoing:
            conf_th *= 0.8

        boxes = np.zeros((0, 5), dtype=np.float64)
        boxes = np.concatenate([boxes, preds])
        for _, box, _, _, _, _ in self._states[cam_key]["last_predictions"]:
            if box.shape[0] > 0:
                boxes = np.concatenate([boxes, box])

        conf = 0.0
        output_predictions: npt.NDArray[np.float64] = np.zeros((0, 5), dtype=np.float64)

        if boxes.shape[0]:
            best_boxes = nms(boxes)
            detections = boxes[boxes[:, -1] > self.conf_thresh, :]
            ious_detections = box_iou(best_boxes[:, :4], detections[:, :4])
            strong_detection = np.sum(ious_detections > 0, axis=0) >= int(self.nb_consecutive_frames / 2)
            best_boxes = best_boxes[strong_detection, :]

            if best_boxes.shape[0]:
                ious = box_iou(best_boxes[:, :4], boxes[:, :4])
                best_boxes_scores = np.array([sum(boxes[iou > 0, 4]) for iou in ious.T])
                combine_predictions = best_boxes[best_boxes_scores > conf_th, :]
                if len(best_boxes_scores) > 0:
                    conf = np.max(best_boxes_scores) / (self.nb_consecutive_frames + 1)

                if combine_predictions.shape[0] > 0:
                    ious = box_iou(combine_predictions[:, :4], preds[:, :4])
                    iou_match = np.array([np.max(iou) > 0 for iou in ious], dtype=bool)
                    matched_preds = preds[iou_match, :]
                    if matched_preds.ndim == 1:
                        matched_preds = matched_preds[np.newaxis, :]
                    output_predictions = matched_preds.astype(np.float64)

        # no zero confidence fabrication before ongoing
        # if empty and we were already ongoing, reuse anchor but set conf to 0
        if output_predictions.shape[0] == 0:
            anchor = self._states[cam_key]["anchor_bbox"]
            if prev_ongoing and anchor is not None:
                output_predictions = anchor.copy()
                output_predictions[:, -1] = 0.0  # filled during ongoing, confidence forced to 0
            else:
                output_predictions = np.empty((0, 5), dtype=np.float64)  # stays empty for backfill later
        else:
            # refresh anchor during ongoing with light smoothing
            if prev_ongoing:
                best_idx = int(np.argmax(output_predictions[:, 4]))
                best = output_predictions[best_idx : best_idx + 1]
                anchor = self._states[cam_key]["anchor_bbox"]
                if anchor is None:
                    self._states[cam_key]["anchor_bbox"] = best.copy()
                else:
                    alpha = 0.3
                    self._states[cam_key]["anchor_bbox"] = alpha * best + (1.0 - alpha) * anchor
                self._states[cam_key]["miss_count"] = 0

        output_predictions = np.round(output_predictions, 3)
        output_predictions = output_predictions[:5, :]
        if output_predictions.size > 0:
            output_predictions = np.atleast_2d(output_predictions)

        self._states[cam_key]["last_predictions"].append((
            frame,
            preds,
            output_predictions.tolist(),  # [] if empty
            datetime.now(timezone.utc).isoformat(),
            False,
            encoded_bytes,
        ))

        new_ongoing = conf > self.conf_thresh
        if prev_ongoing and not new_ongoing:
            self._states[cam_key]["anchor_bbox"] = None
            self._states[cam_key]["anchor_ts"] = None
            self._states[cam_key]["miss_count"] = 0
        elif not prev_ongoing and new_ongoing:
            if output_predictions.size > 0:
                self._states[cam_key]["anchor_bbox"] = output_predictions.copy()
                self._states[cam_key]["miss_count"] = 0

        self._states[cam_key]["ongoing"] = new_ongoing
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
