# Copyright (C) 2022, Pyronear.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

import onnxruntime as ort
import numpy as np
import torch
from pyroengine.models.utils import non_max_suppression, xywh2xyxy


__all__ = ["Yolo_v5"]


class Yolo_v5:
    def __init__(self, model_weights, conf_thres=0.25):
        """Yolov5 onnx instance

        Args:
            model_weights (str): path to onnx file
            conf_thres (float, optional): confidence threshold. Defaults to 0.25.
        """
        self.session = ort.InferenceSession(
            model_weights, providers=["CPUExecutionProvider"]
        )

        self.conf_thres = conf_thres

    def forward(self, im):
        """Run prediction

        Args:
            im (Pillow): Image to analyse

        Returns:
            np.array: prediction with [x1, y1, x2, y2, conf, class] where xy1=top-left, xy2=bottom-right
        """
        im = im.resize((640, 640))
        x = np.expand_dims(np.array(im).astype("float"), axis=0)
        x = np.ascontiguousarray(x.transpose((0, 3, 1, 2)))  # BHWC to BCHW
        x = torch.from_numpy(x).float() / 255  # uint8 to fp16/32

        y = self.session.run(["output"], {"images": x.numpy()})[0]

        output = non_max_suppression(torch.tensor(y), self.conf_thres)[0].numpy()

        return output
