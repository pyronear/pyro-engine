# Copyright (C) 2019-2022, Pyronear.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

from pyrovision.models.rexnet import rexnet1_0x
from torchvision import transforms
import torch


class PyronearPredictor:
    """This class use the last pyronear model to predict if a sigle frame contain a fire or not
    Examples:
        >>> pyronearPredictor = PyronearPredictor()
        >>> im = Image.open("image.jpg")
        >>> res = pyronearPredictor.predict(im)
    """
    def __init__(self):
        """Init predictor"""
        # Model definition
        self.model = rexnet1_0x(pretrained=True).eval()

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        img_size = 448

        self.tf = transforms.Compose([
            transforms.Resize(size=img_size),
            transforms.CenterCrop(size=img_size),
            transforms.ToTensor(),
            normalize
        ])

    def predict(self, im):
        """Run prediction"""
        imT = self.tf(im)

        with torch.no_grad():
            pred = self.model(imT.unsqueeze(0))
            pred = torch.sigmoid(pred).item()

        return pred
