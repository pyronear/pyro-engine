# Copyright (C) 2021, Pyronear contributors.

# This program is licensed under the GNU Affero General Public License version 3.
# See LICENSE or go to <https://www.gnu.org/licenses/agpl-3.0.txt> for full license details.


import torchvision
import torch
from torch import nn
from PIL import Image
from torchvision import transforms


class PyronearPredictor:
    """This class use the last pyronear model to predict if a sigle frame contain a fire or not
    Example
    -------
    pyronearPredictor = PyronearPredictor("model/pyronear.pth")
    res = pyronearPredictor.predict(im)
    print(res) #res[0,1]=fire probability
    """
    def __init__(self, checkpointPath):
        # Model definition
        self.model = torchvision.models.resnet18()

        # Change fc
        in_features = getattr(self.model, 'fc').in_features
        setattr(self.model, 'fc', nn.Linear(in_features, 2))

        self.model.load_state_dict(torch.load(checkpointPath))

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        self.tf = transforms.Compose([transforms.Resize((224, 224)),
                                      transforms.ToTensor(),
                                      normalize])

        self.softmax = nn.Softmax(dim=1)

    def predict(self, im):
        imT = self.tf(im)

        self.model.eval()
        with torch.no_grad():
            pred = self.model(imT.unsqueeze(0))

        return self.softmax(pred).squeeze().numpy()
