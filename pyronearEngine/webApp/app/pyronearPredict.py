import torchvision
import torch
from torch import nn
from PIL import Image
from torchvision import transforms


class PyronearPredictor:
    """This class use the last pyronear model to predict if a sigle frame contain a fire or not
    Example
    -------
    pyronearPredictor = PyronearPredictor()
    res = pyronearPredictor.predict(im)
    print(res)
    """
    def __init__(self):
        # Model definition
        self.model = torchvision.models.resnet18()

        # Change fc
        in_features = getattr(self.model, 'fc').in_features
        setattr(self.model, 'fc', nn.Linear(in_features, 2))

        self.model.load_state_dict(torch.load("model/model_resnet18.txt",map_location=torch.device('cpu')))

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

        if pred[0, 0] > pred[0, 1]:
            return f'Prediction: no fire; Probability: ' + str(100000*self.softmax(pred)[0, 0].item()//100/1000)
        else:
            return f'Prediction: fire Probability: ' + str(100000*self.softmax(pred)[0, 1].item()//100/1000)