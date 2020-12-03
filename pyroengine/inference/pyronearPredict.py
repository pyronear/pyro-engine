import torchvision
import torch
from torch import nn
from PIL import Image
from torchvision import transforms
from pyroengine.configManager import read_config_file
from pyrovision.models.utils import cnn_model
import holocron


class PyronearPredictor:
    """This class use the last pyronear model to predict if a sigle frame contain a fire or not
    Example
    -------
    pyronearPredictor = PyronearPredictor("model/pyronear.pth")
    res = pyronearPredictor.predict(im)
    print(res) #res=fire probability
    """
    def __init__(self, configFile, checkpointPath=None):

        # Load config file
        self.config = read_config_file(configFile)
        if checkpointPath:
            self.config['checkpoint'] = checkpointPath

        # Model definition
        self.model = self.get_model()

        # Transform definition
        self.transforms = self.get_transforms()

        self.sigmoid = nn.Sigmoid()

    def get_transforms(self):
        """Transforms definition"""
        size = self.config['imageSize']
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        if self.config['use_CenterCrop']:
            tf = transforms.Compose([transforms.Resize(size=(size)),
                                     transforms.CenterCrop(size=size),
                                     transforms.ToTensor(),
                                     normalize
                                     ])
        else:
            tf = transforms.Compose([transforms.Resize(size=(size)),
                                     transforms.ToTensor(),
                                     normalize
                                     ])

        return tf

    def get_model(self):
        """Model definition"""
        # Get backbone
        base = holocron.models.__dict__[self.config['backbone']](False, num_classes=self.config['num_classes'])
        # Change head
        if self.config['nb_features']:
            model = cnn_model(base, self.config['cut'], nb_features=self.config['nb_features'],
                              num_classes=self.config['num_classes'])
        else:
            model = base
        # Load Weight
        model.load_state_dict(torch.load(self.config['checkpoint'], map_location=torch.device('cpu')))
        # Move to gpu
        if self.config['device'] == 'cuda' and torch.cuda.is_available():
            model = model.to('cuda:0')

        return model.eval()

    def predict(self, im):
        # Get Data
        im = self.transforms(im)
        if self.config['device'] == 'cuda' and torch.cuda.is_available():
            im = im.to('cuda:0')
        im = im.unsqueeze(0)
        # Predict
        with torch.no_grad():
            res = self.model(im)
            res = self.sigmoid(res)

        return res.item()
