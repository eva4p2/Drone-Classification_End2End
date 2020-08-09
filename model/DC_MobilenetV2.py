import torch
import torch.nn.functional as F
import torch.nn as nn
from BaseModel import *

class DroneMobilenetV2(ClassificationBase):
    def __init__(self):
        super().__init__()
        # Use a pretrained MobilenetV2 model
        self.network = torch.hub.load('pytorch/vision:v0.6.0', 'mobilenet_v2', pretrained=True)
        # Replace last layer
        num_ftrs = self.network.classifier[1].in_features
        self.network.classifier[1] = nn.Linear(num_ftrs, 4)
    
    def forward(self, xb):
        out     = self.network(xb)
        return F.log_softmax(out, dim=1)

    
    def freeze(self):
        # To freeze the residual layers
        for param in self.network.parameters():
            param.require_grad = False
        for param in self.network.classifier[1].parameters():
            param.require_grad = True
    
    def unfreeze(self):
        # Unfreeze all layers
        for param in self.network.parameters():
            param.require_grad = True