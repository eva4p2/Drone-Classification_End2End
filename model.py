import torch
import torch.nn.functional as F
import torch.nn as nn

class DroneMobilenetV2(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Use ROIpoolng to resize image to 224*224
        self.ROIpool = nn.AdaptiveMaxPool2d((224,224))
        # Use a pretrained MobilenetV2 model
        self.network = torch.hub.load('pytorch/vision:v0.6.0', 'mobilenet_v2', pretrained=True)
        # Replace last layer
        num_ftrs = self.network.classifier[1].in_features
        self.network.classifier[1] = nn.Linear(num_ftrs, 4)
    
    def forward(self, xb):
        resized = self.ROIpool(xb)
        out     = self.network(resized)
        return F.log_softmax(out)
#         return torch.sigmoid(self.fc(self.relu(self.network(xb))))

    
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