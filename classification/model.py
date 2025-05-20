import torch
import torch.nn as nn
from torchvision.models import alexnet, AlexNet_Weights

class AlexNetClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super(AlexNetClassifier, self).__init__()
        self.model = alexnet(weights=AlexNet_Weights.DEFAULT)
        self.model.classifier[6] = nn.Linear(4096, num_classes)

    def forward(self, x):
        return self.model(x)
