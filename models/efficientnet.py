import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet


class EfficientNetCoral(nn.Module):
    def __init__(self, model_name, num_classes):
        super(EfficientNetCoral, self).__init__()
        self.num_classes = num_classes
        self.effnet = EfficientNet.from_pretrained(model_name, num_classes=1)
        self.linear_1_bias = nn.Parameter(torch.zeros(self.num_classes-1).float())

    def forward(self, x):
        logits = self.effnet(x)
        logits = logits + self.linear_1_bias
        return logits
