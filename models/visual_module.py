# models/visual_module.py
import torch
import torch.nn as nn
import torchvision.models as models

class VisualModule(nn.Module):
    def __init__(self, backbone="resnet50", pretrained=True, out_features=512):
        super(VisualModule, self).__init__()
        if backbone == "resnet50":
            self.model = models.resnet50(pretrained=pretrained)
            in_features = self.model.fc.in_features
            self.model.fc = nn.Linear(in_features, out_features)
        else:
            raise NotImplementedError(f"Backbone {backbone} not implemented")

    def forward(self, x):
        return self.model(x)  # [batch, out_features]
