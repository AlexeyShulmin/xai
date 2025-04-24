import torch.nn as nn
import torchvision.models as models

def get_model(num_classes: int = 2, pretrained: bool = True):
    """Return a ResNet‑50 with custom classification head."""
    weights = (models.ResNet50_Weights.IMAGENET1K_V2
               if pretrained else None)
    model = models.resnet50(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
