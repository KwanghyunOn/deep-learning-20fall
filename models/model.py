from .resnet import resnet34
from .efficientnet import EfficientNetCoral


def coral_model(backbone, num_classes):
    if backbone == "resnet34":
        return resnet34(num_classes)
    else:
        return EfficientNetCoral(backbone, num_classes)
