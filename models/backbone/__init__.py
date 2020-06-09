from .mobilenet_v2 import *
from .resnet50 import *

__all__ = ['mobilenet_v2', 'resnet50', 'get_backbone']

backbones = {
    'mobilenet_v2': MobileNetV2,
    'resnet50': ResNet50,
}


def get_backbone(name, *args, **kwargs):
    return backbones[name.lower()](*args, **kwargs)
