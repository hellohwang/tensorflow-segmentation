from .fcn8s import *
from .depv3plus import *
from .Unet import *

__all__ = ['DeepLabV3Plus', 'get_segmentation_model']


def get_segmentation_model(name, **kwargs):
    models = {
        'fcn': get_fcn,
        'deeplab': get_deeplab,
    }
    return models[name.lower()](**kwargs)