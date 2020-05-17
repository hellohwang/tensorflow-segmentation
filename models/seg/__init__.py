from .fcn8s import *
from .depv3plus import *

def get_segmentation_model(name, **kwargs):
    models = {
        'fcn': get_fcn,
        'deeplab': get_deeplab,
    }
    return models[name.lower()](**kwargs)