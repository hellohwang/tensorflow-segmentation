import warnings
from .clothes import ClothesSegmentation

__all__ = ['get_dataset']

datasets = {
    # 'coco': COCOSegmentation,
    # 'ade20k': ADE20KSegmentation,
    # 'pascal_voc': VOCSegmentation,
    # 'pascal_aug': VOCAugSegmentation,
    # 'pcontext': ContextSegmentation,
    # 'citys': CitySegmentation,
    # 'imagenet': ImageNetDataset,
    # 'minc': MINCDataset,
    # 'cifar10': CIFAR10,
    'clothes': ClothesSegmentation,
}

acronyms = {
    'coco': 'coco',
    'pascal_voc': 'voc',
    'pascal_aug': 'voc',
    'pcontext': 'pcontext',
    'ade20k': 'ade',
    'citys': 'citys',
    'minc': 'minc',
    'cifar10': 'cifar10',
}

def get_dataset(name, **kwargs):
    return datasets[name.lower()](**kwargs)