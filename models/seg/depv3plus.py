

class DepV3Plus():
    def __init__(self):
        pass

    
def get_deeplab(dataset='pascal_voc', backbone='resnet50s', root='~/.encoding/models', **kwargs):
    from ...datasets import datasets
    # model = DepV3Plus(datasets[dataset.lower()].NUM_CLASS, backbone=backbone, root=root, **kwargs)
    # return model