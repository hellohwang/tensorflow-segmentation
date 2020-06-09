import os
import sys
import numpy as np
import random
import math
from PIL import Image, ImageOps, ImageFilter

import re
from tqdm import tqdm
from ipdb import set_trace as st
from .base import BaseDataset

CLOTHES_DATASET_PATH = '/data3/hwang/proj/PyTorch-Encoding/datasets/'


class ClothesSegmentation(BaseDataset):
    BASE_DIR = 'clothes_segmentation'
    NUM_CLASS = 12

    def __init__(self, root=os.path.expanduser(CLOTHES_DATASET_PATH), split='train',
                 mode=None, transform=None, target_transform=None, **kwargs):
        super(ClothesSegmentation, self).__init__(
            root, split, mode, transform, target_transform, **kwargs)
        # assert exists and prepare dataset automatically
        root = os.path.join(root, self.BASE_DIR)
        # assert os.path.exists(root), "Please setup the dataset using" + \
        #   "encoding/scripts/clothes.py"

        self.images, self.masks = _get_clothes_pairs(root, split)
        if split != 'test':
            assert (len(self.images) == len(self.masks))
        if len(self.images) == 0:
            raise (RuntimeError("Found 0 images in subfolders of: \
                    " + root + "\n"))

    def __getitem__(self, index):
        x_batch = np.zeros((self.batch_size, self.crop_size, self.crop_size, 3))
        y_batch = np.zeros((self.batch_size, self.crop_size, self.crop_size, 1))  # desired network output

        current_batch = self.dataset[index * self.batch_size:(index + 1) * self.batch_size]
        instance_num = 0
        for b_index in range(index * self.batch_size, (index + 1) * self.batch_size):
            img = Image.open(self.images[b_index]).convert('RGB')
            if self.mode == 'test':
                if self.transform:
                    rgb_mean = np.array([0.485, 0.456, 0.406])
                    rgb_std = np.array([0.229, 0.224, 0.225])
                    img = np.array(img).astype('int32')
                    img = (img / 255. - rgb_mean) / rgb_std
                return img, os.path.basename(image_path)

            mask = Image.open(mask_path)

            # synchrosized transform
            if self.mode == 'train':
                img, mask = self._sync_transform(img, mask)
            elif self.mode == 'val':
                img, mask = self._val_sync_transform(img, mask)
            else:
                assert self.mode == 'testval'
                mask = self._mask_transform(mask)

            # general resize, normalize
            if self.transform:
                rgb_mean = np.array([0.485, 0.456, 0.406])
                rgb_std = np.array([0.229, 0.224, 0.225])
                img = np.array(img).astype('int32')
                mask = np.array(mask).astype('int32')
                img = (img / 255. - rgb_mean) / rgb_std

        return img, mask

    def __len__(self):
        return int(np.ceil(float(len(self.dataset)) / self.batch_size))

    def size(self):
        return len(self.images)

    def _mask_transform(self, mask):
        target = np.array(mask).astype('int32')
        target[target == 255] = -1
        return tf.convert_to_tensor(target, dtype=tf.float32)

    @property
    def pred_offset(self):
        return 0


def _get_clothes_pairs(folder, split='train'):
    def get_path_pairs(folder, split_f):
        img_paths = []
        mask_paths = []
        with open(split_f, 'r') as lines:
            for line in tqdm(lines):
                ll_str = re.split(' ', line)
                imgpath = os.path.join(folder, split, 'imgs', ll_str[0].rstrip())
                maskpath = os.path.join(folder, split, 'masks', ll_str[1].rstrip())
                if os.path.isfile(maskpath) and os.path.isfile(imgpath):
                    img_paths.append(imgpath)
                    mask_paths.append(maskpath)
                else:
                    print('cannot find the mask:', maskpath)
        return img_paths, mask_paths

    if split == 'train':
        split_f = os.path.join(folder, 'train.txt')
        img_paths, mask_paths = get_path_pairs(folder, split_f)
    elif split == 'val':
        split_f = os.path.join(folder, 'val.txt')
        img_paths, mask_paths = get_path_pairs(folder, split_f)
    elif split == 'test':
        split_f = os.path.join(folder, 'test.txt')
        img_paths, mask_paths = get_path_pairs(folder, split_f)
    else:
        split_f = os.path.join(folder, 'trainval_fine.txt')
        img_paths, mask_paths = get_path_pairs(folder, split_f)
    return img_paths, mask_paths
