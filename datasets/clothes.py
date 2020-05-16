import os
import sys
import numpy as np
import random
import math
from PIL import Image

import re
import tensorflow as tf
from tqdm import tqdm
from .base import BaseDataset

__all__ = ['ClothesSegmentation']

CLOTHES_DATASET_PATH = '/data3/hwang/proj/tensorflow-segmentation/datasets/data'


class ClothesSegmentation(BaseDataset):
    BASE_DIR = 'clothes_segmentation'
    NUM_CLASS = 12
    LEN_DATASET = 0

    def __init__(self, root=os.path.expanduser(CLOTHES_DATASET_PATH), split='train',
                 mode=None, transform=True, **kwargs):
        super(ClothesSegmentation, self).__init__(
            root, split, mode, transform, **kwargs)
        # assert exists and prepare dataset automatically
        root = os.path.join(root, self.BASE_DIR)
        # assert os.path.exists(root), "Please setup the dataset using" + \
        #   "encoding/scripts/clothes.py"
        self.len_dataset = 0
        self.image_label_path_generator = self._get_clothes_pairs(root, split)
        # if split != 'test':
        #    assert (len(self.image_path_generator) == len(self.masks))
        # if len(self.image_label_path_generator) == 0:
        #     raise (RuntimeError("Found 0 images in subfolders of: \
        #             " + root + "\n"))

    def _mask_transform(self, mask):
        target = np.array(mask).astype('int32')
        target[target == 255] = -1
        return tf.convert_to_tensor(target, dtype=tf.float32)

    def __len__(self):
        return len(self.images)

    @property
    def pred_offset(self):
        return 0

    def process_image_label(self, image_path, mask_path):
        img = Image.open(image_path).convert('RGB')
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

    @classmethod
    def _get_clothes_pairs(cls, folder, split='train'):
        def get_path_pairs(folder, split_f):
            image_label_paths = []
            with open(split_f, 'r') as lines:
                for line in tqdm(lines):
                    ll_str = re.split(' ', line)
                    imgpath = os.path.join(
                        folder, split, 'imgs', ll_str[0].rstrip())
                    maskpath = os.path.join(
                        folder, split, 'masks', ll_str[1].rstrip())
                    if os.path.isfile(maskpath) and os.path.isfile(imgpath):
                        image_label_paths.append((imgpath, maskpath))
                    else:
                        print('cannot find the mask:', maskpath)
            cls.LEN_DATASET = len(image_label_paths)
            while True:
                random.shuffle(image_label_paths)
                for i in range(len(image_label_paths)):
                    yield image_label_paths[i]

        if split == 'train':
            split_f = os.path.join(folder, 'train.txt')
            image_label_paths = get_path_pairs(folder, split_f)
        elif split == 'val':
            split_f = os.path.join(folder, 'val.txt')
            image_label_paths = get_path_pairs(folder, split_f)
        elif split == 'test':
            split_f = os.path.join(folder, 'test.txt')
            image_label_paths = get_path_pairs(folder, split_f)
        else:
            split_f = os.path.join(folder, 'trainval_fine.txt')
            image_label_paths = get_path_pairs(folder, split_f)
        return image_label_paths

    def _DataGenerator(self):
        """
        generate image and mask at the same time
        """
        while True:
            images = np.zeros(
                shape=[self.batch_size, self.crop_size, self.crop_size, 3])
            labels = np.zeros(
                shape=[self.batch_size, self.crop_size, self.crop_size], dtype=np.float)
            for i in range(self.batch_size):
                image_path, label_path = next(self.image_label_path_generator)
                image, label = self.process_image_label(image_path, label_path)
                images[i], labels[i] = image, label
            yield images, labels


# def _get_clothes_pairs(folder, split='train'):
#     def get_path_pairs(folder, split_f):
#         image_label_paths = []
#         with open(split_f, 'r') as lines:
#             for line in tqdm(lines):
#                 ll_str = re.split(' ', line)
#                 imgpath = os.path.join(
#                     folder, split, 'imgs', ll_str[0].rstrip())
#                 maskpath = os.path.join(
#                     folder, split, 'masks', ll_str[1].rstrip())
#                 if os.path.isfile(maskpath) and os.path.isfile(imgpath):
#                     image_label_paths.append((imgpath, maskpath))
#                 else:
#                     print('cannot find the mask:', maskpath)
#         while True:
#             random.shuffle(image_label_paths)
#             for i in range(len(image_label_paths)):
#                 yield image_label_paths[i]
#
#     if split == 'train':
#         split_f = os.path.join(folder, 'train.txt')
#         image_label_paths = get_path_pairs(folder, split_f)
#     elif split == 'val':
#         split_f = os.path.join(folder, 'val.txt')
#         image_label_paths = get_path_pairs(folder, split_f)
#     elif split == 'test':
#         split_f = os.path.join(folder, 'test.txt')
#         image_label_paths = get_path_pairs(folder, split_f)
#     else:
#         split_f = os.path.join(folder, 'trainval_fine.txt')
#         image_label_paths = get_path_pairs(folder, split_f)
#     return image_label_paths