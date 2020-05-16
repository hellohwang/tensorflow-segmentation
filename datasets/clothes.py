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

CLOTHES_DATASET_PATH = '/data3/hwang/proj/PyTorch-Encoding/datasets/'


class ClothesSegmentation(BaseDataset):
    BASE_DIR = 'clothes_segmentation'
    NUM_CLASS = 12

    def __init__(self, root=os.path.expanduser(CLOTHES_DATASET_PATH), split='train',
                 mode=None, transform=True, **kwargs):
        super(ClothesSegmentation, self).__init__(root, split, mode, transform, **kwargs)
        # assert exists and prepare dataset automatically
        root = os.path.join(root, self.BASE_DIR)
        # assert os.path.exists(root), "Please setup the dataset using" + \
        #   "encoding/scripts/clothes.py"

        self.image_label_path_generator = _get_clothes_pairs(root, split)
        if split != 'test':
            assert (len(self.images) == len(self.masks))
        if len(self.images) == 0:
            raise (RuntimeError("Found 0 images in subfolders of: \
                    " + root + "\n"))

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
            img = (img / 255. - rgb_mean) / rgb_std

        return img, mask

    def _DataGenerator(self):
        """
        generate image and mask at the same time
        """
        while True:
            images = np.zeros(shape=[self.batch_size, self.crop_size, self.crop_size, 3])
            labels = np.zeros(shape=[self.batch_size, self.crop_size, self.crop_size], dtype=np.float)
            for i in range(self.batch_size):
                image_path, label_path = next(self.image_label_path_generator)
                image, label = self.process_image_label(image_path, label_path)
                images[i], labels[i] = image, label
            yield images, labels


def _get_clothes_pairs(folder, split='train'):
    def get_path_pairs(folder, split_f):
        image_label_paths = []
        with open(split_f, 'r') as lines:
            for line in tqdm(lines):
                ll_str = re.split(' ', line)
                imgpath = os.path.join(folder, split, 'imgs', ll_str[0].rstrip())
                maskpath = os.path.join(folder, split, 'masks', ll_str[1].rstrip())
                if os.path.isfile(maskpath) and os.path.isfile(imgpath):
                    image_label_paths.append((imgpath, maskpath))
                else:
                    print('cannot find the mask:', maskpath)
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

#
# def create_image_label_path_generator(images_filepath, labels_filepath):
#   image_paths = open(images_filepath).readlines()
#   all_label_txts = os.listdir(labels_filepath)
#   image_label_paths = []
#   for label_txt in all_label_txts:
#     label_name = label_txt[:-4]
#     label_path = labels_filepath + "/" + label_txt
#     for image_path in image_paths:
#       image_path = image_path.rstrip()
#       image_name = image_path.split("/")[-1][:-4]
#       if label_name == image_name:
#         image_label_paths.append((image_path, label_path))
#   while True:
#     random.shuffle(image_label_paths)
#     for i in range(len(image_label_paths)):
#       yield image_label_paths[i]


# def process_image_label(image_path, label_path):
#     # image = misc.imread(image_path)
#     image = cv2.imread(image_path)
#     image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_NEAREST)
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     # data augmentation here
#     # randomly shift gamma
#     gamma = random.uniform(0.8, 1.2)
#     image = image.copy() ** gamma
#     image = np.clip(image, 0, 255)
#     # randomly shift brightness
#     brightness = random.uniform(0.5, 2.0)
#     image = image.copy() * brightness
#     image = np.clip(image, 0, 255)
#     # image transformation here
#     image = (image / 255. - rgb_mean) / rgb_std
#
#     label = open(label_path).readlines()
#     label = [np.array(line.rstrip().split(" ")) for line in label]
#     label = np.array(label, dtype=np.int)
#     label = cv2.resize(label, (224, 224), interpolation=cv2.INTER_NEAREST)
#     label = label.astype(np.int)
#
#     return image, label
#
#
# def DataGenerator(train_image_txt, train_labels_dir, batch_size):
#   """
#   generate image and mask at the same time
#   """
#   image_label_path_generator = create_image_label_path_generator(
#     train_image_txt, train_labels_dir
#   )
#   while True:
#     images = np.zeros(shape=[batch_size, 224, 224, 3])
#     labels = np.zeros(shape=[batch_size, 224, 224], dtype=np.float)
#     for i in range(batch_size):
#       image_path, label_path = next(image_label_path_generator)
#       image, label = process_image_label(image_path, label_path)
#       images[i], labels[i] = image, label
#     yield images, labels
