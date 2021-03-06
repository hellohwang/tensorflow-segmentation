#! /usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2020 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : multi_gpu_train.py
#   Author      : YunYang1994
#   Created date: 2020-02-02 22:14:30
#   Description :
#
#================================================================

import os
import shutil
import tensorflow as tf
from tqdm import tqdm
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import applications

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

import sys
import os

this_dir = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(this_dir, '..'))

from datasets import *

from models.seg import *
import argparse
import tensorflow as tf

# os.environ["TF_CPP_MIN_LOG_LEVEL"] = '1'  # show all information
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'  # show warnings and errors
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'  # only show errors


class Options():
    def __init__(self):
        parser = argparse.ArgumentParser(description='PyTorch \
            Segmentation')
        # model and dataset
        parser.add_argument('--model', type=str, default='encnet',
                            help='model name (default: encnet)')
        parser.add_argument('--backbone', type=str, default='resnet50',
                            help='backbone name (default: resnet50)')
        parser.add_argument('--dataset', type=str, default='ade20k',
                            help='dataset name (default: pascal12)')
        parser.add_argument('--base-size', type=int, default=520,
                            help='base image size')
        parser.add_argument('--crop-size', type=int, default=480,
                            help='crop image size')
        parser.add_argument('--train-split', type=str, default='train',
                            help='dataset train split (default: train)')
        # training hyper params
        parser.add_argument('--aux', action='store_true', default=False,
                            help='Auxilary Loss')
        parser.add_argument('--aux-weight', type=float, default=0.2,
                            help='Auxilary loss weight (default: 0.2)')
        parser.add_argument('--epochs', type=int, default=None, metavar='N',
                            help='number of epochs to train (default: auto)')
        parser.add_argument('--start_epoch', type=int, default=0,
                            metavar='N', help='start epochs (default:0)')
        parser.add_argument('--batch-size', type=int, default=16,
                            metavar='N', help='input batch size for \
                            training (default: auto)')
        parser.add_argument('--test-batch-size', type=int, default=16,
                            metavar='N', help='input batch size for \
                            testing (default: same as batch size)')
        # optimizer params
        parser.add_argument('--lr', type=float, default=None, metavar='LR',
                            help='learning rate (default: auto)')
        parser.add_argument('--lr-scheduler', type=str, default='poly',
                            help='learning rate scheduler (default: poly)')
        parser.add_argument('--momentum', type=float, default=0.9,
                            metavar='M', help='momentum (default: 0.9)')
        parser.add_argument('--weight-decay', type=float, default=1e-4,
                            metavar='M', help='w-decay (default: 1e-4)')
        # cuda, seed and logging
        parser.add_argument('--no-cuda', action='store_true', default=False,
                            help='disables CUDA training')
        parser.add_argument('--seed', type=int, default=1, metavar='S',
                            help='random seed (default: 1)')
        # checking point
        parser.add_argument('--resume', type=str, default=None,
                            help='put the path to resuming file if needed')
        parser.add_argument('--checkname', type=str, default='default',
                            help='set the checkpoint name')
        parser.add_argument('--model-zoo', type=str, default=None,
                            help='evaluating on model zoo model')
        # finetuning pre-trained models
        parser.add_argument('--ft', action='store_true', default=False,
                            help='finetuning on a different dataset')
        # evaluation option
        parser.add_argument('--eval', action='store_true', default=False,
                            help='evaluating mIoU')
        parser.add_argument('--test-val', action='store_true', default=False,
                            help='generate masks on val set')
        parser.add_argument('--no-val', action='store_true', default=False,
                            help='skip validation during training')
        # test option
        parser.add_argument('--test-folder', type=str, default=None,
                            help='path to test image folder')
        # gpu memory limit
        parser.add_argument('--gpu-size', type=int, default=30,
                            metavar='N', help='tensorflow gpu memory limit')

        # the parser
        self.parser = parser

    def parse(self):
        args = self.parser.parse_args()
        args.cuda = not args.no_cuda and tf.cuda.is_available()
        # default settings for epochs, batch_size and lr
        if args.epochs is None:
            epoches = {
                'coco': 30,
                'pascal_aug': 80,
                'pascal_voc': 50,
                'pcontext': 80,
                'ade20k': 180,
                'citys': 240,
            }
            args.epochs = epoches[args.dataset.lower()]
        if args.lr is None:
            lrs = {
                'coco': 0.004,
                'pascal_aug': 0.001,
                'pascal_voc': 0.0001,
                'pcontext': 0.001,
                'ade20k': 0.004,
                'citys': 0.004,
            }
            args.lr = lrs[args.dataset.lower()] / 16 * args.batch_size
        print(args)
        return args

EPOCHS       = 40
SCORE_THRESH = 0.8
NUM_CLASS    = 10
EMB_SIZE     = 2     # Embedding Size
IMG_SIZE     = 112   # Input Image Size
BATCH_SIZE   = 512   # Total 4 GPU, 128 batch per GPU
GPU_SIZE     = 30    # (G)  MemorySIZE per GPU

#------------------------------------ Prepare Dataset ------------------------------------#

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=False)

train_generator = train_datagen.flow_from_directory(
        './mnist/train',
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical')

test_datagen = ImageDataGenerator(
        rescale=1./255,
        horizontal_flip=False)

test_generator = test_datagen.flow_from_directory(
        './mnist/test',
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=2,
        class_mode='categorical')

#------------------------------------ Build Mode -----------------------------------#

tf.debugging.set_log_device_placement(True)
gpus = tf.config.experimental.list_physical_devices('GPU')

for gpu in gpus:
    tf.config.experimental.set_virtual_device_configuration(
        gpu, [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=GPU_SIZE*1024)]
    )
logical_gpus = tf.config.experimental.list_logical_devices('GPU')
print(len(gpus), "Physical GPU,", len(logical_gpus), "Logical GPUs")

tf.debugging.set_log_device_placement(True)
strategy = tf.distribute.MirroredStrategy()

# Defining Model
with strategy.scope():
    backbone = applications.mobilenet_v2.MobileNetV2(include_top=False,
                                                    weights='imagenet', input_shape=(IMG_SIZE,IMG_SIZE,3))
    x = tf.keras.layers.Input(shape=(IMG_SIZE,IMG_SIZE,3))
    y = backbone(x)
    y = tf.keras.layers.AveragePooling2D()(y)
    y = tf.keras.layers.Flatten()(y)
    y = tf.keras.layers.Dense(EMB_SIZE,  activation=None)(y)
    featureExtractor = tf.keras.models.Model(inputs=x, outputs=y)
    model = tf.keras.Sequential([
        featureExtractor,
        tf.keras.layers.Dense(NUM_CLASS, activation='softmax')
    ])

    model.build(input_shape=[1, IMG_SIZE, IMG_SIZE, 3])
    optimizer = tf.keras.optimizers.Adam(0.001)

# Defining Loss and Metrics
with strategy.scope():
    loss_object = tf.keras.losses.CategoricalCrossentropy(
        reduction=tf.keras.losses.Reduction.NONE
    )
    def compute_loss(labels, predictions):
        per_example_loss = loss_object(labels, predictions)
        return tf.nn.compute_average_loss(per_example_loss, global_batch_size=BATCH_SIZE)

    train_accuracy = tf.keras.metrics.CategoricalAccuracy(
        name='train_accuracy'
    )

# Defining Training Step
with strategy.scope():
    def train_step(inputs):
        images, labels = inputs

        with tf.GradientTape() as tape:
            predictions = model(images, training=True)
            loss = compute_loss(labels, predictions)

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        train_accuracy.update_state(labels, predictions)
        return loss

#------------------------------------ Training Loop -----------------------------------#

# Defining Training Loops
with strategy.scope():
    @tf.function
    def distributed_train_step(dataset_inputs):
        per_replica_losses = strategy.experimental_run_v2(train_step,
                                                          args=(dataset_inputs,))
        return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses,
                               axis=None)
    for epoch in range(1, EPOCHS+1):
        if epoch == 30: optimizer.lr.assign(0.0001)

        batchs_per_epoch = len(train_generator)
        train_dataset    = iter(train_generator)
        test_dataset     = iter(test_generator)

        with tqdm(total=batchs_per_epoch,
                  desc="Epoch %2d/%2d" %(epoch, EPOCHS)) as pbar:
            loss_value = 0.
            acc_value  = 0.
            num_batch  = 0

            for _ in range(batchs_per_epoch):

                num_batch  += 1
                batch_loss = distributed_train_step(next(train_dataset))
                batch_acc  = train_accuracy.result()

                loss_value += batch_loss
                acc_value  += batch_acc

                pbar.set_postfix({'loss' : '%.4f'     %(loss_value / num_batch),
                                  'accuracy' : '%.6f' %(acc_value  / num_batch)})
                train_accuracy.reset_states()
                pbar.update(1)

        model_path = "./models/weights_%02d" %epoch
        if not os.path.exists(model_path):
            os.makedirs(model_path)

        model.save(os.path.join(model_path, "model.h5"))
        featureExtractor.save(os.path.join(model_path, "featureExtractor.h5"))


