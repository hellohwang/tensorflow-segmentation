import sys
import os

this_dir = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(this_dir, '..'))

from datasets import *
from models.seg import *
import argparse
import tensorflow as tf
from tensorflow.keras import applications
from option import Options
from tensorflow.keras import models, layers

# os.environ["TF_CPP_MIN_LOG_LEVEL"] = '1'  # show all information
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'  # show warnings and errors
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'  # only show errors

args = Options().parse()

# dataset generator
data_kwargs = {'base_size': args.base_size, 'crop_size': args.crop_size}
trainset = get_dataset(args.dataset, split=args.train_split, mode='train', **data_kwargs)
trainset_datagen = trainset._DataGenerator()
valset = get_dataset(args.dataset, split=args.train_split, mode='val', **data_kwargs)
valset_datagen = valset._DataGenerator()

# model definition
strategy = tf.distribute.MirroredStrategy()

with strategy.scope():
    model = get_segmentation_model('deeplab', dataset='clothes', backbone='mobilenet_v2', input_shape=args.crop_size)
    model.build(input_shape=[args.batch_size, 3, args.crop_size, args.crop_size])
    model.summary()
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr)
    model.compile(optimizer=optimizer, loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
    callback = tf.keras.callbacks.ModelCheckpoint(
        "FCN8s.h5", verbose=1, save_weights_only=True)
    # model.fit_generator(trainset_datagen, steps_per_epoch=6000,
    #                     epochs=30, callbacks=[callback])
    model.fit(trainset_datagen, steps_per_epoch=6000, epochs=30, callbacks=[callback])
    model.save_weights("model.h5")
