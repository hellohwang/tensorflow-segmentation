import sys
import os

this_dir = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(this_dir, '..'))

from datasets import *
from models.seg import *
import argparse
import math
from tqdm import tqdm
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
    # model definition
    backbone = applications.mobilenet_v2.MobileNetV2(include_top=False,
                                                     weights='imagenet', input_shape=(args.crop_size, args.crop_size, 3))
    x = tf.keras.layers.Input(shape=(args.crop_size, args.crop_size, 3))
    y = backbone(x)
    y = tf.keras.layers.AveragePooling2D()(y)
    y = tf.keras.layers.Flatten()(y)
    y = tf.keras.layers.Dense(2, activation=None)(y)
    featureExtractor = tf.keras.models.Model(inputs=x, outputs=y)
    model = tf.keras.Sequential([
        featureExtractor,
        tf.keras.layers.Dense(12, activation='softmax')
    ])

    model.build(input_shape=[args.crop_size, args.crop_size, 3])
    optimizer = tf.keras.optimizers.Adam(0.001)

    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        reduction=tf.keras.losses.Reduction.NONE
    )


    def compute_loss(labels, predictions):
        per_example_loss = loss_object(labels, predictions)
        return tf.nn.compute_average_loss(per_example_loss, global_batch_size=args.batch_size)


    train_accuracy = tf.keras.metrics.CategoricalAccuracy(
        name='train_accuracy'
    )


    def train_step(inputs):
        images, labels = inputs

        with tf.GradientTape() as tape:
            predictions = model(images, training=True)
            loss = compute_loss(labels, predictions)

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        train_accuracy.update_state(labels, predictions)
        return loss


    @tf.function
    def distributed_train_step(dataset_inputs):
        per_replica_losses = strategy.experimental_run_v2(train_step,
                                                          args=(dataset_inputs,))
        return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses,
                               axis=None)


    for epoch in range(1, args.epochs + 1):
        if epoch == 30: optimizer.lr.assign(args.lr)

        batchs_per_epoch = math.ceil(trainset.LEN_TRAINSET / args.batch_size)
        print("batchs_per_epoch:", batchs_per_epoch)
        train_dataset = trainset_datagen
        test_dataset = valset_datagen

        with tqdm(total=batchs_per_epoch,
                  desc="Epoch %2d/%2d" % (epoch, args.epochs)) as pbar:
            loss_value = 0.
            acc_value = 0.
            num_batch = 0

            for _ in range(batchs_per_epoch):
                num_batch += 1
                batch_loss = distributed_train_step(next(train_dataset))
                batch_acc = train_accuracy.result()

                loss_value += batch_loss
                acc_value += batch_acc

                pbar.set_postfix({'loss': '%.4f' % (loss_value / num_batch),
                                  'accuracy': '%.6f' % (acc_value / num_batch)})
                train_accuracy.reset_states()
                pbar.update(1)

        model_path = "./models/weights_%02d" % epoch
        if not os.path.exists(model_path):
            os.makedirs(model_path)

        model.save(os.path.join(model_path, "model.h5"))

# def create_model():
#     model = models.Sequential()
#
#     model.add(layers.Embedding(384, 7, input_length=384))
#     model.add(layers.Conv1D(filters=64, kernel_size=5, activation="relu"))
#     model.add(layers.MaxPool1D(2))
#     model.add(layers.Conv1D(filters=32, kernel_size=3, activation="relu"))
#     model.add(layers.MaxPool1D(2))
#     model.add(layers.Flatten())
#     model.add(layers.Dense(12, activation="softmax"))
#     return (model)
#
#
with strategy.scope():
    model = Unet(12, args.crop_size)
    # model = create_model()
    # model.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-4),
    #               loss='sparse_categorical_crossentropy',
    #               metrics=['accuracy'])
    # model.compile(optimizer=tf.keras.optimizers.Nadam(),
    #               loss='sparse_categorical_crossentropy',
    #               metrics=['accuracy'])
    # train your FCN8s model
    callback = tf.keras.callbacks.ModelCheckpoint(
        "FCN8s.h5", verbose=1, save_weights_only=True)
    # model.fit_generator(trainset_datagen, steps_per_epoch=6000,
    #                     epochs=30, callbacks=[callback])
    model.fit(trainset_datagen, steps_per_epoch=6000, epochs=30, callbacks=[callback])
    model.save_weights("model.h5")
