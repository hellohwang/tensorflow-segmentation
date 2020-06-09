import os
import sys
import random
import shutil
import cv2
import time
import math
import pprint
import numpy as np
import pandas as pd
from tensorboardX import SummaryWriter
from dataset import Dataset, BatchGenerator
import tensorflow as tf

from utils.experiments import LabelSmoother
from utils.tools import AverageMeter, Logger
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, Callback


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model=256, warmup_steps=4000):
        super(CustomSchedule, self).__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps
        self.current_lr = 0.0

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)
        result = tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)
        self.current_lr = result.numpy()
        return result


def evaluate_single_epoch(config, model, dataloader, criterion, log_val, epoch, writer, dataset_size):
    batch_time = AverageMeter()
    losses = AverageMeter()
    scores = AverageMeter()
    eval_loss = tf.keras.metrics.Mean(name='eval_loss')
    eval_accuracy = tf.keras.metrics.BinaryAccuracy(name='eval_accuracy')
    end = time.time()
    for i, (images, labels) in enumerate(dataloader):

        preds = model(images)

        loss = criterion(labels, preds)
        eval_loss(loss)
        loss_mean = eval_loss.result().numpy()

        losses.update(loss_mean, 1)
        eval_accuracy(labels, preds)
        score = eval_accuracy.result().numpy()
        scores.update(score, 1)

        batch_time.update(time.time() - end)
        end = time.time()

        if i % config.PRINT_EVERY == 0:
            print('[%2d/%2d] time: %.2f, loss: %.6f, score: %.4f'
                  % (i, dataset_size, batch_time.sum, loss_mean, score))

        del images, labels, preds
        ## end of epoch. break..
        if i > dataset_size / config.EVAL.BATCH_SIZE: break
    writer.add_scalar('val/loss', losses.avg, epoch)
    writer.add_scalar('val/score', scores.avg, epoch)
    log_val.write('[%d/%d] loss: %.6f, score: %.4f\n'
                  % (epoch, config.TRAIN.NUM_EPOCHS, losses.avg, scores.avg))
    print('average loss over VAL epoch: %f' % losses.avg)

    return scores.avg, losses.avg


def train_single_epoch(config, model, dataloader, criterion, optimizer, log_train, epoch, writer, dataset_size):
    batch_time = AverageMeter()
    losses = AverageMeter()
    scores = AverageMeter()

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.BinaryAccuracy(name='train_accuracy')
    end = time.time()
    for i, (images, labels) in enumerate(dataloader):
        with tf.GradientTape() as grad_tape:
            preds = model(images)
            loss = criterion(labels, preds)

        gradients = grad_tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        # preds = tf.sigmoid(logits)
        train_loss(loss)
        train_accuracy(labels, preds)
        losses.update(train_loss.result().numpy(), 1)
        scores.update(train_accuracy.result().numpy(), 1)

        batch_time.update(time.time() - end)
        end = time.time()
        dataloader_len = dataset_size / config.TRAIN.BATCH_SIZE
        if i % 100 == 0:
            print("[%d/%d][%d/%d] time: %.2f, loss: %.6f, score: %.4f, lr: %f"
                  % (epoch, config.TRAIN.NUM_EPOCHS, i, dataloader_len, batch_time.sum, train_loss.result().numpy(),
                     train_accuracy.result().numpy(),
                     optimizer.learning_rate.numpy()))

        if i == 0:
            iteration = dataloader_len * epoch + i
            annotated_images = utils.tools.annotate_to_images(images, labels, preds.numpy())
            for idx, annotated_image in enumerate(annotated_images):
                writer.add_image('train/image_{}_class_{}'.format(int(idx / 8), idx % 8), annotated_image, iteration)

        del images, labels, preds
        ## end of epoch. break..
        if i > dataset_size / config.TRAIN.BATCH_SIZE: break
    writer.add_scalar('train/score', scores.avg, epoch)
    writer.add_scalar('train/loss', losses.avg, epoch)
    writer.add_scalar('train/lr', optimizer.learning_rate.numpy(), epoch)
    log_train.write('[%d/%d] loss: %.6f, score: %.4f, lr: %f\n'
                    % (epoch, config.TRAIN.NUM_EPOCHS, losses.avg, scores.avg, optimizer.learning_rate.numpy()))
    print('average loss over TRAIN epoch: %f' % losses.avg)


def train(config, model, train_loader, test_loader, optimizer, log_train, log_val, start_epoch, best_score, best_loss,
          writer, dataset_size, criterion):
    if 1:  # keras mode
        ## train phase..
        metric = tf.keras.metrics.BinaryAccuracy()

        checkpoint_all = ModelCheckpoint(
            'checkpoints\\all_models.{epoch:02d}-{loss:.2f}.h5',
            monitor='loss',
            verbose=1,
            save_best_only=False,
            mode='auto',
            period=1
        )

        model.compile(optimizer=optimizer, loss=dice_loss, metrics=['accuracy'])
        model.fit_generator(generator=train_loader, steps_per_epoch=len(train_loader), epochs=config.TRAIN.NUM_EPOCHS,
                            verbose=1,
                            validation_data=test_loader,
                            max_queue_size=10,
                            workers=config.TRAIN.NUM_WORKERS,
                            use_multiprocessing=False,
                            callbacks=[checkpoint_all])
        ## use_multiprocessing=True.. get erorr i don't know..

    else:  # pytorch style mode
        for epoch in range(start_epoch, config.TRAIN.NUM_EPOCHS):
            # ## TODO set a loss function..
            train_single_epoch(config, model, train_loader, criterion, optimizer, log_train, epoch, writer,
                               dataset_size[0])

            test_score, test_loss = evaluate_single_epoch(config, model, test_loader, criterion, log_val, epoch, writer,
                                                          dataset_size[1])
            print('Total Test Score: %.4f, Test Loss: %.4f' % (test_score, test_loss))
            #
            if test_score > best_score:
                best_score = test_score
                print('Test score Improved! Save checkpoint')

                model.save_weights(str(epoch) + "_" + str(best_score) + "_model.h5")
                # utils.checkpoint.save_checkpoint(config, model, epoch, test_score, test_loss)


def run(config):
    ## TODO change to get model

    sm.set_framework('tf.keras')  ## segmentation_model 2.0 support feature..
    backbone = 'mobilenetv2'
    model = sm.Unet(backbone, input_shape=(256, 256, 3), encoder_weights=None,
                    activation='sigmoid')  # activation='identity')#, decoder_attention_type='scse')  # 'imagenet')
    model.summary()

    ## TODO optimizer change
    # optimizer = tf.keras.optimizers.Adam(learning_rate_schedule)#learning_rate=config.OPTIMIZER.LR) #get_optimizer(config, model.parameters())
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=config.OPTIMIZER.LR)  # config.OPTIMIZER.LR) #get_optimizer(config, model.parameters())
    ##loss ##
    criterion = FocalLoss()  # DiceLoss()#tf.keras.losses.BinaryCrossentropy()

    checkpoint = None
    # checkpoint = utils.checkpoint.get_initial_checkpoint(config)
    if checkpoint is not None:
        last_epoch, score, loss = utils.checkpoint.load_checkpoint(config, model, checkpoint)
        # utils.checkpoint.load_checkpoint_legacy(config, model, checkpoint)
    else:
        print('[*] no checkpoint found')
        last_epoch, score, loss = -1, -1, float('inf')
    print('last epoch:{} score:{:.4f} loss:{:.4f}'.format(last_epoch, score, loss))

    # optimizer.param_groups[0]['initial_lr'] = config.OPTIMIZER.LR

    writer = SummaryWriter(os.path.join(config.TRAIN_DIR + config.RECIPE, 'logs'))
    log_train = Logger()
    log_val = Logger()
    log_train.open(os.path.join(config.TRAIN_DIR + config.RECIPE, 'log_train.txt'), mode='a')
    log_val.open(os.path.join(config.TRAIN_DIR + config.RECIPE, 'log_val.txt'), mode='a')
    train_loader = BatchGenerator(config, 'train', config.TRAIN.BATCH_SIZE, None)
    # train_dataset = Dataset(config, 'train', None)
    # train_loader = train_dataset.DataGenerator(config.DATA_DIR, batch_size=config.TRAIN.BATCH_SIZE, shuffle = True)
    train_datasize = len(train_loader)  # train_dataset.get_length()

    # val_dataset = Dataset(config, 'val', None)
    # val_loader = val_dataset.DataGenerator(config.DATA_DIR, batch_size=config.TRAIN.BATCH_SIZE, shuffle=False)

    val_loader = BatchGenerator(config, 'val', config.EVAL.BATCH_SIZE, None)
    val_datasize = len(val_loader)  # val_dataset.get_length()

    ### TODO: add transform

    train(config, model, train_loader, val_loader, optimizer, log_train, log_val, last_epoch + 1, score, loss, writer,
          (train_datasize, val_datasize), criterion)

    model.save_weights("model.h5")


def seed_everything():
    seed = 2019
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def main():
    import warnings
    warnings.filterwarnings("ignore")

    print('start training.')
    seed_everything()

    ymls = ['configs/fastscnn_mv3_sj_add_data_1024.yml']
    for yml in ymls:
        config = utils.config.load(yml)

        os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'  # config.GPU
        prepare_train_directories(config)
        pprint.pprint(config, indent=2)
        utils.config.save_config(yml, config.TRAIN_DIR + config.RECIPE)
        run(config)

    print('success!')


if __name__ == '__main__':
    main()
