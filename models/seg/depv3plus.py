import tensorflow as tf
from tensorflow.keras import layers, models, activations

import os, sys

this_dir = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(this_dir, '..'))

from backbone import *

__all__ = ['DeepLabV3Plus', 'get_deeplab']


class Conv(tf.keras.Model):
    def __init__(self, filters, kernel_size, strides=1, padding='same', dilation_rate=1,
                 kernel_initializer='he_normal', use_bias=False):
        super(Conv, self).__init__()
        self.conv = layers.Conv2D(filters, kernel_size, strides, padding, kernel_initializer=kernel_initializer,
                                  dilation_rate=dilation_rate, use_bias=use_bias)

    def call(self, x):
        return self.conv(x)


class ConvBNReLU(tf.keras.Model):
    def __init__(self, filters, kernel_size, strides=1, padding='same', dilation_rate=1,
                 kernel_initializer='he_normal', use_bias=False):
        super(ConvBNReLU, self).__init__()
        self.conv = layers.Conv2D(filters, kernel_size, strides, padding, kernel_initializer=kernel_initializer,
                                  dilation_rate=dilation_rate, use_bias=use_bias)
        self.bn = layers.BatchNormalization()
        self.relu = layers.ReLU()

    def call(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class ASPPConv(tf.keras.Model):
    def __init__(self, filters, atrous_rate):
        super(ASPPConv, self).__init__()
        self.block = models.Sequential([
            Conv(filters, 3, dilation_rate=atrous_rate, use_bias=False),
            layers.BatchNormalization(),
            layers.ReLU(),
        ])


def call(self, x):
    return self.block(x)


class AsppPooling(tf.keras.Model):
    def __init__(self, filters):
        super(AsppPooling, self).__init__()
        self.gap = models.Sequential([
            layers.GlobalAveragePooling2D(),
            Conv(filters, 1, use_bias=False),
            layers.BatchNormalization(),
            layers.ReLU(),
        ])


def call(self, x):
    size = x.size()[2:]
    pool = self.gap(x)
    out = layers.UpSampling2D(size, interpolation='bilinear')(pool)
    return out


class ASPP(tf.keras.Model):
    def __init__(self, atrous_rates):
        print(atrous_rates)
        super(ASPP, self).__init__()
        filters = 256
        self.b0 = models.Sequential([
            Conv(filters, 1, use_bias=False),
            layers.BatchNormalization(),
            layers.ReLU(),
        ])

        rate1, rate2, rate3 = tuple(atrous_rates)
        self.b1 = ASPPConv(filters, rate1)
        self.b2 = ASPPConv(filters, rate2)
        self.b3 = ASPPConv(filters, rate3)
        self.b4 = AsppPooling(filters)

        self.project = models.Sequential([
            Conv(5 * filters, 1, use_bias=False),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Dropout(0.5),
        ])

    def call(self, x):
        feat1 = self.b0(x)
        feat2 = self.b1(x)
        feat3 = self.b2(x)
        feat4 = self.b3(x)
        feat5 = self.b4(x)
        x = layers.concatenate((feat1, feat2, feat3, feat4, feat5), axis=1)
        x = self.project(x)
        return x


class DeepLabV3PlusHead(tf.keras.Model):
    def __init__(self, nclass, atrous_rates=[6, 12, 18]):
        super(DeepLabV3PlusHead, self).__init__()
        self.aspp = ASPP(atrous_rates)
        self.decoder = Decoder(nclass)

    def call(self, x):
        c = self.aspp(x[0])
        x = self.decoder(x[1], x[2], x[3], c)
        return x


class Decoder(tf.keras.Model):
    def __init__(self, filters):
        super(Decoder, self).__init__()
        self.first_conv = ConvBNReLU(48, 3)
        self.second_conv = ConvBNReLU(48, 3)
        self.third_conv = ConvBNReLU(48, 3)
        self.gamma = tf.Variable(tf.ones(1))
        self.att = Conv(288, 1)

        self.conv = models.Sequential([
            ConvBNReLU(256, 3),
            ConvBNReLU(256, 3),
        ])

        self.last_conv = models.Sequential([
            ConvBNReLU(256, 3),
            layers.Dropout(0.5),
            ConvBNReLU(256, 3),
            layers.Dropout(0.1),
            Conv(filters, 1),
        ])

    def call(self, c1, c2, c3, x):  # x channels:256 aspp output
        fourth_feature = self.first_conv(c1)  # 1/4
        eighth_feature = self.second_conv(c2)  # 1/8
        sixteenth_feature = self.third_conv(c3)  # 1/16
        x = self.conv(layers.concatenate((x, sixteenth_feature), axis=1))
        x = layers.UpSampling2D(eighth_feature.size()[2:], interpolation='bilinear')(x)  # upsample to 1/8
        x = self.gamma * activations.sigmoid(self.att(x)) * x
        x = self.conv(layers.concatenate((x, eighth_feature), axis=1))
        x = layers.UpSampling2D(fourth_feature.size()[2:], interpolation='bilinear')(x)  # upsample to 1/4
        x = self.gamma * activations.sigmoid(self.att(x)) * x
        x = self.last_conv(layers.concatenate((x, fourth_feature), axis=1))  # default channel last, may change
        return x


class DeepLabV3Plus(tf.keras.Model):
    def __init__(self, nclass, backbone, input_shape):
        super(DeepLabV3Plus, self).__init__()
        self.backbone = backbone
        self.backbone_inp = input_shape
        self.head = DeepLabV3PlusHead(nclass)

    def call(self, x):
        # img_size = tf.size(x)[2:]
        print('x shape:', tf.size(x))
        img_size = (self.backbone_inp, self.backbone_inp)
        backbone = get_backbone(self.backbone, include_top=False,
                                weights='imagenet', input_shape=(self.backbone_inp, self.backbone_inp, 3))(x)
        if backbone == 'mobilenet_v2':
            extracted_features = [backbone.get_layer('block_3_').output]
        x = self.head(extracted_features)
        x = layers.UpSampling2D(img_size, interpolation='bilinear')(x)
        return x


def get_deeplab(dataset='pascal_voc', backbone='resnet50s', input_shape=512, **kwargs):
    from datasets import datasets
    model = DeepLabV3Plus(datasets[dataset.lower()].NUM_CLASS, backbone=backbone,
                          input_shape=input_shape, **kwargs)
    return model

# if __name__ == "__main__":
# model = DeepLabV3Plus(2, 'mobilenet_v2', 512)
# print('here', model)
