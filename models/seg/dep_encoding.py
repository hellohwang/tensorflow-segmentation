###########################################################################
# Created by: Hang Zhang
# Email: zhang.hang@rutgers.edu
# Copyright (c) 2017
###########################################################################
from __future__ import division
import os
import math
import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import interpolate
from ipdb import set_trace as st
from encoding.nn import SegmentationLosses, SyncBatchNorm

from .base import BaseNet
from .fcn import FCNHead


class DeepLabV3Plus(BaseNet):
    def __init__(self, nclass, backbone, aux=True, se_loss=False, norm_layer=nn.BatchNorm2d, **kwargs):
        super(DeepLabV3Plus, self).__init__(nclass, backbone,
                                            aux, se_loss, norm_layer=norm_layer, **kwargs)
        self.head = DeepLabV3PlusHead(320, nclass, norm_layer, self._up_kwargs)
        # if aux:
        #     self.auxlayer = FCNHead(1024, nclass, norm_layer)

    def forward(self, x):
        img_size = x.size()[2:]
        multix = self.base_forward(x)
        # print(multix[0].shape, multix[1].shape, multix[2].shape, multix[3].shape)
        x = self.head(multix)
        x = interpolate(x, img_size, **self._up_kwargs)
        outputs = [x]
        # if self.aux:
        #     auxout = self.auxlayer(c3)
        #     auxout = interpolate(auxout, (h, w), **self._up_kwargs)
        #     outputs.append(auxout)
        return tuple(outputs)


class DeepLabV3PlusHead(nn.Module):
    def __init__(self, in_channels, nclass, norm_layer, up_kwargs, atrous_rates=[6, 12, 18], **kwargs):
        super(DeepLabV3PlusHead, self).__init__()
        self.aspp = ASPP(in_channels, atrous_rates, norm_layer)
        self.decoder = Decoder(nclass, norm_layer, up_kwargs)

    def forward(self, x):
        c = self.aspp(x[0])
        x = self.decoder(x[1], x[2], x[3], c)
        return x


class Decoder(nn.Module):
    def __init__(self, out_channels, norm_layer=nn.BatchNorm2d, up_kwargs=None):
        super(Decoder, self).__init__()
        self._up_kwargs = up_kwargs
        self.first_conv = ConvBNReLU(
            144, 48, 3, padding=1, norm_layer=norm_layer)
        self.second_conv = ConvBNReLU(
            192, 48, 3, padding=1, norm_layer=norm_layer)
        self.third_conv = ConvBNReLU(
            576, 48, 3, padding=1, norm_layer=norm_layer)
        self.gamma = nn.Parameter(torch.ones(1))
        self.att = nn.Sequential(nn.Conv2d(288, 1, 1),
                                 nn.Sigmoid())
        self.conv = nn.Sequential(
            ConvBNReLU(304, 256, 3, padding=1, norm_layer=norm_layer),
            ConvBNReLU(256, 256, 3, padding=1, norm_layer=norm_layer))
        self.last_conv = nn.Sequential(
            ConvBNReLU(304, 256, 3, padding=1, norm_layer=norm_layer),
            nn.Dropout(0.5),
            ConvBNReLU(256, 256, 3, padding=1, norm_layer=norm_layer),
            nn.Dropout(0.1),
            nn.Conv2d(256, out_channels, kernel_size=1, stride=1))

    def forward(self, c1, c2, c3, x):  # x channels:256 aspp output
        fourth_feature = self.first_conv(c1)  # 1/4
        eighth_feature = self.second_conv(c2)  # 1/8
        sixteenth_feature = self.third_conv(c3)  # 1/16
        x = self.conv(torch.cat((x, sixteenth_feature), dim=1))
        x = interpolate(x, eighth_feature.size()[
                        2:], **self._up_kwargs)  # upsample to 1/8
        x = self.conv(torch.cat((x, eighth_feature), dim=1))
        x = interpolate(x, fourth_feature.size()[
                        2:], **self._up_kwargs)  # upsample to 1/4

        x = self.last_conv(torch.cat((x, fourth_feature), dim=1))
        return x


class ASPPConv(nn.Module):
    def __init__(self, in_channels, out_channels, atrous_rate, norm_layer):
        super(ASPPConv, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3,
                      padding=atrous_rate, dilation=atrous_rate, bias=False),
            norm_layer(out_channels),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.block(x)


class AsppPooling(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer):
        super(AsppPooling, self).__init__()
        self.gap = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            norm_layer(out_channels),
            nn.ReLU(True)
        )

    def forward(self, x):
        size = x.size()[2:]
        pool = self.gap(x)
        out = interpolate(pool, size, mode='bilinear', align_corners=True)
        return out


class ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates, norm_layer):
        super(ASPP, self).__init__()
        out_channels = 256
        self.b0 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            norm_layer(out_channels),
            nn.ReLU(True)
        )

        rate1, rate2, rate3 = tuple(atrous_rates)
        self.b1 = ASPPConv(in_channels, out_channels, rate1, norm_layer)
        self.b2 = ASPPConv(in_channels, out_channels, rate2, norm_layer)
        self.b3 = ASPPConv(in_channels, out_channels, rate3, norm_layer)
        self.b4 = AsppPooling(in_channels, out_channels, norm_layer=norm_layer)

        self.project = nn.Sequential(
            nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),
            norm_layer(out_channels),
            nn.ReLU(True),
            nn.Dropout(0.5)
        )

    def forward(self, x):
        feat1 = self.b0(x)
        feat2 = self.b1(x)
        feat3 = self.b2(x)
        feat4 = self.b3(x)
        feat5 = self.b4(x)
        x = torch.cat((feat1, feat2, feat3, feat4, feat5), dim=1)
        x = self.project(x)
        return x


class ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, relu6=False, norm_layer=nn.BatchNorm2d, **kwargs):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride, padding, dilation, groups, bias=False)
        self.bn = norm_layer(out_channels)
        self.relu = nn.ReLU6(True) if relu6 else nn.ReLU(True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


def get_deeplab(dataset='pascal_voc', backbone='resnet50', pretrained=False,
                root='/data3/hwang/proj/PyTorch-Encoding/models', **kwargs):
    acronyms = {
        'pascal_voc': 'voc',
        'pascal_aug': 'voc',
        'ade20k': 'ade',
    }
    # infer number of classes
    from ..datasets import datasets, VOCSegmentation, VOCAugSegmentation, ADE20KSegmentation
    model = DeepLabV3Plus(
        datasets[dataset.lower()].NUM_CLASS, backbone=backbone, root=root, **kwargs)
    # model = DeepLabV3(datasets[dataset.lower()].NUM_CLASS, backbone=backbone, root=root, **kwargs)
    if pretrained:
        from .model_store import get_model_file
        model.load_state_dict(torch.load(
            get_model_file('deeplab_%s_%s' % (backbone, acronyms[dataset]), root=root)))
    return model
