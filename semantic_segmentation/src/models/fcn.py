# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from src.utils import load_entire_model
from .backbones import get_segmentation_backbone


class FCN(nn.Layer):
    def __init__(self, config):
        super(FCN, self).__init__()

        self.align_corners = False
        self.pretrained = config.MODEL.PRETRAINED
        self.backbone = config.MODEL.ENCODER.TYPE.lower()
        self.num_classes = config.DATA.NUM_CLASSES

        self.cnn_encoder = get_segmentation_backbone(self.backbone, config, nn.BatchNorm2D)
        self.head = FCNHead(self.num_classes, in_channel=2048, channel=256, bias=True)

    def forward(self, x):
        c1, c2, c3, c4 = self.cnn_encoder(x)
        logit = self.head(c4)
        outputs = []
        out = F.interpolate(logit, paddle.shape(x)[2:], mode='bilinear', align_corners=self.align_corners)
        outputs.append(out)
        return outputs



class FCNHead(nn.Layer):

    def __init__(self, num_classes, in_channel=None, channel=None, bias=True):
        super(FCNHead, self).__init__()

        self.num_classes = num_classes
        self.in_channel = in_channel
        self.channel = channel

        self.conv_1 = nn.Sequential(
            nn.Conv2D(self.in_channel, self.channel, kernel_size=1, stride=1, padding="same", bias_attr=bias),
            nn.SyncBatchNorm(self.channel, data_format='NCHW'),
            nn.ReLU())

        self.cls = nn.Conv2D(in_channels=self.channel, out_channels=self.num_classes, kernel_size=1, stride=1,
                             bias_attr=bias)

    def forward(self, x):
        x = self.conv_1(x)
        logit = self.cls(x)
        return logit
