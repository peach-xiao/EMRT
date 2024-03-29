#  Copyright (c) 2021 PPViT Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This module implements UperNet
Unified Perceptual Parsing for Scene Understanding
<https://arxiv.org/pdf/1807.10221.pdf>
"""

import math
import paddle
import paddle.nn as nn
from src.models.backbones import SwinTransformer
from src.models.backbones import CSwinTransformer
from src.models.backbones import FocalTransformer
from src.models.decoders import UperHead, FCNHead

#
class UperNet(nn.Layer):
    """ UperNet

    Attributes:
        encoder: A backbone network for extract features from image.
        auxi_head: A boolena indicating if we employ the auxilary segmentation head.
        decoder_type: Type of decoder.
        decoder: A decoder module for semantic segmentation.

    """

    def __init__(self, config):
        super(UperNet, self).__init__()
        if config.MODEL.ENCODER.TYPE == "SwinTransformer":
            self.encoder = SwinTransformer(config)
        elif config.MODEL.ENCODER.TYPE == "CSwinTransformer":
            self.encoder = CSwinTransformer(config)
        elif config.MODEL.ENCODER.TYPE == "FocalTransformer":
            self.encoder = FocalTransformer(config)

        self.num_layers = len(config.MODEL.TRANS.STAGE_DEPTHS)  # STAGE_DEPTHS: [2, 2, 6, 2]
        self.auxi_head = config.MODEL.AUX.AUXIHEAD  # True
        self.decoder_type = config.MODEL.DECODER_TYPE  # 'UperHead'
        self.backbone_out_indices = config.MODEL.ENCODER.OUT_INDICES  # [0, 1, 2, 3]

        assert self.decoder_type == "UperHead", "only support UperHead decoder"
        self.num_features = []
        for i in range(self.num_layers):
            self.num_features.append(
                int(config.MODEL.TRANS.EMBED_DIM * 2 ** i))  # EMBED_DIM: 64 输出的通道数目：  96, 192, 288, 384
        self.layer_norms = nn.LayerList()
        for idx in self.backbone_out_indices:  # [0, 1, 2, 3]
            self.layer_norms.append(nn.LayerNorm(self.num_features[idx]))  # nn.LayerNorm(96),(192),(288),(384)

        self.decoder = UperHead(
            pool_scales=config.MODEL.UPERHEAD.POOL_SCALES,  # POOL_SCALES: [1, 2, 3, 6]
            in_channels=config.MODEL.UPERHEAD.IN_CHANNELS,  # [64, 128, 256, 512]
            channels=config.MODEL.UPERHEAD.CHANNELS, # 512
            align_corners=config.MODEL.UPERHEAD.ALIGN_CORNERS, # False
            num_classes=config.DATA.NUM_CLASSES)

        self.auxi_head = config.MODEL.AUX.AUXIHEAD
        if self.auxi_head == True:
            self.aux_decoder = FCNHead(
                in_channels=config.MODEL.AUXFCN.IN_CHANNELS,
                num_classes=config.DATA.NUM_CLASSES,
                up_ratio=config.MODEL.AUXFCN.UP_RATIO)
        self.init__decoder_lr_coef(config)

    def init__decoder_lr_coef(self, config):
        pass

    def to_2D(self, x):
        n, hw, c = x.shape
        h = w = int(math.sqrt(hw))
        x = x.transpose([0, 2, 1]).reshape([n, c, h, w])
        return x

    def forward(self, imgs):
        # imgs.shapes: (B,3,H,W)
        feats = self.encoder(imgs)
        # p2, p3, p4, p5 = feats
        #  print(p2,p3,p4,p5) #shape=[8, 16384, 64],shape=[8, 4096, 128],shape=[8, 1024, 256],shape=[8, 256, 512]

        for idx in self.backbone_out_indices: # [0, 1, 2, 3]
            feat = self.layer_norms[idx](feats[idx]) #nn.LayerNorm(64),(128),(256),(512)
            feats[idx] = self.to_2D(feat)
        p2, p3, p4, p5 = feats
        # print(p2,p3,p4,p5) #shape=[8, 64, 128, 128],shape=[8, 128, 64, 64],shape=[8, 256, 32, 32],shape=[8, 512, 16, 16]

        preds = [self.decoder([p2, p3, p4, p5])] #upernet解码器
        preds.append(self.aux_decoder(p4))
        return preds

        # return  feats #用于pth转换paparams测试