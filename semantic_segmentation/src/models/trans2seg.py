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
import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from src.models.decoders.fcn_head import FCNHead
from src.models.decoders import ConvBNReLU, SeparableConv2d, CNNHEAD, HybridEmbed
from .backbones import get_segmentation_backbone, TransformerEncoder, TransformerDecoder, expand


class Trans2Seg(nn.Layer):
    """Trans2Seg Implement
    
    It contains cnn-encoder, transformer-encoder and transformer-decoder, and a small-cnn-head
    Ref, https://arxiv.org/pdf/2101.08461.pdf

    """

    def __init__(self, config):
        super(Trans2Seg, self).__init__()
        c1_channels = 256
        c4_channels = 2048
        self.nclass = config.DATA.NUM_CLASSES
        self.aux = config.MODEL.AUX.AUXIHEAD  # AUXIHEAD: False
        self.backbone = config.MODEL.ENCODER.TYPE.lower()  # "resnet50c"

        # Create cnn encoder, the input image is fed to CNN to extract features
        self.cnn_encoder = get_segmentation_backbone(self.backbone, config, nn.BatchNorm2D)

        # Get vit hyper params
        vit_params = config.MODEL.TRANS2SEG
        hid_dim = config.MODEL.TRANS2SEG.HID_DIM  # 64

        c4_HxW = (config.DATA.CROP_SIZE[0] // 16) ** 2  # 512//16 **2 = 1024
        vit_params['decoder_feat_HxW'] = c4_HxW

        last_channels = vit_params['EMBED_DIM']  # 256

        # create transformer encoder, for transformer encoder,
        # the features and position embedding are flatten and fed to transformer for self-attention,
        # and output feature(Fe) from transformer encoder.
        self.transformer_encoder = TransformerEncoder(
            embed_dim=last_channels,  # 256
            depth=vit_params['DEPTH'],  # 4
            num_heads=vit_params['NUM_HEADS'],  # 8
            mlp_ratio=vit_params['MLP_RATIO'])  # 3
        # create transformer decoder, for transformer decoder,
        # for transformer decoder, we specifically define a set of learnable class prototype embeddings as query,
        # the features from transformer encoder as key
        self.transformer_decoder = TransformerDecoder(
            nclass=config.DATA.NUM_CLASSES,
            embed_dim=last_channels,  # 256
            depth=vit_params['DEPTH'],  # 4
            num_heads=vit_params['NUM_HEADS'],  # 8
            mlp_ratio=vit_params['MLP_RATIO'],  # 3
            decoder_feat_HxW=vit_params['decoder_feat_HxW'])  # 256
        # Create Hybrid Embedding
        self.hybrid_embed = HybridEmbed(c4_channels, last_channels)  # x.flatten(2).transpose([0, 2, 1]),linear 2048--> 256
        # Create small Conv Head, a small conv head to fuse attention map and Res2 feature from CNN backbone
        self.cnn_head = CNNHEAD(vit_params, c1_channels=c1_channels, hid_dim=hid_dim)  # c1_channels =256,hid_dim = 64

        if self.aux:  # false
            self.auxlayer = FCNHead(in_channels=728, channels=728 // 4, num_classes=self.nclass)

    def forward(self, x):
        size = x.shape[2:]
        c1, c2, c3, c4 = self.cnn_encoder(x)  # C4 = [8, 2048, 32, 32]
        # c1 c2 c3 c4 shape=[8, 256, 128, 128],[8, 512, 64, 64],[8, 1024, 32, 32],[8, 2048, 32, 32]

        outputs = list()
        n, _, h, w = c4.shape
        c4 = self.hybrid_embed(c4)  # 展开特征图 C4 = [8, 2048, 32, 32] -> falstten ->(8, 1024, 256) B H*W,C ,C是通过LinerLayer 2048->256
        cls_token, c4 = self.transformer_encoder.forward_encoder(c4)
        # print(np.array(cls_token).shape,np.array(c4).shape)#(8, 256), (8, 1024, 256)

        attns_list = self.transformer_decoder.forward_decoder(c4)  # c4 = [8, 1024, 256]
        feat_enc = c4.reshape([n, h, w, -1]).transpose([0, 3, 1, 2])  # [8, 256, 32, 32]

        attn_map = attns_list[-1]  # [8, 6, 8, 1024] 通过transformer类别嵌入的decorder学习到的一个注意力地图
        B, nclass, nhead, _ = attn_map.shape
        _, _, H, W = feat_enc.shape
        attn_map = attn_map.reshape([B * nclass, nhead, H, W])  # shape=[48, 8, 32, 32],
        x = paddle.concat([expand(feat_enc, nclass), attn_map], 1)  # shape=[48, 264, 32, 32]
        # expand --> return x.unsqueeze(1).tile([1, nclass, 1, 1, 1]).flatten(0, 1)
        # paddle.tile()根据参数 repeat_times 对输入 x 的各维度进行复制。 平铺后，输出的第 i 个维度的值等于 x.shape[i]*repeat_times[i]
        # print("expand(feat_enc, nclass)", expand(feat_enc, nclass)) #[48, 256, 32, 32]
        x = self.cnn_head(x, c1, nclass, B)  # shape=[8, 6, 128, 128] 通过Separable Conv2D
        x = F.interpolate(x, size, mode='bilinear', align_corners=True)

        outputs.append(x)
        if self.aux:
            auxout = self.auxlayer(c3)
            auxout = F.interpolate(auxout, size, mode='bilinear', align_corners=True)
            outputs.append(auxout)
        return tuple(outputs)
