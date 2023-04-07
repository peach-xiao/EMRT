import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import numpy as np
from .backbones import hrnet
from .EMRT_utils.transformer_encoder_decoder import EncoderDecoder
from src.models.decoders.fcn_head import FCNHead

class Conv2dBlock(nn.Layer):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(Conv2dBlock, self).__init__()
        self.conv = nn.Conv2D(in_channels, out_channels, kernel_size, stride, padding, bias_attr=False)
        self.norm = nn.BatchNorm2D(out_channels)
        self.relu = nn.ReLU(True)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.relu(x)
        return x

class EFP(nn.Layer):
    def __init__(self, in_channels=256, out_channels=256):
        super(EFP, self).__init__()
        self.conv0 = Conv2dBlock(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv1 = Conv2dBlock(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = Conv2dBlock(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x0, x1, x2):
        x_out2 = self.conv2(x2)
        x_out2 = F.interpolate(x_out2, size=x1.shape[2:], mode='bilinear', align_corners=True)

        x_out1 = self.conv1(x1)
        x_out21 = x_out1 + x_out2
        x_out21 = F.interpolate(x_out21, size=x0.shape[2:], mode='bilinear', align_corners=True)

        x_out0 = self.conv0(x0)
        x_out = x_out0 + x_out21
        return x_out

class PyramidPoolingModule(nn.Layer):
    def __init__(self, pool_scales, in_channels, channels, align_corners=False):
        super(PyramidPoolingModule, self).__init__()
        self.pool_scales = pool_scales
        self.in_channels = in_channels
        self.channels = channels
        self.align_corners = align_corners

        norm_bias_attr = paddle.ParamAttr(initializer=paddle.nn.initializer.Constant(0.0))

        self.pool_branches = nn.LayerList()
        for idx in range(len(self.pool_scales)):
            self.pool_branches.append(nn.Sequential(
                nn.AdaptiveAvgPool2D(self.pool_scales[idx]),
                nn.Conv2D(self.in_channels, self.channels, 1, stride=1, bias_attr=False),
                nn.SyncBatchNorm(self.channels, weight_attr=self.get_norm_weight_attr(), bias_attr=norm_bias_attr),
                nn.ReLU()))

    def get_norm_weight_attr(self):
        return paddle.ParamAttr(initializer=paddle.nn.initializer.Constant(value=1.0))

    def forward(self, x):
        n, c, _, _ = x.shape
        outs = []
        for _, pool_layer in enumerate(self.pool_branches):
            out = pool_layer(x)
            reshape_out = out.reshape([n, c, -1])
            outs.append(reshape_out)
        center = paddle.concat(outs, axis=-1)
        return center

class branch_block(nn.Layer):
    def __init__(self, in_channels, out_channels, downsample=True):
        super(branch_block, self).__init__()
        self.downsample = downsample
        self.maxpool = nn.MaxPool2D(kernel_size=3, stride=2, padding=1)
        self.encode = nn.Sequential(
            nn.Conv2D(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias_attr=False),
            nn.BatchNorm2D(out_channels),
            nn.ReLU(True),
            nn.Conv2D(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias_attr=False),
            nn.BatchNorm2D(out_channels),
            nn.ReLU(True))

    def forward(self, x):
        if self.downsample:
            x = self.maxpool(x)
        x = self.encode(x)
        return x

class spatial_branch(nn.Layer):
    def __init__(self, in_channels=3):
        super(spatial_branch, self).__init__()

        self.Enc0 = branch_block(in_channels, 64)
        self.Enc1 = branch_block(64, 128)
        self.Enc2 = branch_block(128, 256)
        # self.Enc3 = branch_block(256, 256)

    def forward(self, x):
        enc0 = self.Enc0(x)
        enc1 = self.Enc1(enc0)
        enc2 = self.Enc2(enc1)
        # enc3 = self.Enc3(enc2)
        return enc2

class UpHead(nn.Layer):
    def __init__(self, embed_dim=256, num_conv=1, num_upsample_layer=1,
                 conv3x3_conv1x1=True, align_corners=False, num_classes=6):
        super(UpHead, self).__init__()
        self.num_classes = num_classes
        self.align_corners = align_corners
        self.num_conv = num_conv
        self.num_upsample_layer = num_upsample_layer
        self.conv3x3_conv1x1 = conv3x3_conv1x1

        if self.num_conv == 2:
            if self.conv3x3_conv1x1:
                self.conv_0 = nn.Conv2D(embed_dim, 256, 3, stride=1, padding=1, bias_attr=True)
            else:
                self.conv_0 = nn.Conv2D(embed_dim, 256, 1, stride=1, bias_attr=True)
            self.conv_1 = nn.Conv2D(256, self.num_classes, 1, stride=1)
            self.syncbn_fc_0 = nn.BatchNorm2D(256)

        elif self.num_conv == 3:
            self.conv_0 = nn.Conv2D(embed_dim, 256, kernel_size=3, stride=1, padding=1)
            self.conv_1 = nn.Conv2D(256, 256, kernel_size=3, stride=1, padding=1)
            self.conv_2 = nn.Conv2D(256, 256, kernel_size=3, stride=1, padding=1)
            # self.conv_3 = nn.Conv2D(256, 256, kernel_size=3, stride=1, padding=1)
            self.conv_3 = nn.Conv2D(256, self.num_classes, kernel_size=1, stride=1)
            self.syncbn_fc_0 = nn.BatchNorm2D(256)
            self.syncbn_fc_1 = nn.BatchNorm2D(256)
            self.syncbn_fc_2 = nn.BatchNorm2D(256)
            # self.syncbn_fc_3 = nn.BatchNorm2D(256)

    def forward(self, x):
        up2x_resolution = [2 * item for item in x.shape[2:]]
        up4x_resolution = [4 * item for item in x.shape[2:]]
        up8x_resolution = [8 * item for item in x.shape[2:]]
        if self.num_conv == 2:
            if self.num_upsample_layer == 2:
                x = self.conv_0(x)
                x = self.syncbn_fc_0(x)
                x = F.relu(x)
                x = F.interpolate(x, up4x_resolution, mode='bilinear', align_corners=self.align_corners)
                x = self.conv_1(x)
                x = F.interpolate(x, up2x_resolution, mode='bilinear', align_corners=self.align_corners)

            elif self.num_upsample_layer == 1:
                x = self.conv_0(x)
                x = self.syncbn_fc_0(x)
                x = F.relu(x)
                x = self.conv_1(x)
                x = F.interpolate(x, up8x_resolution, mode='bilinear', align_corners=self.align_corners)

        elif self.num_conv == 3:
            x = self.conv_0(x)
            x = self.syncbn_fc_0(x)
            x = F.relu(x)
            up2x_resolution = [2 * item for item in x.shape[2:]]
            x = F.interpolate(x, up2x_resolution, mode='bilinear', align_corners=self.align_corners)
            x = self.conv_1(x)
            x = self.syncbn_fc_1(x)
            x = F.relu(x)
            up2x_resolution = [2 * item for item in x.shape[2:]]
            x = F.interpolate(x, up2x_resolution, mode='bilinear', align_corners=self.align_corners)
            x = self.conv_2(x)
            x = self.syncbn_fc_2(x)
            x = F.relu(x)
            x = self.conv_3(x)
            up2x_resolution = [2 * item for item in x.shape[2:]]
            x = F.interpolate(x, up2x_resolution, mode='bilinear', align_corners=self.align_corners)
        return x

class EMRT_HRNet(nn.Layer):
    def __init__(self, config):
        super(EMRT_HRNet, self).__init__()

        self.nclass = config.DATA.NUM_CLASSES
        self.backbone = config.MODEL.ENCODER.TYPE.lower()
        self.backbone_num_channels = [144, 336, 720]
        self.hidden_dim = 256
        self.psp_scale = [1, 3, 6, 8]

        self.spatial_branch = spatial_branch(in_channels=3)
        self.psp_module = PyramidPoolingModule(pool_scales=self.psp_scale, in_channels=256, channels=256)
        self.uphead = UpHead(embed_dim=256, num_conv=3, num_upsample_layer=1, align_corners=False,
                             num_classes=self.nclass)

        self.input_proj = nn.LayerList()
        for in_channels in self.backbone_num_channels:
            self.input_proj.append(
                nn.Sequential(
                    nn.Conv2D(in_channels, self.hidden_dim, kernel_size=3, stride=2, padding=1, bias_attr=False),
                    nn.BatchNorm2D(self.hidden_dim),
                    nn.ReLU(True)
                ))

        self.cls_psp = nn.Sequential(
            nn.Conv2D(self.hidden_dim * (2 + len(self.psp_scale)), 512, kernel_size=3, padding=1, bias_attr=False),
            nn.BatchNorm2D(512),
            nn.ReLU(),
            nn.Conv2D(512, 256, kernel_size=3, padding=1, bias_attr=False),
            nn.BatchNorm2D(256),
            nn.ReLU(),
            nn.Dropout2D(p=0.1),
            # nn.Conv2D(512, self.nclass, kernel_size=1)
        )

        self.EFP = EFP(in_channels=256, out_channels=256)
        self.auxlayer = FCNHead(in_channels=336, channels=336 // 4, num_classes=self.nclass)

        # for paddle init weight:
        for m in self.sublayers():
            if isinstance(m, (nn.Conv2D, nn.Conv2DTranspose)):
                m.weight = paddle.create_parameter(shape=m.weight.shape,
                                                   dtype='float32', default_initializer=nn.initializer.KaimingNormal())
            elif isinstance(m, nn.BatchNorm2D):
                m.weight = paddle.create_parameter(shape=m.weight.shape, dtype='float32',
                                                   default_initializer=nn.initializer.Constant(value=1.0))
                m.bias = paddle.create_parameter(shape=m.bias.shape, dtype='float32',
                                                 default_initializer=nn.initializer.Constant(value=0.0))

        if self.backbone == "hrnet":
            self.backbone = hrnet.HRNet_W48(pretrained=config.MODEL.PRETRAINED)

        self.model = EncoderDecoder(hidden_dim=256, dim_feedforward=1024,
                                    backbone_num_channels=[256, 256, 256],  # hrnet
                                    dropout=0.1, activation='relu',
                                    num_feature_levels=3, nhead=8,
                                    num_encoder_layers=4, num_decoder_layers=2,
                                    num_encoder_points=6, num_decoder_points=6,
                                    nclass=self.nclass)

    def forward(self, inputs):
        bs, c, h, w = inputs.shape
        c1, st2, st3, st4 = self.backbone(inputs)

        size = paddle.shape(st2[0])[2:]
        x1 = F.interpolate(
            st2[1], size, mode='bilinear', align_corners=False)
        c2 = paddle.concat([st2[0], x1], axis=1)

        # layer3
        size = paddle.shape(st3[0])[2:]
        x1 = F.interpolate(
            st3[1], size, mode='bilinear', align_corners=False)
        x2 = F.interpolate(
            st3[2], size, mode='bilinear', align_corners=False)
        c3 = paddle.concat([st3[0], x1, x2], axis=1)

        # layer4
        size = paddle.shape(st4[0])[2:]
        x1 = F.interpolate(
            st4[1], size, mode='bilinear', align_corners=False)
        x2 = F.interpolate(
            st4[2], size, mode='bilinear', align_corners=False)
        x3 = F.interpolate(
            st4[3], size, mode='bilinear', align_corners=False)
        c4 = paddle.concat([st4[0], x1, x2, x3], axis=1)

        x_fea = []
        # x_fea.append(c1)
        x_fea.append(self.input_proj[0](c2))
        x_fea.append(self.input_proj[1](c3))
        x_fea.append(self.input_proj[2](c4))

        x_context = self.spatial_branch(inputs)
        x_psp = self.psp_module(x_context)

        x_trans, memory = self.model(x_fea, x_psp)
        x_trans = x_trans.squeeze(0).transpose([0, 2, 1])

        x0_index = x_fea[0].shape[-1] * x_fea[0].shape[-2]
        x1_index = x_fea[1].shape[-1] * x_fea[1].shape[-2]
        x2_index = x_fea[2].shape[-1] * x_fea[2].shape[-2]

        x0 = memory[:, 0:x0_index].transpose([0, 2, 1]).reshape(
            [x_fea[0].shape[0], 256, x_fea[0].shape[-2], x_fea[0].shape[-1]])
        x1 = memory[:, x0_index:x0_index + x1_index].transpose([0, 2, 1]).reshape(
            [x_fea[1].shape[0], 256, x_fea[1].shape[-2], x_fea[1].shape[-1]])
        x2 = memory[:, x0_index + x1_index::].transpose([0, 2, 1]).reshape(
            [x_fea[2].shape[0], 256, x_fea[2].shape[-2], x_fea[2].shape[-1]])

        x_fpn = self.EFP(x0, x1, x2)

        # psp moudle
        psp_idx = 0
        psp_cat = x_context
        bs, ctx_c, ctx_h, ctx_w = x_context.shape
        for i in self.psp_scale:
            square = i ** 2
            pooled_output = x_trans[:, :, psp_idx:psp_idx + square].reshape([bs, ctx_c, i, i])
            # print("pooled_output", i, pooled_output.shape)  # [4, 256, 1, 1] ,.. 3,3,..6,6,..8,8
            pooled_resized_output = F.interpolate(pooled_output, size=x_context.shape[2:], mode='bilinear',
                                                  align_corners=True)
            psp_cat = paddle.concat([psp_cat, pooled_resized_output], 1)

            psp_idx = psp_idx + square

        psp_cat = paddle.concat([psp_cat, x_fpn], 1)
        x_out = self.cls_psp(psp_cat)  # 256*6 --> 256
        x = self.uphead(x_out)

        outputs = list()
        outputs.append(x)
        auxout = self.auxlayer(c3)
        auxout = F.interpolate(auxout, inputs.shape[2:], mode='bilinear', align_corners=True)
        outputs.append(auxout)

        return tuple(outputs)
