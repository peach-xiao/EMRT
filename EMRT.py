import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import numpy as np
from .backbones import get_segmentation_backbone  # for resnet50c
from .backbones import paddle_vision_resnet as resnet

from .DeformableTrans_utils_paddle.deformable_transformer import DeformableTransformer
from src.models.decoders.fcn_head import FCNHead
from .DeformableTrans_utils_paddle.deformable_head import CNNHEAD


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


class ResBlock(nn.Layer):

    def __init__(self, inplanes, planes):
        super(ResBlock, self).__init__()
        self.resconv1 = Conv2dBlock(inplanes, planes, kernel_size=3, stride=1, padding=1)
        self.resconv2 = Conv2dBlock(planes, planes, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        residual = x
        out = self.resconv1(x)
        out = self.resconv2(out)
        out = out + residual

        return out


class FinalBlock(nn.Layer):

    def __init__(self, inplanes, hid_dim=64):
        super(FinalBlock, self).__init__()
        self.conv1 = nn.Conv2D(inplanes, inplanes, kernel_size=3, stride=1, padding=1)
        self.norm1 = nn.BatchNorm2D(inplanes)
        self.relu1 = nn.ReLU(True)
        self.conv2 = nn.Conv2D(inplanes, hid_dim, kernel_size=3, stride=1, padding=1)
        self.norm2 = nn.BatchNorm2D(hid_dim)
        self.relu2 = nn.ReLU(True)
        self.pred = nn.Conv2D(hid_dim, 1, 1)

    def forward(self, x, B, nclass):
        size = x.shape[2:]
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.norm2(out)
        out = self.relu2(out)

        out = self.pred(out).reshape([B, nclass, size[0], size[1]])

        return out


def expand(x, nclass):
    return x.unsqueeze(1).tile([1, nclass, 1, 1, 1]).flatten(0, 1)


class PyramidPoolingModule(nn.Layer):
    def __init__(self, pool_scales, in_channels, channels, align_corners=False):
        super(PyramidPoolingModule, self).__init__()
        self.pool_scales = pool_scales  # [1, 2, 3, 6]
        self.in_channels = in_channels  # 768
        self.channels = channels  # 512
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


class _EncoderBlock(nn.Layer):
    def __init__(self, in_channels, out_channels, downsample=True):
        super(_EncoderBlock, self).__init__()
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
    def __init__(self, in_channels=3, num_classes=1):
        super(spatial_branch, self).__init__()

        self.Enc0 = _EncoderBlock(in_channels, 64, downsample=False)
        self.Enc1 = _EncoderBlock(64, 128)
        self.Enc2 = _EncoderBlock(128, 256)
        self.Enc3 = _EncoderBlock(256, 256)

    def forward(self, x):
        enc0 = self.Enc0(x)  # (conv 3x3 + bn +relu) x2
        enc1 = self.Enc1(enc0)
        enc2 = self.Enc2(enc1)
        enc3 = self.Enc3(enc2)

        return enc3


class ConvBlock(nn.Layer):
    def __init__(self, in_channels, out_channels, activation=True, kernel_size=3, stride=2, padding=1):
        super().__init__()
        self.conv1 = nn.Conv2D(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                               bias_attr=False)
        self.bn = nn.SyncBatchNorm(out_channels)
        self.activation = activation
        if activation:
            self.relu = nn.ReLU()

    def forward(self, input):
        x = self.conv1(input)
        if self.activation:
            return self.relu(self.bn(x))
        else:
            return self.bn(x)


class spatial_branch1(nn.Layer):
    def __init__(self):
        super().__init__()
        self.convblock1 = ConvBlock(in_channels=3, out_channels=64)
        self.convblock2 = ConvBlock(in_channels=64, out_channels=128)
        self.convblock3 = ConvBlock(in_channels=128, out_channels=256, activation=False)

    def forward(self, input):
        x = self.convblock1(input)
        x = self.convblock2(x)
        x = self.convblock3(x)
        return x


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
                x = self.conv_1(x)  #-->class
                x = F.interpolate(x, up2x_resolution, mode='bilinear', align_corners=self.align_corners)

            elif self.num_upsample_layer == 1:
                x = self.conv_0(x)
                x = self.syncbn_fc_0(x)
                x = F.relu(x)
                x = self.conv_1(x)  #-->class
                x = F.interpolate(x, up8x_resolution, mode='bilinear', align_corners=self.align_corners)

        elif self.num_conv == 3:
            x = self.conv_0(x)
            x = self.syncbn_fc_0(x)
            x = F.relu(x)
            up2x_resolution = [2 * item for item in x.shape[2:]]
            x = F.interpolate(x, up2x_resolution, mode='bilinear', align_corners=self.align_corners)
            # print("x1", x.shape) #[8, 256, 64, 64]
            x = self.conv_1(x)
            x = self.syncbn_fc_1(x)
            x = F.relu(x)
            up2x_resolution = [2 * item for item in x.shape[2:]]
            x = F.interpolate(x, up2x_resolution, mode='bilinear', align_corners=self.align_corners)
            # print("x2", x.shape) #[8, 256, 128, 128]
            x = self.conv_2(x)
            x = self.syncbn_fc_2(x)
            x = F.relu(x)
            x = self.conv_3(x)  # --->class
            up2x_resolution = [2 * item for item in x.shape[2:]]
            x = F.interpolate(x, up2x_resolution, mode='bilinear', align_corners=self.align_corners)

        return x


class DeformableTranNet(nn.Layer):
    def __init__(self, config):
        super(DeformableTranNet, self).__init__()

        self.nclass = config.DATA.NUM_CLASSES
        self.backbone = config.MODEL.ENCODER.TYPE.lower()  # "resnet50c"
        self.backbone_num_channels = [512, 1024, 2048]
        self.hidden_dim = 256
        self.psp_scale = [1, 3, 6, 8]
        # self.psp_scale = [1, 2, 4, 8]  #[1, 2, 3, 6] -50, [1, 2, 4, 8] -85 ,  [1, 3, 6, 8]-110 , [1, 4, 8, 12] -225

        # self.transposeconv_stage2 = nn.Conv2DTranspose(256, 256, kernel_size=2, stride=2, bias_attr=False)
        # self.transposeconv_stage1 = nn.Conv2DTranspose(256, 128, kernel_size=2, stride=2, bias_attr=False)
        # self.transposeconv_stage0 = nn.Conv2DTranspose(128, 64, kernel_size=2, stride=2, bias_attr=False)

        self.spatial_branch = spatial_branch()
        self.feat_proj = nn.Conv2D(2048, self.hidden_dim, kernel_size=1)

        self.psp_module = PyramidPoolingModule(pool_scales=self.psp_scale, in_channels=256, channels=256)
        self.uphead = UpHead(embed_dim=256, num_conv=3, num_upsample_layer=1, align_corners=False, num_classes= self.nclass)
        # self.conv_2 = nn.Conv2D(256, 256, kernel_size=3, stride=1, padding=1)
        # self.conv_3 = nn.Conv2D(256, 256, kernel_size=3, stride=1, padding=1)

        self.final_head = FinalBlock(inplanes=256, hid_dim=64)
        self.bn_0 = nn.BatchNorm2D(256)
        # self.bn_1 = nn.BatchNorm2D(256)
        self.input_proj = nn.LayerList()
        for in_channels in self.backbone_num_channels:
            self.input_proj.append(
                nn.Sequential(
                    nn.Conv2D(in_channels, self.hidden_dim, kernel_size=1),
                    nn.BatchNorm2D(self.hidden_dim)
                ))

        self.cls_psp = nn.Sequential(
            nn.Conv2D(self.hidden_dim*(2+len(self.psp_scale)), 512, kernel_size=3, padding=1,  bias_attr=False),
            nn.BatchNorm2D(512),
            nn.ReLU(),

            nn.Conv2D(512, 256, kernel_size=3, padding=1, bias_attr=False),
            nn.BatchNorm2D(256),
            nn.ReLU(),
            nn.Dropout2D(p=0.1),
            # nn.Conv2D(512, self.nclass, kernel_size=1)
        )

        self.cls = nn.Sequential(
            nn.Conv2D(256, 256, kernel_size=3, padding=1, bias_attr=False),
            nn.BatchNorm2D(256),
            nn.ReLU(),
            # nn.Dropout2D(p=0.1),
            nn.Conv2D(256, self.nclass, kernel_size=1)
        )

        self.conv_0 = nn.Conv2D(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv_1 = nn.Conv2D(256, self.nclass, kernel_size=3, stride=1, padding=1)

        self.ResBlock0 = ResBlock(256, 256)
        self.ResBlock1 = ResBlock(256, 256)
        self.ResBlock2 = ResBlock(256, 256)

        self.auxlayer = FCNHead(in_channels=1024, channels=1024 // 4, num_classes=self.nclass)

        # for paddle init weight:
        # for m in self.modules(): #paddle 是self.sublayers(), pytorch是self.modules()
        for m in self.sublayers():
            if isinstance(m, (nn.Conv2D, nn.Conv2DTranspose)):
                m.weight = paddle.create_parameter(shape=m.weight.shape,
                                                   dtype='float32', default_initializer=nn.initializer.KaimingNormal())
            elif isinstance(m, nn.BatchNorm2D):
                m.weight = paddle.create_parameter(shape=m.weight.shape, dtype='float32',
                                                   default_initializer=nn.initializer.Constant(value=1.0))
                m.bias = paddle.create_parameter(shape=m.bias.shape, dtype='float32',
                                                 default_initializer=nn.initializer.Constant(value=0.0))

        if self.backbone == "resnet50c":
            self.backbone = get_segmentation_backbone(self.backbone, config, nn.BatchNorm2D)
        elif self.backbone == "resnet50": #for paddle vision resnet
            self.backbone = resnet.resnet50()
        elif self.backbone == "resnet101":
            self.backbone = resnet.resnet101()

        self.encoder_Detrans = DeformableTransformer(hidden_dim=256, dim_feedforward=1024,
                                                     backbone_num_channels=[512, 1024, 2048],
                                                     dropout=0.1, activation='relu',
                                                     num_feature_levels=3, nhead=8, num_encoder_layers=6,
                                                     num_encoder_points=4, num_decoder_points=4, nclass=self.nclass)

    def forward(self, inputs):
        bs, c, h, w = inputs.shape
        c1, c2, c3, c4 = self.backbone(inputs)  # [x0, x1, x2, x3]
        #  c1 c2 c3 c4 shape=[8, 256, 64, 64],[8, 512, 32, 32],[8, 1024, 16, 16],[8, 2048, 8, 8]
        x_fea = []
        # x_fea.append(c1)
        x_fea.append(c2)
        x_fea.append(c3)
        x_fea.append(c4)

        x_context = self.spatial_branch(inputs)
        # print("x_context", x_context.shape) #[4, 256, 32, 32]

        x_psp = self.psp_module(x_context)
        # print("psp_size", self.psp_scale)
        # print("x_psp", x_psp.shape) #[4, 256, 110]: 1x1 + 3x3 + 6x6 + 8x8

        x_trans, memory = self.encoder_Detrans(x_fea, x_psp)  # [1, 4, 110, 256],[8, 1344, 256],1344 = 8*8+16*16+32*32
        x_trans = x_trans.squeeze(0).transpose([0, 2, 1])  # [4, 256, 110]

        x0_index = x_fea[0].shape[-1] * x_fea[0].shape[-2]  # 32 * 32 = 1024
        x1_index = x_fea[1].shape[-1] * x_fea[1].shape[-2]  # 16 * 16 = 256
        x2_index = x_fea[2].shape[-1] * x_fea[2].shape[-2]  # 8 * 8 =64

        x0 = memory[:, 0:x0_index].transpose([0, 2, 1]).reshape(
            [x_fea[0].shape[0], 256, x_fea[0].shape[-2], x_fea[0].shape[-1]])
        x1 = memory[:, x0_index:x0_index + x1_index].transpose([0, 2, 1]).reshape(
            [x_fea[1].shape[0], 256, x_fea[1].shape[-2], x_fea[1].shape[-1]])
        x2 = memory[:, x0_index + x1_index::].transpose([0, 2, 1]).reshape(
            [x_fea[2].shape[0], 256, x_fea[2].shape[-2], x_fea[2].shape[-1]])


        x_fpn = self.ResBlock2(x2)
        x_fpn = F.interpolate(x_fpn, size=x1.shape[2:], mode='bilinear', align_corners=True)

        x_fpn = self.ResBlock1(x_fpn + x1)
        x_fpn = F.interpolate(x_fpn, size=x0.shape[2:], mode='bilinear', align_corners=True)

        x_fpn = self.ResBlock0(x_fpn + x0) # [8, 256, 32, 32]

        # psp moudle
        psp_idx = 0
        psp_cat = x_context
        bs, ctx_c, ctx_h, ctx_w = x_context.shape
        for i in self.psp_scale:  # (1, 3, 6, 8)([B, n_class, C])
            square = i ** 2
            pooled_output = x_trans[:, :, psp_idx:psp_idx + square].reshape([bs, ctx_c, i, i])
            # print("pooled_output", i, pooled_output.shape)  # [4, 256, 1, 1] ,.. 3,3,..6,6,..8,8
            pooled_resized_output = F.interpolate(pooled_output, size=x_context.shape[2:], mode='bilinear',
                                                  align_corners=True)
            psp_cat = paddle.concat([psp_cat, pooled_resized_output], 1)

            psp_idx = psp_idx + square

        # print("psp_cat", psp_cat.shape)  # [4, 1280, 32, 32]
        psp_cat = paddle.concat([psp_cat, x_fpn], 1)
        # print("psp_cat", psp_cat.shape)  # [4, 1536, 32, 32]

        # # x_out = self.cls(x_fpn) #for encoder-fpn
        x_out = self.cls_psp(psp_cat) # 256*6 --> 256
        # x = F.interpolate(x_out, inputs.shape[2:], mode='bilinear', align_corners=True)

        x=self.uphead(x_out)

        # print("x_out",x.shape)#[8, 6, 256, 256]

        outputs = list()
        # outputs.append(x_fpn)
        outputs.append(x)

        auxout = self.auxlayer(c3)
        auxout = F.interpolate(auxout, inputs.shape[2:], mode='bilinear', align_corners=True)
        outputs.append(auxout)

        return tuple(outputs)
