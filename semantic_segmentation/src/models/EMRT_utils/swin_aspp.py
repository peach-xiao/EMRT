import math
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from src.models.backbones.swin_transformer import SwinTransformerStage
from paddleseg.cvlibs import param_init

class ChannelAttention(nn.Layer):
    def __init__(self, channel, reduction):
        super().__init__()
        self.maxpool = nn.AdaptiveMaxPool2D(1)
        self.avgpool = nn.AdaptiveAvgPool2D(1)
        self.se = nn.Sequential(
            nn.Conv2D(channel, channel // reduction, 1, bias_attr=False),
            nn.ReLU(),
            nn.Conv2D(channel // reduction, channel, 1, bias_attr=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_result = self.maxpool(x)
        avg_result = self.avgpool(x)
        max_out = self.se(max_result)
        avg_out = self.se(avg_result)
        output = self.sigmoid(max_out + avg_out)
        return output


class SpatialAttention(nn.Layer):
    def __init__(self, kernel_size):
        super().__init__()
        self.conv = nn.Conv2D(2, 1, kernel_size=kernel_size, padding=kernel_size // 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # max_result, _ = torch.max(x, axis=1, keepdim=True) #torch.max返回最大值和对应的索引
        max_result = paddle.max(x, axis=1, keepdim=True) # padlle直接返回最大值
        avg_result = paddle.mean(x, axis=1, keepdim=True)
        result = paddle.concat([max_result, avg_result], 1)
        output = self.conv(result)
        output = self.sigmoid(output)
        return output


class CBAMBlock(nn.Layer):

    def __init__(self, input_dim, reduction, input_size, out_dim):
        super().__init__()
        self.input_size = input_size
        self.ca = ChannelAttention(channel=input_dim, reduction=reduction)
        self.sa = SpatialAttention(kernel_size=1)
        self.proj = nn.Linear(input_dim, out_dim)


    def forward(self, x):
        # B, L, C = x.shape
        B, C, H, W = x.shape
        # assert L == self.input_size ** 2
        # x = x.permute(0, 2, 1).contiguous()
        # x = x.view(B, C, self.input_size, self.input_size)

        residual = x
        out = x * self.ca(x)
        out = out * self.sa(out)
        out = out + residual

        # out = out.view(B, C, L).permute(0, 2, 1).contiguous()
        out = out.flatten(2).transpose([0, 2, 1])
        return self.proj(out)

class ConvModule(nn.Layer):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(ConvModule, self).__init__()
        self.conv = nn.Conv2D(in_channels, out_channels, kernel_size, stride, padding, bias_attr=False)
        self.norm = nn.BatchNorm2D(out_channels)
        self.relu = nn.ReLU(True)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.relu(x)
        return x

class SwinASPP(nn.Layer):
    def __init__(self,input_size, input_dim, out_dim, config):

        super().__init__()

        self.out_dim = out_dim
        if input_size == 32:
            self.possible_window_sizes = [4, 8, 16, 32]

        norm_bias_attr = paddle.ParamAttr(initializer=paddle.nn.initializer.Constant(0.0))

        self.image_pool = nn.Sequential(
            nn.AdaptiveAvgPool2D(1),
            nn.Conv2D(input_dim, input_dim, 1, stride=1, bias_attr=False),
            nn.BatchNorm2D(input_dim, weight_attr=self.get_norm_weight_attr(), bias_attr=norm_bias_attr),
            nn.ReLU())

        self.conv_path = nn.Sequential(
                nn.Conv2D(input_dim, input_dim, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2D(input_dim, weight_attr=self.get_norm_weight_attr(), bias_attr=norm_bias_attr),
                nn.ReLU())

        self.layers = nn.LayerList()
        for ws in self.possible_window_sizes:
            layer = SwinTransformerStage(dim=int(input_dim),
                               input_resolution=(input_size, input_size),
                               depth=1 if ws == input_size else 2,
                               num_heads= 3,
                               window_size=ws,
                               mlp_ratio=config.MODEL.TRANS.MLP_RATIO,
                               qkv_bias=config.MODEL.TRANS.QKV_BIAS,
                               qk_scale=config.MODEL.TRANS.QK_SCALE,
                               dropout=config.MODEL.DROPOUT,
                               attention_dropout=config.MODEL.ATTENTION_DROPOUT,
                               droppath = 0.1,
                               downsample= None)

            self.layers.append(layer)

        self.cbam = CBAMBlock(input_dim=(len(self.layers) + 2 )* input_dim,
                              reduction=12,
                              input_size=input_size,
                              out_dim=out_dim) #if cross_attn == 'CBAM':

        # self.proj = nn.Linear(len(self.layers) * input_dim, out_dim) #除了CBAM另外的方式

        self.norm = nn.ReLU()
        self.dropout = nn.Dropout(0.1)

    def get_norm_weight_attr(self):
        return paddle.ParamAttr(initializer=paddle.nn.initializer.Constant(value=1.0))

    def to_2D(self, x):
        n, hw, c = x.shape
        h = w = int(math.sqrt(hw))
        x = x.transpose([0, 2, 1]).reshape([n, c, h, w])
        return x

    def forward(self, x):
        """
        x: input tensor (high level features) with shape (batch_size, input_size, input_size, input_dim)

        returns ...
        """
        B, C, H, W = x.shape  #[8, 192, 32, 32]

        src = x.flatten(2).transpose([0, 2, 1])
        # print("src.shape", src)  # [8, 1024, 192]

        # src = x.flatten(start_axis=2, stop_axis=-1)  # [batch, embed_dim, h*w] h*w = num_patches
        # src = src.transpose([0, 2, 1])  # [batch, h*w, embed_dim]
        # src = self.norm(src)  # [batch, num_patches, embed_dim]
        # print("x2.shape", src) #[8, 1024, 192]

        features = []
        features.append(self.conv_path(x))
        img_pool = self.image_pool(x)#[8, 192, 1, 1],
        img_pool = F.interpolate(img_pool, x.shape[2:], mode='bilinear', align_corners=True)#shape=[8, 192, 32, 32],
        features.append(img_pool)

        for layer in self.layers: #每次的输出相同, 4层
            out, _ = layer(src) # -->shape=[8, 1024, 192]
            out = self.to_2D(out)#[8, 192, 32, 32]
            features.append(out)

        features = paddle.concat(features, 1) #[8, 1152, 32, 32], 192x6 = 1152

        features = self.cbam(features)  # cross attention --> cbam
        # print("cbam",features) #shape=[8, 1024, 192]

        features = self.norm(features)

        features = self.dropout(features)

        return features
