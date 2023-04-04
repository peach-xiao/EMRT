import math
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from src.models.backbones.swin_transformer import SwinTransformerStage
from paddleseg.cvlibs import param_init


class ChannelAttention(nn.Layer):
    def __init__(self, channel, reduction = 16):
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
    def __init__(self, kernel_size = 7):
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

    def __init__(self, input_dim, reduction, out_dim = 256):
        super().__init__()
        self.ca = ChannelAttention(channel=input_dim, reduction=reduction)
        self.sa = SpatialAttention(kernel_size=1)
        self.proj = nn.Linear(input_dim, out_dim)


    def forward(self, x):
        # B, L, C = x.shape
        # B, C, H, W = x.shape
        # assert L == self.input_size ** 2
        # x = x.permute(0, 2, 1).contiguous()
        # x = x.view(B, C, self.input_size, self.input_size)

        residual = x
        out = x * self.ca(x)
        out = out * self.sa(out)
        out = out + residual
        return out

        # out = out.view(B, C, L).permute(0, 2, 1).contiguous()
        # out = out.flatten(2).transpose([0, 2, 1])
        # return self.proj(out)


class Patch_Attention(nn.Layer):
    def __init__(self, in_channels, reduction=16, pool_window=4, add_input=True):
        super(Patch_Attention, self).__init__()
        self.pool_window = pool_window
        self.add_input = add_input
        self.se = nn.Sequential(
            nn.Conv2D(in_channels, in_channels // reduction, 1),
            nn.BatchNorm2D(in_channels // reduction, momentum=0.95),
            nn.ReLU(),
            nn.Conv2D(in_channels // reduction, in_channels, 1),
            # nn.Sigmoid()
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.shape

        pool_h = h // self.pool_window
        pool_w = w // self.pool_window

        A = F.adaptive_avg_pool2d(x, (pool_h, pool_w))# [8, 1536, 8, 8]
        B = F.adaptive_avg_pool2d(x, (pool_h, pool_w))
        # print("A",A.shape)# [8, 1536, 8, 8]
        # print("B", B.shape)# [8, 1536, 8, 8]

        A = self.se(A)
        B = self.se(B)

        C = self.sigmoid(A+B)
        C = F.upsample(C, (h, w), mode='bilinear')
        # print("C", C.shape) #[8, 1536, 32, 32]

        output = x * C
        if self.add_input:
            output += x

        return output